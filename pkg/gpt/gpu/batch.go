package gpu

import (
	"context"
	"encoding/binary"
	"fmt"
	"time"

	"github.com/gogpu/wgpu"
)

// CommandBatch records kernel dispatches into a single command encoder and
// compute pass and submits them with one queue submission. A training step
// is meant to be one batch: on Vulkan every extra submit-and-wait costs
// about 3 ms, more than most of the kernels in the step.
//
// Dispatches run in the order they were recorded and each one sees the
// writes of the ones before it (the HAL inserts a compute-to-compute
// barrier after every dispatch). Kernel methods panic on shape mismatches,
// like pkg/gpt/tensor; GPU-side errors are reported by Submit.
//
// Only one batch may be recording at a time per Device.
type CommandBatch struct {
	d     *Device
	enc   *wgpu.CommandEncoder
	pass  *wgpu.ComputePassEncoder
	slots int // uniform slots used so far
	reads []*Readback
	err   error
	done  bool
}

// Readback is a tensor copy requested with CommandBatch.Read. Its values
// are available once the batch has been submitted.
type Readback struct {
	t      *Tensor
	n      int
	values []float32
}

// Values returns the values read back, or nil before Submit succeeded.
func (r *Readback) Values() []float32 { return r.values }

// Batch starts recording a new batch.
func (d *Device) Batch() *CommandBatch {
	if d.recording {
		panic("gpu: Batch while another batch is still recording")
	}
	d.recording = true
	return &CommandBatch{d: d}
}

// Run records a batch with fn and submits it.
func (d *Device) Run(fn func(b *CommandBatch)) error {
	b := d.Batch()
	fn(b)
	return b.Submit()
}

// Read schedules a copy of t to the host after every dispatch in the batch.
// Readback is slow (about 1 GB/s measured), so a training step should read
// only scalars such as the loss and the gradient norm.
func (b *CommandBatch) Read(t *Tensor) *Readback {
	return b.ReadN(t, t.Len())
}

// ReadN is Read limited to the first n elements of t.
func (b *CommandBatch) ReadN(t *Tensor, n int) *Readback {
	if n <= 0 || n > t.Len() {
		panic(fmt.Sprintf("gpu: ReadN %d of %s", n, t.Shape()))
	}
	r := &Readback{t: t, n: n}
	b.reads = append(b.reads, r)
	return r
}

// dispatch records one kernel launch of groups workgroups with params as
// its uniform block and bufs bound to its storage slots in order.
func (b *CommandBatch) dispatch(k *kernel, groups [3]uint32, params []uint32, bufs ...*wgpu.Buffer) {
	if b.done {
		panic("gpu: dispatch on a submitted batch")
	}
	if len(bufs) != len(k.binds) {
		panic(fmt.Sprintf("gpu: %s takes %d buffers, got %d", k.name, len(k.binds), len(bufs)))
	}
	if len(params)*4 > uniformSlot {
		panic(fmt.Sprintf("gpu: %s parameters exceed %d bytes", k.name, uniformSlot))
	}
	for i, buf := range bufs {
		if buf == nil {
			panic(fmt.Sprintf("gpu: %s binding %d is a released tensor", k.name, i))
		}
	}
	if b.err != nil || groups[0] == 0 || groups[1] == 0 || groups[2] == 0 {
		return
	}
	d := b.d
	p, err := d.pipelineFor(k)
	if err != nil {
		b.err = err
		return
	}
	arena, off, err := d.slot(b.slots)
	if err != nil {
		b.err = err
		return
	}
	need := (b.slots + 1) * uniformSlot
	if len(d.staging) < need {
		d.staging = append(d.staging, make([]byte, need-len(d.staging))...)
	}
	blk := d.staging[b.slots*uniformSlot : need]
	clear(blk)
	for i, w := range params {
		binary.LittleEndian.PutUint32(blk[4*i:], w)
	}
	b.slots++
	g, err := d.bindGroup(p, bufs, arena, off)
	if err != nil {
		b.err = fmt.Errorf("gpu: bind group for %s: %w", k.name, err)
		return
	}
	if b.pass == nil {
		if b.enc, err = d.device.CreateCommandEncoder(nil); err != nil {
			b.err = err
			return
		}
		if b.pass, err = b.enc.BeginComputePass(nil); err != nil {
			b.err = err
			return
		}
	}
	b.pass.SetPipeline(p.pipeline)
	b.pass.SetBindGroup(0, g, nil)
	b.pass.Dispatch(groups[0], groups[1], groups[2])
}

// Submit ends recording, submits everything as one command buffer, waits
// for the GPU to finish and resolves the batch's readbacks.
func (b *CommandBatch) Submit() error {
	if b.done {
		return fmt.Errorf("gpu: batch submitted twice")
	}
	b.done = true
	d := b.d
	defer func() { d.recording = false }()
	if b.err != nil {
		if b.enc != nil {
			b.enc.DiscardEncoding()
		}
		return b.err
	}
	var err error
	if b.pass != nil {
		if err = b.pass.End(); err != nil {
			return fmt.Errorf("gpu: end compute pass: %w", err)
		}
	} else if len(b.reads) == 0 {
		return nil
	} else if b.enc, err = d.device.CreateCommandEncoder(nil); err != nil {
		return err
	}
	var staging []*wgpu.Buffer
	defer func() {
		for _, s := range staging {
			s.Release()
		}
	}()
	for _, r := range b.reads {
		size := uint64(r.n) * 4
		s, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "readback", Size: max(4, size), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			return fmt.Errorf("gpu: readback buffer: %w", err)
		}
		staging = append(staging, s)
		b.enc.CopyBufferToBuffer(r.t.buf, 0, s, 0, size)
	}
	cb, err := b.enc.Finish()
	if err != nil {
		return fmt.Errorf("gpu: finish command buffer: %w", err)
	}
	// One WriteBuffer per arena chunk carries every dispatch's parameters.
	for i := 0; i*arenaSlots < b.slots; i++ {
		lo := i * arenaSlots * uniformSlot
		hi := min(b.slots, (i+1)*arenaSlots) * uniformSlot
		if err := d.queue.WriteBuffer(d.arena[i], 0, d.staging[lo:hi]); err != nil {
			return fmt.Errorf("gpu: write parameters: %w", err)
		}
	}
	if _, err := d.queue.Submit(cb); err != nil {
		return fmt.Errorf("gpu: submit: %w", err)
	}
	if err := d.device.WaitIdle(); err != nil {
		return fmt.Errorf("gpu: wait: %w", err)
	}
	for i, r := range b.reads {
		size := uint64(r.n) * 4
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		err := staging[i].Map(ctx, wgpu.MapModeRead, 0, size)
		cancel()
		if err != nil {
			return fmt.Errorf("gpu: map readback: %w", err)
		}
		mr, err := staging[i].MappedRange(0, size)
		if err != nil {
			return err
		}
		r.values = bytesF32(mr.Bytes())
		if err := staging[i].Unmap(); err != nil {
			return err
		}
	}
	return nil
}
