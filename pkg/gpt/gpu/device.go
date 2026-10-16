// Package gpu is a float32, device-resident tensor layer for training the
// GPT in pkg/gpt on a GPU through gogpu/wgpu (pure Go WebGPU, no cgo).
//
// It holds the pieces a training step needs and nothing else: a Device that
// owns the WebGPU objects and compiled pipelines, a Tensor that lives in a
// storage buffer, and a CommandBatch that records any number of kernel
// dispatches into one command encoder and submits them once. Every kernel is
// checked against the float64 CPU code in pkg/gpt by the package tests.
//
// The CPU path stays the default; nothing in pkg/gpt depends on this package.
// See readme.md for the design and the constraints a caller must respect.
package gpu

import (
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends" // registers Metal, Vulkan, DX12, GLES and the software fallback
)

// ErrNoAdapter is returned by Open when no hardware GPU adapter is available.
// The software rasterizer that gogpu/wgpu falls back to does not count: it
// would run these kernels far slower than the CPU code they replace.
var ErrNoAdapter = errors.New("gpu: no hardware GPU adapter available")

// uniformSlot is the stride between per-dispatch parameter blocks in the
// uniform arena. WebGPU requires uniform binding offsets to be a multiple of
// minUniformBufferOffsetAlignment, which is 256 on every mainstream device.
const uniformSlot = 256

// arenaSlots is the number of parameter blocks per uniform arena chunk. A
// batch that records more dispatches than this simply spills into another
// chunk.
const arenaSlots = 4096

// maxBindings is the most storage plus uniform bindings any kernel uses.
const maxBindings = 8

// Device owns the WebGPU instance, adapter, device and queue, plus every
// compiled pipeline, the bind-group cache and the uniform arena that per
// dispatch parameters are written to. It is not safe for concurrent use:
// one goroutine records and submits batches.
type Device struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
	info     gputypes.AdapterInfo

	modules   map[string]*wgpu.ShaderModule // by WGSL source
	pipelines map[*kernel]*pipeline
	groups    map[groupKey]*wgpu.BindGroup

	// arena holds the uniform parameter blocks of the batch being recorded:
	// dispatch i of a batch uses slot i. Because a training step records the
	// same dispatches in the same order every time, slot offsets repeat and
	// the bind groups built for one step are cache hits for the next.
	arena   []*wgpu.Buffer
	staging []byte // host copy of the arena, written with one WriteBuffer per chunk at submit

	partials  *wgpu.Buffer // scratch for the first stage of Sum
	recording bool

	mu sync.Mutex // guards Close against double release only
}

// pipeline is a kernel compiled for this device.
type pipeline struct {
	layout   *wgpu.BindGroupLayout
	pl       *wgpu.PipelineLayout
	pipeline *wgpu.ComputePipeline
}

// groupKey identifies a bind group: the pipeline, the buffers bound to its
// storage slots, and the uniform arena chunk and offset of its parameters.
type groupKey struct {
	pipe  *pipeline
	bufs  [maxBindings]*wgpu.Buffer
	arena *wgpu.Buffer
	off   uint64
}

// Open picks the high-performance hardware adapter and creates a device on
// it. It returns ErrNoAdapter (possibly wrapped) when there is none, which
// callers and tests use to fall back to the CPU. Setting BRAIN_GPU_ALLOW_CPU=1
// accepts gogpu/wgpu's software adapter too, for debugging kernels on a
// machine without a GPU.
//
// BRAIN_GPU_DISABLE=1 makes Open report ErrNoAdapter unconditionally, to
// force the CPU path or check that the tests skip cleanly.
func Open() (*Device, error) {
	if os.Getenv("BRAIN_GPU_DISABLE") == "1" {
		return nil, fmt.Errorf("%w: disabled by BRAIN_GPU_DISABLE=1", ErrNoAdapter)
	}
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("%w: CreateInstance: %v", ErrNoAdapter, err)
	}
	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: gputypes.PowerPreferenceHighPerformance})
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("%w: RequestAdapter: %v", ErrNoAdapter, err)
	}
	info := adapter.Info()
	software := info.DeviceType == gputypes.DeviceTypeCPU || info.Backend == gputypes.BackendEmpty
	if software && os.Getenv("BRAIN_GPU_ALLOW_CPU") != "1" {
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("%w: only the software adapter %q (%v) was found", ErrNoAdapter, info.Name, info.Backend)
	}
	// Ask for the adapter's own buffer limits: the defaults cap a storage
	// binding at 128 MiB, which a large vocabulary's logits can exceed.
	limits := gputypes.DefaultLimits()
	al := adapter.Limits()
	limits.MaxBufferSize = max(limits.MaxBufferSize, al.MaxBufferSize)
	limits.MaxStorageBufferBindingSize = max(limits.MaxStorageBufferBindingSize, al.MaxStorageBufferBindingSize)
	limits.MaxStorageBuffersPerShaderStage = max(limits.MaxStorageBuffersPerShaderStage, min(al.MaxStorageBuffersPerShaderStage, maxBindings))
	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{Label: "brain-gpu", RequiredLimits: limits})
	if err != nil {
		// Some backends reject raised limits; the defaults still cover the
		// model shapes this package is tested at.
		device, err = adapter.RequestDevice(nil)
	}
	if err != nil {
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("gpu: RequestDevice on %s: %w", info.Name, err)
	}
	if a := device.Limits().MinUniformBufferOffsetAlignment; a > uniformSlot {
		device.Release()
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("gpu: %s needs uniform offsets aligned to %d bytes, more than %d", info.Name, a, uniformSlot)
	}
	d := &Device{
		instance:  instance,
		adapter:   adapter,
		device:    device,
		queue:     device.Queue(),
		info:      info,
		modules:   map[string]*wgpu.ShaderModule{},
		pipelines: map[*kernel]*pipeline{},
		groups:    map[groupKey]*wgpu.BindGroup{},
	}
	d.partials, err = d.newBuffer("sum-partials", sumMaxPartials*4, 0)
	if err != nil {
		d.Close()
		return nil, err
	}
	return d, nil
}

// Name describes the adapter, for logs and benchmark output.
func (d *Device) Name() string {
	return fmt.Sprintf("%s (%v, %v)", d.info.Name, d.info.Backend, d.info.DeviceType)
}

// Close releases every pipeline, cached bind group and buffer the device
// owns, then the device itself. Tensors must not be used afterwards.
func (d *Device) Close() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.device == nil {
		return
	}
	for k, g := range d.groups {
		g.Release()
		delete(d.groups, k)
	}
	for _, p := range d.pipelines {
		p.pipeline.Release()
		p.pl.Release()
		p.layout.Release()
	}
	for _, m := range d.modules {
		m.Release()
	}
	for _, a := range d.arena {
		a.Release()
	}
	if d.partials != nil {
		d.partials.Release()
	}
	d.device.Release()
	d.adapter.Release()
	d.instance.Release()
	d.device = nil
}

const storageUsage = wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst

// newBuffer creates a zero-filled storage buffer of at least size bytes.
func (d *Device) newBuffer(label string, size uint64, extra gputypes.BufferUsage) (*wgpu.Buffer, error) {
	size = max(4, (size+3)&^3)
	buf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: size, Usage: storageUsage | extra})
	if err != nil {
		return nil, fmt.Errorf("gpu: create %s buffer (%d bytes): %w", label, size, err)
	}
	// WebGPU promises zeroed buffers but not every HAL backend here does;
	// the write is staged and lands before the next submission.
	if err := d.queue.WriteBuffer(buf, 0, make([]byte, size)); err != nil {
		buf.Release()
		return nil, fmt.Errorf("gpu: zero %s buffer: %w", label, err)
	}
	return buf, nil
}

// forget drops the cached bind groups that reference buf, before it is
// released.
func (d *Device) forget(buf *wgpu.Buffer) {
	for k, g := range d.groups {
		for _, b := range k.bufs {
			if b == buf {
				g.Release()
				delete(d.groups, k)
				break
			}
		}
	}
}

// pipelineFor compiles k on first use.
func (d *Device) pipelineFor(k *kernel) (*pipeline, error) {
	if p, ok := d.pipelines[k]; ok {
		return p, nil
	}
	src := k.source()
	mod, ok := d.modules[src]
	if !ok {
		var err error
		mod, err = d.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: k.name, WGSL: src})
		if err != nil {
			return nil, fmt.Errorf("gpu: compile %s: %w", k.name, err)
		}
		d.modules[src] = mod
	}
	entries := make([]gputypes.BindGroupLayoutEntry, 0, len(k.binds)+1)
	for i, c := range k.binds {
		t := gputypes.BufferBindingTypeReadOnlyStorage
		if c == 'w' {
			t = gputypes.BufferBindingTypeStorage
		}
		entries = append(entries, gputypes.BindGroupLayoutEntry{Binding: uint32(i), Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: t}})
	}
	entries = append(entries, gputypes.BindGroupLayoutEntry{Binding: uint32(len(k.binds)), Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}})
	layout, err := d.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Label: k.name, Entries: entries})
	if err != nil {
		return nil, fmt.Errorf("gpu: bind group layout %s: %w", k.name, err)
	}
	pl, err := d.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: k.name, BindGroupLayouts: []*wgpu.BindGroupLayout{layout}})
	if err != nil {
		layout.Release()
		return nil, fmt.Errorf("gpu: pipeline layout %s: %w", k.name, err)
	}
	cp, err := d.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: k.name, Layout: pl, Module: mod, EntryPoint: "main"})
	if err != nil {
		pl.Release()
		layout.Release()
		return nil, fmt.Errorf("gpu: compute pipeline %s: %w", k.name, err)
	}
	p := &pipeline{layout: layout, pl: pl, pipeline: cp}
	d.pipelines[k] = p
	return p, nil
}

// slot returns the arena chunk and byte offset of uniform slot i, growing
// the arena as needed.
func (d *Device) slot(i int) (*wgpu.Buffer, uint64, error) {
	chunk := i / arenaSlots
	for len(d.arena) <= chunk {
		buf, err := d.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "uniform-arena",
			Size:  arenaSlots * uniformSlot,
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return nil, 0, fmt.Errorf("gpu: uniform arena: %w", err)
		}
		d.arena = append(d.arena, buf)
	}
	return d.arena[chunk], uint64(i%arenaSlots) * uniformSlot, nil
}

// bindGroup returns the cached bind group for these buffers and parameter
// slot, creating it on first use.
func (d *Device) bindGroup(p *pipeline, bufs []*wgpu.Buffer, arena *wgpu.Buffer, off uint64) (*wgpu.BindGroup, error) {
	key := groupKey{pipe: p, arena: arena, off: off}
	copy(key.bufs[:], bufs)
	if g, ok := d.groups[key]; ok {
		return g, nil
	}
	entries := make([]wgpu.BindGroupEntry, 0, len(bufs)+1)
	for i, b := range bufs {
		entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b})
	}
	entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(len(bufs)), Buffer: arena, Offset: off, Size: uniformSlot})
	g, err := d.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: p.layout, Entries: entries})
	if err != nil {
		return nil, err
	}
	d.groups[key] = g
	return g, nil
}
