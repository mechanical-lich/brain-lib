package gpu

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/gogpu/wgpu"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// Tensor is a dense row-major float32 matrix resident in a GPU storage
// buffer: the device-side counterpart of tensor.Mat.
type Tensor struct {
	Rows, Cols int

	d   *Device
	buf *wgpu.Buffer
}

// NewTensor allocates a zeroed rows x cols tensor on the device.
func (d *Device) NewTensor(rows, cols int) *Tensor {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("gpu: tensor shape %dx%d", rows, cols))
	}
	buf, err := d.newBuffer("tensor", uint64(rows*cols)*4, 0)
	if err != nil {
		panic(err)
	}
	return &Tensor{Rows: rows, Cols: cols, d: d, buf: buf}
}

// FromMat allocates a tensor shaped like m and uploads it.
func (d *Device) FromMat(m *tensor.Mat) *Tensor {
	t := d.NewTensor(m.Rows, m.Cols)
	t.Upload(m)
	return t
}

// Len is the number of elements.
func (t *Tensor) Len() int { return t.Rows * t.Cols }

// Shape returns a "RxC" string for error messages.
func (t *Tensor) Shape() string { return fmt.Sprintf("%dx%d", t.Rows, t.Cols) }

// Upload copies m, converted to float32, into the tensor. The write is
// staged and lands before the next submission, so a batch that is being
// recorded sees the new values in all of its dispatches.
func (t *Tensor) Upload(m *tensor.Mat) {
	if m.Rows != t.Rows || m.Cols != t.Cols {
		panic(fmt.Sprintf("gpu: Upload %s into %s", m.Shape(), t.Shape()))
	}
	f := make([]float32, len(m.Data))
	for i, v := range m.Data {
		f[i] = float32(v)
	}
	t.UploadF32(f)
}

// UploadF32 copies data (len Rows*Cols) into the tensor.
func (t *Tensor) UploadF32(data []float32) {
	if len(data) != t.Len() {
		panic(fmt.Sprintf("gpu: UploadF32 of %d values into %s", len(data), t.Shape()))
	}
	if err := t.d.queue.WriteBuffer(t.buf, 0, f32Bytes(data)); err != nil {
		panic(fmt.Errorf("gpu: upload: %w", err))
	}
}

// Download copies the tensor into m, converting to float64. It submits and
// waits on its own, so it must not be used inside a training step; see
// CommandBatch.Read for reading back a few values at the end of a step.
func (t *Tensor) Download(m *tensor.Mat) error {
	if m.Rows != t.Rows || m.Cols != t.Cols {
		panic(fmt.Sprintf("gpu: Download %s into %s", t.Shape(), m.Shape()))
	}
	f, err := t.DownloadF32()
	if err != nil {
		return err
	}
	for i, v := range f {
		m.Data[i] = float64(v)
	}
	return nil
}

// ToMat downloads the tensor into a new tensor.Mat.
func (t *Tensor) ToMat() (*tensor.Mat, error) {
	m := tensor.New(t.Rows, t.Cols)
	return m, t.Download(m)
}

// DownloadF32 returns the tensor's contents.
func (t *Tensor) DownloadF32() ([]float32, error) {
	b := t.d.Batch()
	r := b.Read(t)
	if err := b.Submit(); err != nil {
		return nil, err
	}
	return r.Values(), nil
}

// Release frees the tensor's buffer and the bind groups that used it.
func (t *Tensor) Release() {
	if t.buf == nil {
		return
	}
	t.d.forget(t.buf)
	t.buf.Release()
	t.buf = nil
}

// Index is a list of row indices into a table with TableRows rows, as used
// by the embedding lookups: Gather reads table rows by index, and
// ScatterAdd adds gradient rows back into the table rows they came from.
//
// WGSL has no float atomics, so ScatterAdd cannot let rows that share an
// index race. The Index therefore also holds the inverse mapping in CSR
// form (for each table row, the positions that use it), built on the host
// where the ids already are, so each table row is summed by one workgroup
// in a fixed order and the result is deterministic.
type Index struct {
	N, TableRows int

	d         *Device
	ids       *wgpu.Buffer // [N] u32
	offsets   *wgpu.Buffer // [TableRows+1] u32
	positions *wgpu.Buffer // [N] u32
}

// NewIndex uploads ids, each in [0, tableRows).
func (d *Device) NewIndex(ids []int, tableRows int) *Index {
	if len(ids) == 0 || tableRows <= 0 {
		panic(fmt.Sprintf("gpu: NewIndex of %d ids into %d rows", len(ids), tableRows))
	}
	ix := &Index{N: len(ids), TableRows: tableRows, d: d}
	var err error
	if ix.ids, err = d.newBuffer("index-ids", uint64(len(ids))*4, 0); err != nil {
		panic(err)
	}
	if ix.offsets, err = d.newBuffer("index-offsets", uint64(tableRows+1)*4, 0); err != nil {
		panic(err)
	}
	if ix.positions, err = d.newBuffer("index-positions", uint64(len(ids))*4, 0); err != nil {
		panic(err)
	}
	ix.Set(ids)
	return ix
}

// Set replaces the ids (same length) and rebuilds the inverse mapping.
func (ix *Index) Set(ids []int) {
	if len(ids) != ix.N {
		panic(fmt.Sprintf("gpu: Index.Set with %d ids, want %d", len(ids), ix.N))
	}
	u := make([]uint32, len(ids))
	counts := make([]uint32, ix.TableRows+1)
	for i, id := range ids {
		if id < 0 || id >= ix.TableRows {
			panic(fmt.Sprintf("gpu: index %d out of range [0, %d)", id, ix.TableRows))
		}
		u[i] = uint32(id)
		counts[id+1]++
	}
	for i := 1; i < len(counts); i++ {
		counts[i] += counts[i-1]
	}
	pos := make([]uint32, len(ids))
	next := append([]uint32(nil), counts[:ix.TableRows]...)
	for i, id := range ids {
		pos[next[id]] = uint32(i)
		next[id]++
	}
	q := ix.d.queue
	for _, w := range []struct {
		buf  *wgpu.Buffer
		data []uint32
	}{{ix.ids, u}, {ix.offsets, counts}, {ix.positions, pos}} {
		if err := q.WriteBuffer(w.buf, 0, u32Bytes(w.data)); err != nil {
			panic(fmt.Errorf("gpu: upload index: %w", err))
		}
	}
}

// Release frees the index buffers.
func (ix *Index) Release() {
	for _, b := range []*wgpu.Buffer{ix.ids, ix.offsets, ix.positions} {
		if b != nil {
			ix.d.forget(b)
			b.Release()
		}
	}
	ix.ids, ix.offsets, ix.positions = nil, nil, nil
}

func f32Bytes(f []float32) []byte {
	if len(f) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}

func u32Bytes(u []uint32) []byte {
	if len(u) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&u[0])), len(u)*4)
}

func bytesF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	copy(out, unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(out)))
	return out
}

// fbits encodes a float parameter for the uniform block.
func fbits(v float64) uint32 { return math.Float32bits(float32(v)) }

// flag encodes a bool parameter.
func flag(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}
