// Package tensor is a tiny, dependency-free float64 matrix library: just the
// handful of row-major 2D operations a transformer forward/backward pass needs.
//
// Everything is deliberately simple and explicit. Higher-rank data (batch,
// sequence, heads) is handled by the callers flattening into 2D and indexing
// by hand, which keeps the gradient code easy to read and verify.
package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
)

// Mat is a dense row-major matrix.
type Mat struct {
	Rows, Cols int
	Data       []float64
}

// New returns a zeroed rows x cols matrix.
func New(rows, cols int) *Mat {
	if rows < 0 || cols < 0 {
		panic(fmt.Sprintf("tensor: negative shape %dx%d", rows, cols))
	}
	return &Mat{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// FromSlice wraps data (not copied) as a rows x cols matrix.
func FromSlice(rows, cols int, data []float64) *Mat {
	if len(data) != rows*cols {
		panic(fmt.Sprintf("tensor: %d values do not fit %dx%d", len(data), rows, cols))
	}
	return &Mat{Rows: rows, Cols: cols, Data: data}
}

// Randn returns a matrix of N(0, std^2) samples.
func Randn(rows, cols int, std float64, rng *rand.Rand) *Mat {
	m := New(rows, cols)
	for i := range m.Data {
		m.Data[i] = rng.NormFloat64() * std
	}
	return m
}

// At returns element (i, j).
func (m *Mat) At(i, j int) float64 { return m.Data[i*m.Cols+j] }

// Set assigns element (i, j).
func (m *Mat) Set(i, j int, v float64) { m.Data[i*m.Cols+j] = v }

// Row returns row i as a slice view (not a copy).
func (m *Mat) Row(i int) []float64 { return m.Data[i*m.Cols : (i+1)*m.Cols] }

// RowRange returns rows [i, j) as a matrix view sharing m's storage.
func (m *Mat) RowRange(i, j int) *Mat {
	if i < 0 || j < i || j > m.Rows {
		panic(fmt.Sprintf("tensor: RowRange [%d, %d) of %s", i, j, m.Shape()))
	}
	return &Mat{Rows: j - i, Cols: m.Cols, Data: m.Data[i*m.Cols : j*m.Cols : j*m.Cols]}
}

// Clone returns a deep copy.
func (m *Mat) Clone() *Mat {
	c := New(m.Rows, m.Cols)
	copy(c.Data, m.Data)
	return c
}

// Zero sets every element to 0.
func (m *Mat) Zero() {
	for i := range m.Data {
		m.Data[i] = 0
	}
}

// Fill sets every element to v.
func (m *Mat) Fill(v float64) {
	for i := range m.Data {
		m.Data[i] = v
	}
}

// Scale multiplies every element by s in place.
func (m *Mat) Scale(s float64) {
	for i := range m.Data {
		m.Data[i] *= s
	}
}

// Add adds o to m in place.
func (m *Mat) Add(o *Mat) {
	m.mustMatch(o)
	for i, v := range o.Data {
		m.Data[i] += v
	}
}

// AddScaled adds s*o to m in place.
func (m *Mat) AddScaled(o *Mat, s float64) {
	m.mustMatch(o)
	for i, v := range o.Data {
		m.Data[i] += s * v
	}
}

// AddRow adds the vector v to every row (bias broadcast).
func (m *Mat) AddRow(v []float64) {
	if len(v) != m.Cols {
		panic(fmt.Sprintf("tensor: AddRow len %d vs cols %d", len(v), m.Cols))
	}
	for i := 0; i < m.Rows; i++ {
		row := m.Row(i)
		for j, x := range v {
			row[j] += x
		}
	}
}

// ColSum returns the sum over rows of each column.
func (m *Mat) ColSum() []float64 {
	out := make([]float64, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j, x := range m.Row(i) {
			out[j] += x
		}
	}
	return out
}

// Norm returns the Frobenius norm.
func (m *Mat) Norm() float64 {
	var s float64
	for _, v := range m.Data {
		s += v * v
	}
	return math.Sqrt(s)
}

// SameShape reports whether m and o have equal dimensions.
func (m *Mat) SameShape(o *Mat) bool { return m.Rows == o.Rows && m.Cols == o.Cols }

// Shape returns a "RxC" string for error messages.
func (m *Mat) Shape() string { return fmt.Sprintf("%dx%d", m.Rows, m.Cols) }

func (m *Mat) mustMatch(o *Mat) {
	if !m.SameShape(o) {
		panic(fmt.Sprintf("tensor: shape mismatch %s vs %s", m.Shape(), o.Shape()))
	}
}

// Dot returns the inner product of two equal-length vectors.
func Dot(a, b []float64) float64 {
	var s float64
	for i, x := range a {
		s += x * b[i]
	}
	return s
}

// Parallel splits [0, n) into contiguous chunks and runs fn on each chunk,
// using goroutines when n is large enough to be worth it.
//
// Chunks are handed out dynamically rather than split evenly up front: on
// machines that mix fast and slow cores an equal split leaves the fast ones
// idle while the slow ones finish, so each worker keeps pulling the next
// chunk until none are left.
func Parallel(n int, fn func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if n < 2 || workers <= 1 {
		fn(0, n)
		return
	}
	const chunksPerWorker = 4
	chunks := min(n, chunksPerWorker*workers)
	chunk := (n + chunks - 1) / chunks
	workers = min(workers, chunks)
	var next atomic.Int64
	work := func() {
		for {
			lo := int(next.Add(1)-1) * chunk
			if lo >= n {
				return
			}
			fn(lo, min(lo+chunk, n))
		}
	}
	var wg sync.WaitGroup
	for w := 1; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			work()
		}()
	}
	work()
	wg.Wait()
}
