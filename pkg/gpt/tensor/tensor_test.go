package tensor

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func naiveMatMul(a, b *Mat) *Mat {
	out := New(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			var s float64
			for k := 0; k < a.Cols; k++ {
				s += a.At(i, k) * b.At(k, j)
			}
			out.Set(i, j, s)
		}
	}
	return out
}

func transpose(m *Mat) *Mat {
	t := New(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			t.Set(j, i, m.At(i, j))
		}
	}
	return t
}

func TestMatMulVariantsMatchNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	// Shapes chosen so Parallel takes both the serial and goroutine paths,
	// every kernel sees partial 4x4 tiles on both edges, and the inner
	// dimension is both shorter than and not a multiple of the k slabs
	// MatMulTA walks.
	for _, sh := range [][3]int{{3, 7, 5}, {40, 7, 5}, {101, 7, 9}, {9, 3, 6}, {5, 2*taKBlock + 3, 13}, {6, 300, 101}} {
		n, k, m := sh[0], sh[1], sh[2]
		a := Randn(n, k, 1, rng)
		b := Randn(k, m, 1, rng)
		want := naiveMatMul(a, b).Data
		assert.InDeltaSlice(t, want, MatMul(a, b).Data, 1e-12)

		bt := transpose(b) // [m,k]
		assert.InDeltaSlice(t, want, MatMulT(a, bt).Data, 1e-12)

		at := transpose(a) // [k,n]
		assert.InDeltaSlice(t, want, MatMulTA(at, b).Data, 1e-12)
	}
}

func TestIntoVariantsMatchNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, sh := range [][3]int{{3, 7, 5}, {9, 3, 6}, {66, 2*taKBlock + 3, 13}} {
		n, k, m := sh[0], sh[1], sh[2]
		a := Randn(n, k, 1, rng)
		b := Randn(k, m, 1, rng)
		want := naiveMatMul(a, b).Data

		bt := New(m, k)
		TransposeInto(bt, b)
		assert.Equal(t, transpose(b).Data, bt.Data)
		out := New(n, m)
		MatMulTInto(out, a, bt)
		assert.InDeltaSlice(t, want, out.Data, 1e-12)

		// MatMulTAInto accumulates: two half-k passes must sum to the product.
		at := transpose(a) // [k,n]
		acc := New(n, m)
		h := k / 2
		MatMulTAInto(acc, at.RowRange(0, h), b.RowRange(0, h))
		MatMulTAInto(acc, at.RowRange(h, k), b.RowRange(h, k))
		assert.InDeltaSlice(t, want, acc.Data, 1e-12)
	}
}

func TestRowRangeIsView(t *testing.T) {
	m := New(4, 3)
	v := m.RowRange(1, 3)
	assert.Equal(t, 2, v.Rows)
	assert.Equal(t, 3, v.Cols)
	v.Set(1, 2, 7)
	assert.Equal(t, 7.0, m.At(2, 2))
	assert.Panics(t, func() { m.RowRange(2, 5) })
	assert.Panics(t, func() { TransposeInto(New(3, 3), m) })
	assert.Panics(t, func() { MatMulTInto(New(4, 4), m, New(5, 2)) })
	assert.Panics(t, func() { MatMulTAInto(New(3, 3), m, New(5, 3)) })
}

func TestParallelCoversRangeOnce(t *testing.T) {
	for _, n := range []int{0, 1, 2, 7, 40, 1000} {
		seen := make([]int32, n)
		var mu sync.Mutex
		var calls int
		Parallel(n, func(lo, hi int) {
			mu.Lock()
			calls++
			mu.Unlock()
			require.LessOrEqual(t, lo, hi)
			for i := lo; i < hi; i++ {
				atomic.AddInt32(&seen[i], 1)
			}
		})
		for i, c := range seen {
			assert.EqualValues(t, 1, c, "n=%d index %d", n, i)
		}
		if n < 2 {
			assert.Equal(t, 1, calls)
		}
	}
}

func TestShapePanics(t *testing.T) {
	a, b := New(2, 3), New(2, 3)
	assert.Panics(t, func() { MatMul(a, b) })
	assert.Panics(t, func() { a.AddRow([]float64{1}) })
	assert.Panics(t, func() { FromSlice(2, 2, []float64{1}) })
}

func TestElementwise(t *testing.T) {
	m := FromSlice(2, 2, []float64{1, 2, 3, 4})
	m.AddRow([]float64{10, 20})
	assert.Equal(t, []float64{11, 22, 13, 24}, m.Data)
	assert.Equal(t, []float64{24, 46}, m.ColSum())
	c := m.Clone()
	c.Scale(0)
	assert.Equal(t, []float64{11, 22, 13, 24}, m.Data, "Clone must copy")
	require.InDelta(t, 5.0, FromSlice(1, 2, []float64{3, 4}).Norm(), 1e-12)
}
