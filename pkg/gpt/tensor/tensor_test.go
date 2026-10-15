package tensor

import (
	"math/rand"
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
	// sizes chosen so Parallel takes both the serial and goroutine paths
	for _, n := range []int{3, 40} {
		a := Randn(n, 7, 1, rng)
		b := Randn(7, 5, 1, rng)
		assert.InDeltaSlice(t, naiveMatMul(a, b).Data, MatMul(a, b).Data, 1e-12)

		bt := transpose(b) // [5,7]
		assert.InDeltaSlice(t, naiveMatMul(a, b).Data, MatMulT(a, bt).Data, 1e-12)

		at := transpose(a) // [7,n]
		assert.InDeltaSlice(t, naiveMatMul(a, b).Data, MatMulTA(at, b).Data, 1e-12)
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
