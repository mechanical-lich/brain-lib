package model

import (
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// Linear is an affine map y = x @ W + b with W [in, out] and b [1, out].
type Linear struct {
	W *Param
	B *Param

	x *tensor.Mat // cached input for Backward
}

// NewLinear creates a Linear with N(0, std^2) weights and zero bias.
func NewLinear(name string, in, out int, std float64, rng *rand.Rand) *Linear {
	return &Linear{
		W: newParam(name+".w", tensor.Randn(in, out, std, rng), true),
		B: newParam(name+".b", tensor.New(1, out), false),
	}
}

// Forward computes x @ W + b for x [n, in].
func (l *Linear) Forward(x *tensor.Mat) *tensor.Mat {
	l.x = x
	y := tensor.MatMul(x, l.W.Value)
	y.AddRow(l.B.Value.Data)
	return y
}

// Backward accumulates dW, db and returns dx given dy [n, out].
func (l *Linear) Backward(dy *tensor.Mat) *tensor.Mat {
	l.W.Grad.Add(tensor.MatMulTA(l.x, dy))
	// db += column sums of dy, accumulated per chunk and folded in order.
	parallelRows(dy.Rows, dy.Cols, [][]float64{l.B.Grad.Data}, func(lo, hi int, acc [][]float64) {
		bg := acc[0]
		for i := lo; i < hi; i++ {
			for j, v := range dy.Row(i) {
				bg[j] += v
			}
		}
	})
	return tensor.MatMulT(dy, l.W.Value)
}

// Params lists the trainable parameters.
func (l *Linear) Params() []*Param { return []*Param{l.W, l.B} }
