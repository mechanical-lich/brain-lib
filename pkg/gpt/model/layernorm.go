package model

import (
	"math"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// LayerNorm normalizes each row to zero mean / unit variance, then applies a
// learned per-feature gain and bias.
type LayerNorm struct {
	G   *Param // gain  [1, dim]
	B   *Param // bias  [1, dim]
	Eps float64

	xhat *tensor.Mat // cached normalized input
	rstd []float64   // cached 1/std per row
}

// NewLayerNorm creates a LayerNorm with gain 1 and bias 0.
func NewLayerNorm(name string, dim int) *LayerNorm {
	g := tensor.New(1, dim)
	g.Fill(1)
	return &LayerNorm{
		G:   newParam(name+".g", g, false),
		B:   newParam(name+".b", tensor.New(1, dim), false),
		Eps: 1e-5,
	}
}

// Forward normalizes x [n, dim] row-wise.
func (ln *LayerNorm) Forward(x *tensor.Mat) *tensor.Mat {
	n, d := x.Rows, x.Cols
	ln.xhat = tensor.New(n, d)
	ln.rstd = make([]float64, n)
	y := tensor.New(n, d)
	g, b := ln.G.Value.Data, ln.B.Value.Data
	tensor.Parallel(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := x.Row(i)
			var mean float64
			for _, v := range row {
				mean += v
			}
			mean /= float64(d)
			var variance float64
			for _, v := range row {
				dv := v - mean
				variance += dv * dv
			}
			variance /= float64(d)
			rstd := 1 / math.Sqrt(variance+ln.Eps)
			ln.rstd[i] = rstd
			xh := ln.xhat.Row(i)
			yr := y.Row(i)
			for j, v := range row {
				xh[j] = (v - mean) * rstd
				yr[j] = xh[j]*g[j] + b[j]
			}
		}
	})
	return y
}

// Backward accumulates dG, dB and returns dx given dy [n, dim].
func (ln *LayerNorm) Backward(dy *tensor.Mat) *tensor.Mat {
	n, d := dy.Rows, dy.Cols
	dx := tensor.New(n, d)
	g := ln.G.Value.Data
	// dg and db are sums over rows, so each chunk accumulates its own and
	// parallelRows folds them into the parameter gradients afterwards.
	parallelRows(n, d, [][]float64{ln.G.Grad.Data, ln.B.Grad.Data}, func(lo, hi int, acc [][]float64) {
		dg, db := acc[0], acc[1]
		for i := lo; i < hi; i++ {
			dyr := dy.Row(i)
			xh := ln.xhat.Row(i)
			// dxhat = dy * g; dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
			var sumDxhat, sumDxhatXhat float64
			for j, v := range dyr {
				dg[j] += v * xh[j]
				db[j] += v
				dxh := v * g[j]
				sumDxhat += dxh
				sumDxhatXhat += dxh * xh[j]
			}
			meanDxhat := sumDxhat / float64(d)
			meanDxhatXhat := sumDxhatXhat / float64(d)
			dxr := dx.Row(i)
			for j, v := range dyr {
				dxh := v * g[j]
				dxr[j] = ln.rstd[i] * (dxh - meanDxhat - xh[j]*meanDxhatXhat)
			}
		}
	})
	return dx
}

// Params lists the trainable parameters.
func (ln *LayerNorm) Params() []*Param { return []*Param{ln.G, ln.B} }
