package model

import (
	"math"
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// MLP is the position-wise feed-forward network: D -> 4D -> GELU -> D.
type MLP struct {
	FC   *Linear
	Proj *Linear

	pre *tensor.Mat // cached pre-activation for GELU backward
}

// NewMLP creates an MLP for an EmbedDim-wide stream.
func NewMLP(name string, cfg Config, rng *rand.Rand) *MLP {
	d := cfg.EmbedDim
	projStd := 0.02 / math.Sqrt(2*float64(cfg.NumLayers))
	return &MLP{
		FC:   NewLinear(name+".fc", d, 4*d, 0.02, rng),
		Proj: NewLinear(name+".proj", 4*d, d, projStd, rng),
	}
}

// Forward applies the MLP to x [n, D].
func (m *MLP) Forward(x *tensor.Mat) *tensor.Mat {
	m.pre = m.FC.Forward(x)
	act := tensor.New(m.pre.Rows, m.pre.Cols)
	// Element-wise, so split the flat data across workers in row chunks.
	cols := m.pre.Cols
	tensor.Parallel(m.pre.Rows, func(lo, hi int) {
		src, dst := m.pre.Data[lo*cols:hi*cols], act.Data[lo*cols:hi*cols]
		for i, v := range src {
			dst[i] = gelu(v)
		}
	})
	return m.Proj.Forward(act)
}

// Backward returns dx given dy [n, D].
func (m *MLP) Backward(dy *tensor.Mat) *tensor.Mat {
	dAct := m.Proj.Backward(dy)
	cols := m.pre.Cols
	tensor.Parallel(m.pre.Rows, func(lo, hi int) {
		pre, d := m.pre.Data[lo*cols:hi*cols], dAct.Data[lo*cols:hi*cols]
		for i, v := range pre {
			d[i] *= geluGrad(v)
		}
	})
	return m.FC.Backward(dAct)
}

// Params lists the trainable parameters.
func (m *MLP) Params() []*Param { return append(m.FC.Params(), m.Proj.Params()...) }

const (
	geluC = 0.7978845608028654 // sqrt(2/pi)
	geluK = 0.044715
)

// gelu is the tanh approximation of the Gaussian Error Linear Unit (as used
// by GPT-2).
func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(geluC*(x+geluK*x*x*x)))
}

func geluGrad(x float64) float64 {
	u := geluC * (x + geluK*x*x*x)
	t := math.Tanh(u)
	du := geluC * (1 + 3*geluK*x*x)
	return 0.5*(1+t) + 0.5*x*(1-t*t)*du
}
