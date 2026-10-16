package model

import (
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// dropout is inverted dropout: in training each element is zeroed with
// probability p and the survivors are scaled by 1/(1-p), so the expected
// activation is unchanged and inference needs no rescaling.
//
// The mask (0 or 1/(1-p) per element) is kept from Forward for Backward. It
// is nil when the last Forward did not drop anything, which makes both
// passes the identity.
type dropout struct {
	mask []float64
	buf  []float64 // backing storage reused for mask across steps
}

// forward applies dropout with probability p to x in place when rng is
// non-nil and p > 0. x must be owned by the caller (a fresh sub-layer output).
func (d *dropout) forward(x *tensor.Mat, p float64, rng *rand.Rand) {
	if rng == nil || p <= 0 {
		d.mask = nil
		return
	}
	if cap(d.buf) < len(x.Data) {
		d.buf = make([]float64, len(x.Data))
	}
	d.mask = d.buf[:len(x.Data)]
	// Drawn serially from one rng so a seeded rng gives a reproducible mask.
	scale := 1 / (1 - p)
	for i := range d.mask {
		if rng.Float64() < p {
			d.mask[i] = 0
		} else {
			d.mask[i] = scale
		}
	}
	for i, m := range d.mask {
		x.Data[i] *= m
	}
}

// backward multiplies dy by the stored mask. If inPlace is false and a mask
// is active, a new matrix is returned and dy is left untouched.
func (d *dropout) backward(dy *tensor.Mat, inPlace bool) *tensor.Mat {
	if d.mask == nil {
		return dy
	}
	out := dy
	if !inPlace {
		out = tensor.New(dy.Rows, dy.Cols)
	}
	for i, m := range d.mask {
		out.Data[i] = dy.Data[i] * m
	}
	return out
}
