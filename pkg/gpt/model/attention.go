package model

import (
	"math"
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// CausalSelfAttention is multi-head self-attention where each position may
// only attend to itself and earlier positions.
//
// Activations are stored as [B*T, D] matrices (row b*T + t is token t of
// sequence b); the per-sequence, per-head structure is handled by explicit
// indexing rather than reshapes.
type CausalSelfAttention struct {
	QKV   *Linear // D -> 3D, columns laid out as [q heads | k heads | v heads]
	Proj  *Linear // D -> D
	Heads int

	// caches for Backward
	b, t    int
	headDim int
	qkv     *tensor.Mat // [B*T, 3D]
	att     []float64   // softmax weights, B*H*T*T, index ((b*H + h)*T + t)*T + s
}

// NewCausalSelfAttention creates attention for an EmbedDim-wide stream.
func NewCausalSelfAttention(name string, cfg Config, rng *rand.Rand) *CausalSelfAttention {
	d := cfg.EmbedDim
	// Output projections are scaled down by 1/sqrt(2*layers) as in GPT-2 so
	// the residual stream variance does not grow with depth.
	projStd := 0.02 / math.Sqrt(2*float64(cfg.NumLayers))
	return &CausalSelfAttention{
		QKV:     NewLinear(name+".qkv", d, 3*d, 0.02, rng),
		Proj:    NewLinear(name+".proj", d, d, projStd, rng),
		Heads:   cfg.NumHeads,
		headDim: d / cfg.NumHeads,
	}
}

// Forward runs attention on x [B*T, D] with the given batch size and
// sequence length.
func (a *CausalSelfAttention) Forward(x *tensor.Mat, batch, seq int) *tensor.Mat {
	d := x.Cols
	h, hs := a.Heads, a.headDim
	a.b, a.t = batch, seq
	a.qkv = a.QKV.Forward(x)
	a.att = make([]float64, batch*h*seq*seq)
	y := tensor.New(batch*seq, d)
	scale := 1 / math.Sqrt(float64(hs))

	tensor.Parallel(batch*h, func(lo, hi int) {
		scores := make([]float64, seq)
		for bh := lo; bh < hi; bh++ {
			b, head := bh/h, bh%h
			qOff, kOff, vOff := head*hs, d+head*hs, 2*d+head*hs
			for t := 0; t < seq; t++ {
				row := b*seq + t
				q := a.qkv.Row(row)[qOff : qOff+hs]
				// scores over s <= t, numerically stable softmax
				maxv := math.Inf(-1)
				for s := 0; s <= t; s++ {
					k := a.qkv.Row(b*seq + s)[kOff : kOff+hs]
					scores[s] = tensor.Dot(q, k) * scale
					if scores[s] > maxv {
						maxv = scores[s]
					}
				}
				var sum float64
				for s := 0; s <= t; s++ {
					scores[s] = math.Exp(scores[s] - maxv)
					sum += scores[s]
				}
				attRow := a.att[(bh*seq+t)*seq : (bh*seq+t+1)*seq]
				out := y.Row(row)[qOff : qOff+hs]
				for s := 0; s <= t; s++ {
					w := scores[s] / sum
					attRow[s] = w
					v := a.qkv.Row(b*seq + s)[vOff : vOff+hs]
					for j := range out {
						out[j] += w * v[j]
					}
				}
				// attRow[s] for s > t stays 0 (causal mask)
			}
		}
	})
	return a.Proj.Forward(y)
}

// Backward returns dx [B*T, D] given dy [B*T, D] and accumulates parameter
// gradients.
func (a *CausalSelfAttention) Backward(dy *tensor.Mat) *tensor.Mat {
	d := dy.Cols
	h, hs := a.Heads, a.headDim
	batch, seq := a.b, a.t
	scale := 1 / math.Sqrt(float64(hs))

	dAttnOut := a.Proj.Backward(dy) // [B*T, D], gradient w.r.t. concatenated heads
	dqkv := tensor.New(batch*seq, 3*d)

	tensor.Parallel(batch*h, func(lo, hi int) {
		dAtt := make([]float64, seq)
		for bh := lo; bh < hi; bh++ {
			b, head := bh/h, bh%h
			qOff, kOff, vOff := head*hs, d+head*hs, 2*d+head*hs
			for t := 0; t < seq; t++ {
				row := b*seq + t
				dOut := dAttnOut.Row(row)[qOff : qOff+hs]
				attRow := a.att[(bh*seq+t)*seq : (bh*seq+t+1)*seq]
				q := a.qkv.Row(row)[qOff : qOff+hs]

				// dAtt[s] = dOut . v_s ; dV_s += att[s] * dOut
				var dot float64 // sum_s att[s] * dAtt[s], for softmax backward
				for s := 0; s <= t; s++ {
					v := a.qkv.Row(b*seq + s)[vOff : vOff+hs]
					dv := dqkv.Row(b*seq + s)[vOff : vOff+hs]
					dAtt[s] = tensor.Dot(dOut, v)
					w := attRow[s]
					for j := range dv {
						dv[j] += w * dOut[j]
					}
					dot += w * dAtt[s]
				}
				// softmax backward: dScore[s] = att[s] * (dAtt[s] - dot), then
				// score = scale * q.k  =>  dq += scale*dScore*k ; dk += scale*dScore*q
				dq := dqkv.Row(row)[qOff : qOff+hs]
				for s := 0; s <= t; s++ {
					ds := attRow[s] * (dAtt[s] - dot) * scale
					if ds == 0 {
						continue
					}
					k := a.qkv.Row(b*seq + s)[kOff : kOff+hs]
					dk := dqkv.Row(b*seq + s)[kOff : kOff+hs]
					for j := range dq {
						dq[j] += ds * k[j]
						dk[j] += ds * q[j]
					}
				}
			}
		}
	})
	return a.QKV.Backward(dqkv)
}

// Params lists the trainable parameters.
func (a *CausalSelfAttention) Params() []*Param {
	return append(a.QKV.Params(), a.Proj.Params()...)
}
