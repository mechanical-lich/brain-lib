package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// scalarAttention is the original per-position implementation of
// CausalSelfAttention, kept as the reference the tiled version must match:
// for every query it dots q against each earlier k, softmaxes, and
// accumulates v, with the backward pass mirroring that.
type scalarAttention struct {
	a    *CausalSelfAttention
	b, t int
	qkv  *tensor.Mat
	att  []float64 // B*H*T*T, index ((b*H + h)*T + t)*T + s
}

func (r *scalarAttention) forward(x *tensor.Mat, batch, seq int) *tensor.Mat {
	a := r.a
	d := x.Cols
	h, hs := a.Heads, a.headDim
	r.b, r.t = batch, seq
	r.qkv = a.QKV.Forward(x)
	r.att = make([]float64, batch*h*seq*seq)
	y := tensor.New(batch*seq, d)
	scale := 1 / math.Sqrt(float64(hs))
	scores := make([]float64, seq)
	for bh := 0; bh < batch*h; bh++ {
		b, head := bh/h, bh%h
		qOff, kOff, vOff := head*hs, d+head*hs, 2*d+head*hs
		for t := 0; t < seq; t++ {
			row := b*seq + t
			q := r.qkv.Row(row)[qOff : qOff+hs]
			maxv := math.Inf(-1)
			for s := 0; s <= t; s++ {
				k := r.qkv.Row(b*seq + s)[kOff : kOff+hs]
				scores[s] = tensor.Dot(q, k) * scale
				maxv = max(maxv, scores[s])
			}
			var sum float64
			for s := 0; s <= t; s++ {
				scores[s] = math.Exp(scores[s] - maxv)
				sum += scores[s]
			}
			attRow := r.att[(bh*seq+t)*seq : (bh*seq+t+1)*seq]
			out := y.Row(row)[qOff : qOff+hs]
			for s := 0; s <= t; s++ {
				w := scores[s] / sum
				attRow[s] = w
				v := r.qkv.Row(b*seq + s)[vOff : vOff+hs]
				for j := range out {
					out[j] += w * v[j]
				}
			}
		}
	}
	return a.Proj.Forward(y)
}

func (r *scalarAttention) backward(dy *tensor.Mat) *tensor.Mat {
	a := r.a
	d := dy.Cols
	h, hs := a.Heads, a.headDim
	batch, seq := r.b, r.t
	scale := 1 / math.Sqrt(float64(hs))
	dAttnOut := a.Proj.Backward(dy)
	dqkv := tensor.New(batch*seq, 3*d)
	dAtt := make([]float64, seq)
	for bh := 0; bh < batch*h; bh++ {
		b, head := bh/h, bh%h
		qOff, kOff, vOff := head*hs, d+head*hs, 2*d+head*hs
		for t := 0; t < seq; t++ {
			row := b*seq + t
			dOut := dAttnOut.Row(row)[qOff : qOff+hs]
			attRow := r.att[(bh*seq+t)*seq : (bh*seq+t+1)*seq]
			q := r.qkv.Row(row)[qOff : qOff+hs]
			var dot float64
			for s := 0; s <= t; s++ {
				v := r.qkv.Row(b*seq + s)[vOff : vOff+hs]
				dv := dqkv.Row(b*seq + s)[vOff : vOff+hs]
				dAtt[s] = tensor.Dot(dOut, v)
				w := attRow[s]
				for j := range dv {
					dv[j] += w * dOut[j]
				}
				dot += w * dAtt[s]
			}
			dq := dqkv.Row(row)[qOff : qOff+hs]
			for s := 0; s <= t; s++ {
				ds := attRow[s] * (dAtt[s] - dot) * scale
				k := r.qkv.Row(b*seq + s)[kOff : kOff+hs]
				dk := dqkv.Row(b*seq + s)[kOff : kOff+hs]
				for j := range dq {
					dq[j] += ds * k[j]
					dk[j] += ds * q[j]
				}
			}
		}
	}
	return a.QKV.Backward(dqkv)
}

// TestAttentionMatchesScalar checks the tiled attention against the scalar
// reference, forward and backward, at sequence lengths that exercise one
// partial block, several whole blocks, and a ragged last block.
func TestAttentionMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	cfg := Config{VocabSize: 7, ContextLen: 512, NumLayers: 2, NumHeads: 3, EmbedDim: 24}
	for _, seq := range []int{1, 5, attRows, 2 * attRows, 2*attRows + 7} {
		a := NewCausalSelfAttention("attn", cfg, rng)
		for _, p := range a.Params() {
			p.Value.Scale(10)
		}
		ref := &scalarAttention{a: a}
		batch := 2
		x := tensor.Randn(batch*seq, cfg.EmbedDim, 1, rng)
		dy := tensor.Randn(batch*seq, cfg.EmbedDim, 1, rng)

		wantY := ref.forward(x, batch, seq)
		wantDx := ref.backward(dy)
		var wantGrads []*tensor.Mat
		for _, p := range a.Params() {
			wantGrads = append(wantGrads, p.Grad.Clone())
			p.ZeroGrad()
		}

		gotY := a.Forward(x, batch, seq)
		gotDx := a.Backward(dy)
		require.Equal(t, wantY.Shape(), gotY.Shape())
		assert.InDeltaSlice(t, wantY.Data, gotY.Data, 1e-12, "seq %d forward", seq)
		assert.InDeltaSlice(t, wantDx.Data, gotDx.Data, 1e-12, "seq %d dx", seq)
		for i, p := range a.Params() {
			assert.InDeltaSlice(t, wantGrads[i].Data, p.Grad.Data, 1e-12, "seq %d grad %s", seq, p.Name)
		}
	}
}
