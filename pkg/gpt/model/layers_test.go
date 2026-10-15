package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// checkInputGrad verifies d(sum(w * f(x)))/dx by finite differences for a
// layer exposing Forward/Backward over a [n, d] input.
func checkInputGrad(t *testing.T, name string, x *tensor.Mat, fwd func(*tensor.Mat) *tensor.Mat, bwd func(*tensor.Mat) *tensor.Mat, rng *rand.Rand) {
	t.Helper()
	y := fwd(x)
	w := tensor.Randn(y.Rows, y.Cols, 1, rng)
	loss := func(m *tensor.Mat) float64 { return tensor.Dot(fwd(m).Data, w.Data) }
	fwd(x)
	dx := bwd(w.Clone())
	const eps = 1e-6
	for k := 0; k < 12; k++ {
		idx := rng.Intn(len(x.Data))
		orig := x.Data[idx]
		x.Data[idx] = orig + eps
		lp := loss(x)
		x.Data[idx] = orig - eps
		lm := loss(x)
		x.Data[idx] = orig
		num := (lp - lm) / (2 * eps)
		assert.InDeltaf(t, num, dx.Data[idx], 1e-6+1e-5*math.Abs(num), "%s dx[%d]", name, idx)
	}
}

func TestLayerNormInputGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	ln := NewLayerNorm("ln", 6)
	ln.G.Value = tensor.Randn(1, 6, 1, rng)
	ln.B.Value = tensor.Randn(1, 6, 1, rng)
	x := tensor.Randn(4, 6, 2, rng)
	checkInputGrad(t, "layernorm", x, ln.Forward, ln.Backward, rng)
}

func TestLayerNormNormalizes(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	ln := NewLayerNorm("ln", 8)
	y := ln.Forward(tensor.Randn(3, 8, 3, rng))
	for i := 0; i < y.Rows; i++ {
		row := y.Row(i)
		var mean, v float64
		for _, x := range row {
			mean += x
		}
		mean /= 8
		for _, x := range row {
			v += (x - mean) * (x - mean)
		}
		assert.InDelta(t, 0, mean, 1e-12)
		assert.InDelta(t, 1, v/8, 1e-4)
	}
}

func TestLinearInputGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	l := NewLinear("lin", 5, 3, 1, rng)
	checkInputGrad(t, "linear", tensor.Randn(4, 5, 1, rng), l.Forward, l.Backward, rng)
}

func TestMLPInputGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	m := NewMLP("mlp", tinyCfg, rng)
	for _, p := range m.Params() {
		p.Value.Scale(20)
	}
	checkInputGrad(t, "mlp", tensor.Randn(3, tinyCfg.EmbedDim, 1, rng), m.Forward, m.Backward, rng)
}

func TestAttentionInputGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	a := NewCausalSelfAttention("attn", tinyCfg, rng)
	for _, p := range a.Params() {
		p.Value.Scale(20)
	}
	batch, seq := 2, 3
	x := tensor.Randn(batch*seq, tinyCfg.EmbedDim, 1, rng)
	checkInputGrad(t, "attention", x,
		func(m *tensor.Mat) *tensor.Mat { return a.Forward(m, batch, seq) },
		a.Backward, rng)
}

func TestBlockInputGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	b := NewBlock(0, tinyCfg, rng)
	for _, p := range b.Params() {
		if p.Decay {
			p.Value.Scale(10)
		}
	}
	batch, seq := 1, 4
	x := tensor.Randn(batch*seq, tinyCfg.EmbedDim, 1, rng)
	checkInputGrad(t, "block", x,
		func(m *tensor.Mat) *tensor.Mat { return b.Forward(m, batch, seq) },
		b.Backward, rng)
}
