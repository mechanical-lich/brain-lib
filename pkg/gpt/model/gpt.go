package model

import (
	"fmt"
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// GPT is a decoder-only transformer language model over integer tokens.
//
// Forward takes a batch of equal-length token sequences and returns next-token
// logits for every position; Backward takes the gradient of the loss with
// respect to those logits and accumulates gradients into every Param.
// The output head shares weights with the token embedding (GPT-2 style).
type GPT struct {
	Config Config

	TokEmb *Param // [VocabSize, EmbedDim]
	PosEmb *Param // [ContextLen, EmbedDim]
	Blocks []*Block
	LNF    *LayerNorm

	// caches for Backward
	ids    []int // flattened input ids, len B*T
	b, t   int
	hFinal *tensor.Mat // output of LNF, [B*T, D]
}

// New creates a randomly initialized GPT. Panics on an invalid config; call
// cfg.Validate first if the config is untrusted.
func New(cfg Config, rng *rand.Rand) *GPT {
	if err := cfg.Validate(); err != nil {
		panic(err)
	}
	g := &GPT{
		Config: cfg,
		TokEmb: newParam("tok_emb", tensor.Randn(cfg.VocabSize, cfg.EmbedDim, 0.02, rng), true),
		PosEmb: newParam("pos_emb", tensor.Randn(cfg.ContextLen, cfg.EmbedDim, 0.02, rng), true),
		LNF:    NewLayerNorm("ln_f", cfg.EmbedDim),
	}
	for i := 0; i < cfg.NumLayers; i++ {
		g.Blocks = append(g.Blocks, NewBlock(i, cfg, rng))
	}
	return g
}

// Params lists every trainable parameter in a stable, deterministic order.
func (g *GPT) Params() []*Param {
	ps := []*Param{g.TokEmb, g.PosEmb}
	for _, b := range g.Blocks {
		ps = append(ps, b.Params()...)
	}
	return append(ps, g.LNF.Params()...)
}

// NumParams returns the total number of scalar parameters.
func (g *GPT) NumParams() int {
	n := 0
	for _, p := range g.Params() {
		n += p.Size()
	}
	return n
}

// ZeroGrads clears every parameter gradient.
func (g *GPT) ZeroGrads() {
	for _, p := range g.Params() {
		p.ZeroGrad()
	}
}

// Forward computes logits for tokens, a batch of B sequences each of length
// T <= ContextLen. The result is [B*T, VocabSize]; row b*T + t holds the
// next-token logits for position t of sequence b.
func (g *GPT) Forward(tokens [][]int) *tensor.Mat {
	batch := len(tokens)
	if batch == 0 {
		panic("model: empty batch")
	}
	seq := len(tokens[0])
	if seq == 0 || seq > g.Config.ContextLen {
		panic(fmt.Sprintf("model: sequence length %d not in [1, %d]", seq, g.Config.ContextLen))
	}
	d := g.Config.EmbedDim
	g.b, g.t = batch, seq
	g.ids = make([]int, 0, batch*seq)

	x := tensor.New(batch*seq, d)
	for b, row := range tokens {
		if len(row) != seq {
			panic(fmt.Sprintf("model: ragged batch: row %d has length %d, want %d", b, len(row), seq))
		}
		for t, id := range row {
			if id < 0 || id >= g.Config.VocabSize {
				panic(fmt.Sprintf("model: token id %d out of range [0, %d)", id, g.Config.VocabSize))
			}
			g.ids = append(g.ids, id)
			xr := x.Row(b*seq + t)
			copy(xr, g.TokEmb.Value.Row(id))
			for j, v := range g.PosEmb.Value.Row(t) {
				xr[j] += v
			}
		}
	}
	for _, blk := range g.Blocks {
		x = blk.Forward(x, batch, seq)
	}
	g.hFinal = g.LNF.Forward(x)
	// tied output head: logits = h @ TokEmb^T
	return tensor.MatMulT(g.hFinal, g.TokEmb.Value)
}

// Backward accumulates parameter gradients given dLogits [B*T, VocabSize],
// the gradient of the loss with respect to the logits returned by the most
// recent Forward.
func (g *GPT) Backward(dLogits *tensor.Mat) {
	// logits = h @ E^T  =>  dh = dLogits @ E ; dE += dLogits^T @ h
	g.TokEmb.Grad.Add(tensor.MatMulTA(dLogits, g.hFinal))
	dx := tensor.MatMul(dLogits, g.TokEmb.Value)

	dx = g.LNF.Backward(dx)
	for i := len(g.Blocks) - 1; i >= 0; i-- {
		dx = g.Blocks[i].Backward(dx)
	}
	// embedding lookups: scatter-add into the rows that were used
	for r, id := range g.ids {
		dxr := dx.Row(r)
		te := g.TokEmb.Grad.Row(id)
		pe := g.PosEmb.Grad.Row(r % g.t)
		for j, v := range dxr {
			te[j] += v
			pe[j] += v
		}
	}
}
