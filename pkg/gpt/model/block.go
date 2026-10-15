package model

import (
	"fmt"
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// Block is one pre-norm transformer layer:
//
//	x = x + Attn(LN1(x))
//	x = x + MLP(LN2(x))
type Block struct {
	LN1  *LayerNorm
	Attn *CausalSelfAttention
	LN2  *LayerNorm
	MLP  *MLP
}

// NewBlock creates block number idx (used only for parameter naming).
func NewBlock(idx int, cfg Config, rng *rand.Rand) *Block {
	name := fmt.Sprintf("blocks.%d", idx)
	return &Block{
		LN1:  NewLayerNorm(name+".ln1", cfg.EmbedDim),
		Attn: NewCausalSelfAttention(name+".attn", cfg, rng),
		LN2:  NewLayerNorm(name+".ln2", cfg.EmbedDim),
		MLP:  NewMLP(name+".mlp", cfg, rng),
	}
}

// Forward applies the block to x [B*T, D].
func (b *Block) Forward(x *tensor.Mat, batch, seq int) *tensor.Mat {
	h := x.Clone()
	h.Add(b.Attn.Forward(b.LN1.Forward(x), batch, seq))
	out := h.Clone()
	out.Add(b.MLP.Forward(b.LN2.Forward(h)))
	return out
}

// Backward returns dx given dy [B*T, D].
func (b *Block) Backward(dy *tensor.Mat) *tensor.Mat {
	// out = h + MLP(LN2(h))  =>  dh = dy + LN2'(MLP'(dy))
	dh := dy.Clone()
	dh.Add(b.LN2.Backward(b.MLP.Backward(dy)))
	// h = x + Attn(LN1(x))   =>  dx = dh + LN1'(Attn'(dh))
	dx := dh.Clone()
	dx.Add(b.LN1.Backward(b.Attn.Backward(dh)))
	return dx
}

// Params lists the trainable parameters in a stable order.
func (b *Block) Params() []*Param {
	var ps []*Param
	ps = append(ps, b.LN1.Params()...)
	ps = append(ps, b.Attn.Params()...)
	ps = append(ps, b.LN2.Params()...)
	ps = append(ps, b.MLP.Params()...)
	return ps
}
