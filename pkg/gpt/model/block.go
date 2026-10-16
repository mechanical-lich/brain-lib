package model

import (
	"fmt"
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// Block is one pre-norm transformer layer:
//
//	x = x + Dropout(Attn(LN1(x)))
//	x = x + Dropout(MLP(LN2(x)))
//
// Dropout is active only in GPT training mode; Block.Forward never drops.
type Block struct {
	LN1  *LayerNorm
	Attn *CausalSelfAttention
	LN2  *LayerNorm
	MLP  *MLP

	attnDrop, mlpDrop dropout
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

// Forward applies the block to x [B*T, D] without dropout.
func (b *Block) Forward(x *tensor.Mat, batch, seq int) *tensor.Mat {
	return b.forward(x, batch, seq, 0, nil)
}

// forward applies the block with dropout probability p, drawing masks from
// rng when it is non-nil.
func (b *Block) forward(x *tensor.Mat, batch, seq int, p float64, rng *rand.Rand) *tensor.Mat {
	// Each sub-layer returns a fresh matrix, so dropout and the residual
	// are applied to it in place rather than cloning x first.
	h := b.Attn.Forward(b.LN1.Forward(x), batch, seq)
	b.attnDrop.forward(h, p, rng)
	h.Add(x)
	out := b.MLP.Forward(b.LN2.Forward(h))
	b.mlpDrop.forward(out, p, rng)
	out.Add(h)
	return out
}

// Backward returns dx given dy [B*T, D].
func (b *Block) Backward(dy *tensor.Mat) *tensor.Mat {
	// out = h + M*MLP(LN2(h))  =>  dh = dy + LN2'(MLP'(M*dy))
	dh := b.LN2.Backward(b.MLP.Backward(b.mlpDrop.backward(dy, false)))
	dh.Add(dy)
	// h = x + M*Attn(LN1(x))   =>  dx = dh + LN1'(Attn'(M*dh))
	dx := b.LN1.Backward(b.Attn.Backward(b.attnDrop.backward(dh, false)))
	dx.Add(dh)
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
