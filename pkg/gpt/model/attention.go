package model

import (
	"math"
	"math/rand"
	"sync"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// CausalSelfAttention is multi-head self-attention where each position may
// only attend to itself and earlier positions.
//
// Activations are stored as [B*T, D] matrices (row b*T + t is token t of
// sequence b). Each (batch, head) pair is processed on its own: its q, k and
// v columns are gathered into [T, hs] matrices and the attention products
// run through the tiled tensor kernels, so the per-sequence, per-head
// structure costs a few copies rather than a reshape.
type CausalSelfAttention struct {
	QKV   *Linear // D -> 3D, columns laid out as [q heads | k heads | v heads]
	Proj  *Linear // D -> D
	Heads int

	// caches for Backward
	b, t    int
	headDim int
	qkv     *tensor.Mat // [B*T, 3D]

	// att holds the softmax weights for Backward. Each (batch, head) owns
	// headLen floats laid out as the row blocks in blocks: block i stores
	// rows [t0, t1) of the T x T attention matrix as a dense [t1-t0, t1]
	// matrix, because columns s >= t1 are all above the diagonal (masked
	// to zero) and so are never computed or stored. Skipping them saves
	// about half the score work at long contexts.
	att     []float64
	blocks  []attBlock
	headLen int

	scratch sync.Pool // *attnScratch, reused across calls and heads
}

// attRows is the number of query rows per attention block. Larger blocks
// give the kernels longer inner loops; smaller ones waste less of the
// triangle above the diagonal (about attRows/2 columns per row).
const attRows = 64

// attBlock describes one row block of a head's attention matrix.
type attBlock struct {
	t0, t1 int // query rows [t0, t1); key columns [0, t1)
	off    int // offset of the block within the head's att region
}

// attnScratch is the per-worker working set for one (batch, head).
type attnScratch struct {
	q, k, v, o *tensor.Mat // [T, hs]: this head's q, k, v, and its output (dOut in Backward)
	dq, dk, dv *tensor.Mat // [T, hs]: Backward gradients
	kvT        []float64   // hs*T: k^T or v^T of the keys a block attends to
	ds         []float64   // attRows*T: dScore for one block
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
	a.layout(batch, seq)
	y := tensor.New(batch*seq, d)
	scale := 1 / math.Sqrt(float64(hs))

	tensor.Parallel(batch*h, func(lo, hi int) {
		s := a.getScratch(seq)
		defer a.scratch.Put(s)
		for bh := lo; bh < hi; bh++ {
			b, head := bh/h, bh%h
			qOff, kOff, vOff := head*hs, d+head*hs, 2*d+head*hs
			gatherHead(s.q, a.qkv, b*seq, qOff)
			gatherHead(s.k, a.qkv, b*seq, kOff)
			gatherHead(s.v, a.qkv, b*seq, vOff)
			for _, blk := range a.blocks {
				att := a.attBlock(bh, blk)
				// att = softmax(scale * q k^T) over keys s <= t
				tensor.MatMulTInto(att, s.q.RowRange(blk.t0, blk.t1), s.k.RowRange(0, blk.t1))
				for r := 0; r < att.Rows; r++ {
					causalSoftmax(att.Row(r), blk.t0+r+1, scale)
				}
				// out = att v, computed as att @ (v^T)^T so both operands stream along s
				vT := tensor.FromSlice(hs, blk.t1, s.kvT[:hs*blk.t1])
				tensor.TransposeInto(vT, s.v.RowRange(0, blk.t1))
				tensor.MatMulTInto(s.o.RowRange(blk.t0, blk.t1), att, vT)
			}
			scatterHead(y, s.o, b*seq, qOff)
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
		s := a.getScratch(seq)
		defer a.scratch.Put(s)
		for bh := lo; bh < hi; bh++ {
			b, head := bh/h, bh%h
			qOff, kOff, vOff := head*hs, d+head*hs, 2*d+head*hs
			gatherHead(s.q, a.qkv, b*seq, qOff)
			gatherHead(s.k, a.qkv, b*seq, kOff)
			gatherHead(s.v, a.qkv, b*seq, vOff)
			gatherHead(s.o, dAttnOut, b*seq, qOff) // dOut
			s.dq.Zero()
			s.dk.Zero()
			s.dv.Zero()
			for _, blk := range a.blocks {
				att := a.attBlock(bh, blk)
				dOut := s.o.RowRange(blk.t0, blk.t1)
				// out = att v  =>  dAtt = dOut v^T ; dV += att^T dOut
				dS := tensor.FromSlice(att.Rows, att.Cols, s.ds[:att.Rows*att.Cols])
				tensor.MatMulTInto(dS, dOut, s.v.RowRange(0, blk.t1))
				tensor.MatMulTAInto(s.dv.RowRange(0, blk.t1), att, dOut)
				// softmax backward, in place: dScore = att * (dAtt - sum(att*dAtt))
				for r := 0; r < att.Rows; r++ {
					causalSoftmaxBackward(dS.Row(r), att.Row(r), blk.t0+r+1, scale)
				}
				// score = scale * q k^T  =>  dQ = dScore k ; dK += dScore^T q
				kT := tensor.FromSlice(hs, blk.t1, s.kvT[:hs*blk.t1])
				tensor.TransposeInto(kT, s.k.RowRange(0, blk.t1))
				tensor.MatMulTInto(s.dq.RowRange(blk.t0, blk.t1), dS, kT)
				tensor.MatMulTAInto(s.dk.RowRange(0, blk.t1), dS, s.q.RowRange(blk.t0, blk.t1))
			}
			scatterHead(dqkv, s.dq, b*seq, qOff)
			scatterHead(dqkv, s.dk, b*seq, kOff)
			scatterHead(dqkv, s.dv, b*seq, vOff)
		}
	})
	return a.QKV.Backward(dqkv)
}

// layout sizes att and blocks for a batch of sequences of length seq,
// keeping the existing buffer when the shape is unchanged.
func (a *CausalSelfAttention) layout(batch, seq int) {
	if len(a.blocks) > 0 && a.blocks[len(a.blocks)-1].t1 == seq && len(a.att) == batch*a.Heads*a.headLen {
		return
	}
	a.blocks = a.blocks[:0]
	a.headLen = 0
	for t0 := 0; t0 < seq; t0 += attRows {
		t1 := min(t0+attRows, seq)
		a.blocks = append(a.blocks, attBlock{t0: t0, t1: t1, off: a.headLen})
		a.headLen += (t1 - t0) * t1
	}
	a.att = make([]float64, batch*a.Heads*a.headLen)
}

// attBlock returns block blk of (batch, head) pair bh as a [t1-t0, t1] view.
func (a *CausalSelfAttention) attBlock(bh int, blk attBlock) *tensor.Mat {
	rows, cols := blk.t1-blk.t0, blk.t1
	start := bh*a.headLen + blk.off
	return tensor.FromSlice(rows, cols, a.att[start:start+rows*cols])
}

// getScratch returns a working set sized for sequence length seq.
func (a *CausalSelfAttention) getScratch(seq int) *attnScratch {
	hs := a.headDim
	if s, ok := a.scratch.Get().(*attnScratch); ok && s.q.Rows == seq {
		return s
	}
	return &attnScratch{
		q: tensor.New(seq, hs), k: tensor.New(seq, hs), v: tensor.New(seq, hs), o: tensor.New(seq, hs),
		dq: tensor.New(seq, hs), dk: tensor.New(seq, hs), dv: tensor.New(seq, hs),
		kvT: make([]float64, hs*seq),
		ds:  make([]float64, min(attRows, seq)*seq),
	}
}

// gatherHead copies the dst.Cols-wide column block starting at column off
// of rows [row0, row0+dst.Rows) of src into dst.
func gatherHead(dst, src *tensor.Mat, row0, off int) {
	for i := 0; i < dst.Rows; i++ {
		copy(dst.Row(i), src.Row(row0 + i)[off:off+dst.Cols])
	}
}

// scatterHead is the inverse of gatherHead: it copies src into the column
// block of dst starting at column off of rows [row0, row0+src.Rows).
func scatterHead(dst, src *tensor.Mat, row0, off int) {
	for i := 0; i < src.Rows; i++ {
		copy(dst.Row(row0 + i)[off:off+src.Cols], src.Row(i))
	}
}

// causalSoftmax replaces the first n raw scores of row with
// softmax(scale * score) and zeroes the rest (the masked keys after the
// query position).
func causalSoftmax(row []float64, n int, scale float64) {
	maxv := math.Inf(-1)
	for s := 0; s < n; s++ {
		row[s] *= scale
		maxv = max(maxv, row[s])
	}
	var sum float64
	for s := 0; s < n; s++ {
		row[s] = math.Exp(row[s] - maxv)
		sum += row[s]
	}
	for s := 0; s < n; s++ {
		row[s] /= sum
	}
	clear(row[n:])
}

// causalSoftmaxBackward turns dAtt (the gradient w.r.t. the softmax output
// in the first n entries) into the gradient w.r.t. the pre-scale scores,
// in place, given the softmax output att; the masked tail is zeroed.
func causalSoftmaxBackward(dAtt, att []float64, n int, scale float64) {
	var dot float64
	for s := 0; s < n; s++ {
		dot += att[s] * dAtt[s]
	}
	for s := 0; s < n; s++ {
		dAtt[s] = att[s] * (dAtt[s] - dot) * scale
	}
	clear(dAtt[n:])
}

// Params lists the trainable parameters.
func (a *CausalSelfAttention) Params() []*Param {
	return append(a.QKV.Params(), a.Proj.Params()...)
}
