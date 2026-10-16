package gpumodel

import (
	"fmt"
	"math"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/gpu"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

// headOperand describes the [T, hs] matrix of every (batch, head) pair
// inside a [B*T, ld] activation whose head columns start at col: entry
// z = b*H + h lives at column col + h*hs of rows b*T .. b*T+T.
func headOperand(col, ld, seq, hs int) gpu.Operand {
	return gpu.Operand{Offset: col, LD: ld, Outer: seq * ld, InnerStride: hs}
}

// scoreOperand describes the [T, T] matrix of every (batch, head) pair in a
// [B*H*T, T] tensor.
func scoreOperand(heads, seq int) gpu.Operand {
	return gpu.Operand{Outer: heads * seq * seq, InnerStride: seq * seq}
}

// recordForward records the forward pass for the ids already set on a,
// with dropout when drop is true, leaving logits in a.logits.
func (m *Model) recordForward(b *gpu.CommandBatch, a *activations, drop bool, step uint32) {
	cfg := m.Config
	d, h, hs := cfg.EmbedDim, cfg.NumHeads, cfg.HeadDim()
	batch, seq := a.batch, a.seq
	p := cfg.Dropout
	scale := attnScale(cfg)

	b.Gather(a.x0, m.tokEmb.W, a.ids, false)
	b.Gather(a.x0, m.posEmb.W, a.pos, true)
	if drop {
		b.Dropout(a.x0, a.embMask, p, m.seed+seedEmb, step)
	}
	x := a.x0
	for i, blk := range m.blocks {
		l := a.layers[i]
		// h = x + Dropout(Proj(Attn(LN1(x))))
		b.LayerNorm(l.h1, l.xhat1, l.rstd1, x, blk.ln1g.W, blk.ln1b.W, lnEps)
		b.MatMul(l.qkv, l.h1, blk.qkvW.W, false)
		b.AddRow(l.qkv, blk.qkvB.W)
		b.GEMM(l.scores, l.qkv, l.qkv, gpu.GEMM{M: seq, N: seq, K: hs, TransB: true, Batch: batch * h, Inner: h,
			A: headOperand(0, 3*d, seq, hs), B: headOperand(d, 3*d, seq, hs), C: scoreOperand(h, seq)})
		b.Softmax(l.scores, scale, true)
		b.GEMM(l.attOut, l.scores, l.qkv, gpu.GEMM{M: seq, N: hs, K: seq, Batch: batch * h, Inner: h,
			A: scoreOperand(h, seq), B: headOperand(2*d, 3*d, seq, hs), C: headOperand(0, d, seq, hs)})
		b.MatMul(l.h, l.attOut, blk.projW.W, false)
		b.AddRow(l.h, blk.projB.W)
		if drop {
			b.Dropout(l.h, l.attnMask, p, m.seed+seedAttn+uint32(2*i), step)
		}
		b.Add(l.h, x)
		// out = h + Dropout(MLP(LN2(h)))
		b.LayerNorm(l.h2, l.xhat2, l.rstd2, l.h, blk.ln2g.W, blk.ln2b.W, lnEps)
		b.MatMul(l.pre, l.h2, blk.fcW.W, false)
		b.AddRow(l.pre, blk.fcB.W)
		b.GELU(l.actv, l.pre)
		b.MatMul(l.out, l.actv, blk.mprW.W, false)
		b.AddRow(l.out, blk.mprB.W)
		if drop {
			b.Dropout(l.out, l.mlpMask, p, m.seed+seedMLP+uint32(2*i), step)
		}
		b.Add(l.out, l.h)
		x = l.out
	}
	b.LayerNorm(a.hF, a.xhatF, a.rstdF, x, m.lnfG.W, m.lnfB.W, lnEps)
	// tied output head: logits = hF @ tokEmb^T
	b.MatMulT(a.logits, a.hF, m.tokEmb.W, false)
}

// lnEps matches model.LayerNorm.
const lnEps = 1e-5

// recordBackward records the backward pass from dLogits (held in a.logits)
// into every parameter's gradient, which must have been zeroed.
func (m *Model) recordBackward(b *gpu.CommandBatch, a *activations, drop bool) {
	cfg := m.Config
	d, h, hs := cfg.EmbedDim, cfg.NumHeads, cfg.HeadDim()
	batch, seq := a.batch, a.seq
	scale := attnScale(cfg)
	dLogits := a.logits

	// logits = hF @ E^T  =>  dE += dLogits^T @ hF ; dhF = dLogits @ E
	b.MatMulTA(m.tokEmb.G, dLogits, a.hF, true)
	dy, dh, dx := a.dA, a.dB, a.dC
	b.MatMul(a.dLN, dLogits, m.tokEmb.W, false)
	b.LayerNormBackward(dy, m.lnfG.G, m.lnfB.G, a.dLN, a.xhatF, a.rstdF, m.lnfG.W)

	for i := len(m.blocks) - 1; i >= 0; i-- {
		blk, l := m.blocks[i], a.layers[i]
		// out = h + M*MLP(LN2(h))  =>  dh = dy + LN2'(MLP'(M*dy))
		g := dy
		if drop {
			b.Copy(a.dMasked, dy)
			b.Mul(a.dMasked, l.mlpMask)
			g = a.dMasked
		}
		b.MatMulTA(blk.mprW.G, l.actv, g, true)
		b.ColSum(blk.mprB.G, g, true)
		b.MatMulT(a.dAct, g, blk.mprW.W, false)
		b.GELUBackward(a.dAct, l.pre)
		b.MatMulTA(blk.fcW.G, l.h2, a.dAct, true)
		b.ColSum(blk.fcB.G, a.dAct, true)
		b.MatMulT(a.dLN, a.dAct, blk.fcW.W, false)
		b.LayerNormBackward(dh, blk.ln2g.G, blk.ln2b.G, a.dLN, l.xhat2, l.rstd2, blk.ln2g.W)
		b.Add(dh, dy)

		// h = x + M*Proj(Attn(LN1(x)))  =>  dx = dh + LN1'(Attn'(Proj'(M*dh)))
		g = dh
		if drop {
			b.Copy(a.dMasked, dh)
			b.Mul(a.dMasked, l.attnMask)
			g = a.dMasked
		}
		b.MatMulTA(blk.projW.G, l.attOut, g, true)
		b.ColSum(blk.projB.G, g, true)
		b.MatMulT(a.dAttOut, g, blk.projW.W, false)
		// attention: out = att v  =>  dAtt = dOut v^T ; dV = att^T dOut
		b.GEMM(a.dScores, a.dAttOut, l.qkv, gpu.GEMM{M: seq, N: seq, K: hs, TransB: true, Batch: batch * h, Inner: h,
			A: headOperand(0, d, seq, hs), B: headOperand(2*d, 3*d, seq, hs), C: scoreOperand(h, seq)})
		b.GEMM(a.dqkv, l.scores, a.dAttOut, gpu.GEMM{M: seq, N: hs, K: seq, TransA: true, Batch: batch * h, Inner: h,
			A: scoreOperand(h, seq), B: headOperand(0, d, seq, hs), C: headOperand(2*d, 3*d, seq, hs)})
		// softmax backward (in place, includes the 1/sqrt(hs) scale), then
		// score = scale * q k^T  =>  dQ = dS k ; dK = dS^T q
		b.SoftmaxBackward(a.dScores, l.scores, scale, true)
		b.GEMM(a.dqkv, a.dScores, l.qkv, gpu.GEMM{M: seq, N: hs, K: seq, Batch: batch * h, Inner: h,
			A: scoreOperand(h, seq), B: headOperand(d, 3*d, seq, hs), C: headOperand(0, 3*d, seq, hs)})
		b.GEMM(a.dqkv, a.dScores, l.qkv, gpu.GEMM{M: seq, N: hs, K: seq, TransA: true, Batch: batch * h, Inner: h,
			A: scoreOperand(h, seq), B: headOperand(0, 3*d, seq, hs), C: headOperand(d, 3*d, seq, hs)})
		// qkv = h1 @ Wqkv + b
		b.MatMulTA(blk.qkvW.G, l.h1, a.dqkv, true)
		b.ColSum(blk.qkvB.G, a.dqkv, true)
		b.MatMulT(a.dLN, a.dqkv, blk.qkvW.W, false)
		b.LayerNormBackward(dx, blk.ln1g.G, blk.ln1b.G, a.dLN, l.xhat1, l.rstd1, blk.ln1g.W)
		b.Add(dx, dh)

		dy, dh, dx = dx, dy, dh // dx becomes the next layer's dy
	}
	// embeddings: x0 = M*(tok[ids] + pos[t])
	if drop {
		b.Mul(dy, a.embMask)
	}
	b.ScatterAdd(m.tokEmb.G, dy, a.ids)
	b.ScatterAdd(m.posEmb.G, dy, a.pos)
}

// StepResult is what TrainStep reads back.
type StepResult struct {
	Loss     float64 // mean cross-entropy over the batch, before the update
	GradNorm float64 // global L2 norm of the gradient, before clipping
}

// TrainStep runs one optimizer step on the batch x with targets y: forward
// with dropout (when training mode is on), the mean cross-entropy, backward,
// the global gradient norm with optional clipping, and the AdamW update for
// every parameter, all in one submission. step is the 1-based update
// number, used for AdamW's bias correction and as the dropout counter.
// Only the loss and the gradient norm are read back.
func (m *Model) TrainStep(x, y [][]int, opt train.AdamWConfig, maxNorm float64, step int) (StepResult, error) {
	ids, batch, seq, err := m.flatten(x)
	if err != nil {
		return StepResult{}, err
	}
	targets, tb, ts, err := m.flatten(y)
	if err != nil {
		return StepResult{}, err
	}
	if tb != batch || ts != seq {
		return StepResult{}, errShape("targets", tb, ts, batch, seq)
	}
	a := m.ensure(batch, seq)
	a.ids.Set(ids)
	a.targets.Set(targets)
	drop := m.dropping()
	n := batch * seq

	b := m.dev.Batch()
	for _, p := range m.params {
		b.Zero(p.G)
	}
	m.recordForward(b, a, drop, uint32(step))
	b.CrossEntropy(a.losses, a.logits, a.logits, a.targets)
	b.Sum(a.lossT, 0, a.losses, false, false, 1/float64(n))
	m.recordBackward(b, a, drop)
	for i, p := range m.params {
		b.Sum(a.normT, 0, p.G, true, i > 0, 1)
	}
	for _, p := range m.params {
		b.AdamW(p.W, p.G, p.M, p.V, a.normT, gpu.AdamWStep{
			LR: opt.LR, Beta1: opt.Beta1, Beta2: opt.Beta2, Eps: opt.Eps,
			WeightDecay: opt.WeightDecay, Decay: p.Decay, Step: step, MaxNorm: maxNorm,
		})
	}
	rbLoss, rbNorm := b.Read(a.lossT), b.Read(a.normT)
	if err := b.Submit(); err != nil {
		return StepResult{}, err
	}
	return StepResult{Loss: float64(rbLoss.Values()[0]), GradNorm: math.Sqrt(float64(rbNorm.Values()[0]))}, nil
}

// Eval returns the mean cross-entropy of y given x without dropout and
// without touching the parameters: one submission, one scalar back.
func (m *Model) Eval(x, y [][]int) (float64, error) {
	ids, batch, seq, err := m.flatten(x)
	if err != nil {
		return 0, err
	}
	targets, tb, ts, err := m.flatten(y)
	if err != nil {
		return 0, err
	}
	if tb != batch || ts != seq {
		return 0, errShape("targets", tb, ts, batch, seq)
	}
	a := m.ensure(batch, seq)
	a.ids.Set(ids)
	a.targets.Set(targets)
	b := m.dev.Batch()
	m.recordForward(b, a, false, 0)
	b.CrossEntropy(a.losses, a.logits, a.logits, a.targets)
	b.Sum(a.lossT, 0, a.losses, false, false, 1/float64(batch*seq))
	rb := b.Read(a.lossT)
	if err := b.Submit(); err != nil {
		return 0, err
	}
	return float64(rb.Values()[0]), nil
}

// Forward returns the logits [B*T, V] for tokens as a float64 matrix, the
// same contract as model.GPT.Forward. Dropout applies only in training mode,
// as on the CPU. It downloads the whole logits matrix, so it is for
// evaluation, generation and tests, not for the training loop.
func (m *Model) Forward(tokens [][]int) (*tensor.Mat, error) {
	ids, batch, seq, err := m.flatten(tokens)
	if err != nil {
		return nil, err
	}
	a := m.ensure(batch, seq)
	a.ids.Set(ids)
	if err := m.dev.Run(func(b *gpu.CommandBatch) {
		m.recordForward(b, a, m.dropping(), 0)
	}); err != nil {
		return nil, err
	}
	return a.logits.ToMat()
}

// Backward accumulates every parameter gradient from dLogits [B*T, V] for
// the most recent Forward, like model.GPT.Backward, and leaves the
// gradients on the device. It exists so the backward pass can be checked
// against the CPU on its own; TrainStep does not use it.
func (m *Model) Backward(dLogits *tensor.Mat) error {
	a := m.act
	if a == nil || dLogits.Rows != a.batch*a.seq || dLogits.Cols != m.Config.VocabSize {
		return errShape("dLogits", dLogits.Rows, dLogits.Cols, a.batch*a.seq, m.Config.VocabSize)
	}
	a.logits.Upload(dLogits)
	return m.dev.Run(func(b *gpu.CommandBatch) {
		for _, p := range m.params {
			b.Zero(p.G)
		}
		m.recordBackward(b, a, m.dropping())
	})
}

// Grads downloads every parameter gradient, in Params order.
func (m *Model) Grads() ([]*tensor.Mat, error) {
	out := make([]*tensor.Mat, len(m.params))
	for i, p := range m.params {
		g, err := p.G.ToMat()
		if err != nil {
			return nil, err
		}
		out[i] = g
	}
	return out, nil
}

func errShape(what string, rows, cols, wantR, wantC int) error {
	return fmt.Errorf("gpumodel: %s is %dx%d, want %dx%d", what, rows, cols, wantR, wantC)
}
