package gpumodel

import (
	"fmt"
	"math"
	"time"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/gpu"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

// rec is the batch a step records into. In profiling mode mark() submits
// the batch so far, times it under a phase name and starts a new one;
// otherwise mark is a no-op and the whole step stays one submission.
type rec struct {
	m     *Model
	b     *gpu.CommandBatch
	prof  map[string]time.Duration
	order []string
	t0    time.Time
}

func (m *Model) newRec(profile bool) *rec {
	r := &rec{m: m, b: m.dev.Batch()}
	if profile {
		r.prof = map[string]time.Duration{}
		r.t0 = time.Now()
	}
	return r
}

// mark ends the current phase. In profiling mode the dispatches recorded
// since the last mark are submitted and waited for, so the phase's time
// is GPU time (plus one submit per phase, which is why profiling is slower
// than a real step and only used to see where time goes).
func (r *rec) mark(name string) {
	if r.prof == nil {
		return
	}
	if err := r.b.Submit(); err != nil {
		panic(err)
	}
	if _, seen := r.prof[name]; !seen {
		r.order = append(r.order, name)
	}
	r.prof[name] += time.Since(r.t0)
	r.b = r.m.dev.Batch()
	r.t0 = time.Now()
}

// Phase is one named part of a profiled step.
type Phase struct {
	Name string
	Time time.Duration
}

// Profile runs one training step split into phases, each submitted and
// timed on its own, and returns them in order of first occurrence with the
// per-layer phases summed over layers. The parameters are updated as by
// TrainStep. It is a diagnostic: the per-phase submits make it slower than
// TrainStep.
func (m *Model) Profile(x, y [][]int, opt train.AdamWConfig, maxNorm float64, step int) ([]Phase, error) {
	r, err := m.step(x, y, opt, maxNorm, step, true)
	if err != nil {
		return nil, err
	}
	out := make([]Phase, 0, len(r.order))
	for _, n := range r.order {
		out = append(out, Phase{n, r.prof[n]})
	}
	return out, nil
}

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
func (m *Model) recordForward(r *rec, a *activations, drop bool, step uint32) {
	cfg := m.Config
	d, h, hs := cfg.EmbedDim, cfg.NumHeads, cfg.HeadDim()
	batch, seq := a.batch, a.seq
	p := cfg.Dropout
	scale := attnScale(cfg)

	r.b.Gather(a.x0, m.tokEmb.W, a.ids, false)
	r.b.Gather(a.x0, m.posEmb.W, a.pos, true)
	if drop {
		r.b.Dropout(a.x0, a.embMask, p, m.seed+seedEmb, step)
	}
	r.mark("fwd.embed")
	x := a.x0
	for i, blk := range m.blocks {
		l := a.layers[i]
		n := batch * seq
		// h = x + Dropout(Proj(Attn(LN1(x))))
		r.b.LayerNorm(l.h1, l.xhat1, l.rstd1, x, blk.ln1g.W, blk.ln1b.W, lnEps)
		r.mark("fwd.ln")
		r.b.GEMM(l.qkv, l.h1, blk.qkvW.W, gpu.GEMM{M: n, N: 3 * d, K: d, Bias: blk.qkvB.W})
		r.mark("fwd.qkv")
		r.b.GEMM(l.scores, l.qkv, l.qkv, gpu.GEMM{M: seq, N: seq, K: hs, TransB: true, Batch: batch * h, Inner: h,
			A: headOperand(0, 3*d, seq, hs), B: headOperand(d, 3*d, seq, hs), C: scoreOperand(h, seq)})
		r.mark("fwd.attn.scores")
		r.b.Softmax(l.scores, scale, true)
		r.mark("fwd.attn.softmax")
		r.b.GEMM(l.attOut, l.scores, l.qkv, gpu.GEMM{M: seq, N: hs, K: seq, Batch: batch * h, Inner: h,
			A: scoreOperand(h, seq), B: headOperand(2*d, 3*d, seq, hs), C: headOperand(0, d, seq, hs)})
		r.mark("fwd.attn.values")
		// projection, bias, dropout and residual in one epilogue
		r.b.GEMM(l.h, l.attOut, blk.projW.W, m.residualGEMM(n, d, d, blk.projB.W, x, l.attnMask, drop, p, m.seed+seedAttn+uint32(2*i), step))
		r.mark("fwd.proj+res")
		// out = h + Dropout(MLP(LN2(h)))
		r.b.LayerNorm(l.h2, l.xhat2, l.rstd2, l.h, blk.ln2g.W, blk.ln2b.W, lnEps)
		r.mark("fwd.ln")
		// fc with bias and GELU: pre in l.pre, gelu(pre) in l.actv
		r.b.GEMM(l.pre, l.h2, blk.fcW.W, gpu.GEMM{M: n, N: 4 * d, K: d, Bias: blk.fcB.W, Mode: gpu.EpilogueGELU, Out2: l.actv})
		r.mark("fwd.fc+gelu")
		r.b.GEMM(l.out, l.actv, blk.mprW.W, m.residualGEMM(n, d, 4*d, blk.mprB.W, l.h, l.mlpMask, drop, p, m.seed+seedMLP+uint32(2*i), step))
		r.mark("fwd.mlpproj+res")
		x = l.out
	}
	r.b.LayerNorm(a.hF, a.xhatF, a.rstdF, x, m.lnfG.W, m.lnfB.W, lnEps)
	r.mark("fwd.ln")
	// tied output head: logits = hF @ tokEmb^T
	r.b.MatMulT(a.logits, a.hF, m.tokEmb.W, false)
	r.mark("fwd.head")
}

// residualGEMM is the epilogue of a sub-layer's output projection: bias,
// then dropout (writing mask) when drop is set, then the residual res.
func (m *Model) residualGEMM(n, outDim, k int, bias, res, mask *gpu.Tensor, drop bool, p float64, seed, step uint32) gpu.GEMM {
	g := gpu.GEMM{M: n, N: outDim, K: k, Bias: bias, Aux: res, Mode: gpu.EpilogueResidual}
	if drop {
		g.Mode, g.Out2 = gpu.EpilogueDropoutResidual, mask
		g.Drop = gpu.DropoutParams{P: p, Seed: seed, Step: step}
	}
	return g
}

// lnEps matches model.LayerNorm.
const lnEps = 1e-5

// recordBackward records the backward pass from dLogits (held in a.logits)
// into every parameter's gradient, which must have been zeroed.
func (m *Model) recordBackward(r *rec, a *activations, drop bool) {
	cfg := m.Config
	d, h, hs := cfg.EmbedDim, cfg.NumHeads, cfg.HeadDim()
	batch, seq := a.batch, a.seq
	scale := attnScale(cfg)
	dLogits := a.logits

	// logits = hF @ E^T  =>  dE += dLogits^T @ hF ; dhF = dLogits @ E
	r.b.MatMulTA(m.tokEmb.G, dLogits, a.hF, true)
	dy, dh, dx := a.dA, a.dB, a.dC
	r.b.MatMul(a.dLN, dLogits, m.tokEmb.W, false)
	r.mark("bwd.head")
	r.b.LayerNormBackward(dy, m.lnfG.G, m.lnfB.G, a.dLN, a.xhatF, a.rstdF, m.lnfG.W)
	r.mark("bwd.ln")

	for i := len(m.blocks) - 1; i >= 0; i-- {
		blk, l := m.blocks[i], a.layers[i]
		n := batch * seq
		// out = h + M*MLP(LN2(h))  =>  dh = dy + LN2'(MLP'(M*dy))
		g := dy
		if drop {
			r.b.MulInto(a.dMasked, dy, l.mlpMask)
			g = a.dMasked
		}
		r.mark("bwd.drop")
		r.b.MatMulTA(blk.mprW.G, l.actv, g, true)
		r.b.ColSum(blk.mprB.G, g, true)
		// dAct = (g @ Wproj^T) * gelu'(pre), the GELU backward folded in
		r.b.GEMM(a.dAct, g, blk.mprW.W, gpu.GEMM{M: n, N: 4 * d, K: d, TransB: true, Mode: gpu.EpilogueGELUGrad, Aux: l.pre})
		r.mark("bwd.mlpproj+gelu")
		r.b.MatMulTA(blk.fcW.G, l.h2, a.dAct, true)
		r.b.ColSum(blk.fcB.G, a.dAct, true)
		r.b.MatMulT(a.dLN, a.dAct, blk.fcW.W, false)
		r.mark("bwd.fc")
		r.b.LayerNormBackwardAdd(dh, blk.ln2g.G, blk.ln2b.G, a.dLN, l.xhat2, l.rstd2, blk.ln2g.W, dy)
		r.mark("bwd.ln+res")

		// h = x + M*Proj(Attn(LN1(x)))  =>  dx = dh + LN1'(Attn'(Proj'(M*dh)))
		g = dh
		if drop {
			r.b.MulInto(a.dMasked, dh, l.attnMask)
			g = a.dMasked
		}
		r.mark("bwd.drop")
		r.b.MatMulTA(blk.projW.G, l.attOut, g, true)
		r.b.ColSum(blk.projB.G, g, true)
		r.b.MatMulT(a.dAttOut, g, blk.projW.W, false)
		r.mark("bwd.proj")
		// attention: out = att v  =>  dAtt = dOut v^T ; dV = att^T dOut
		r.b.GEMM(a.dScores, a.dAttOut, l.qkv, gpu.GEMM{M: seq, N: seq, K: hs, TransB: true, Batch: batch * h, Inner: h,
			A: headOperand(0, d, seq, hs), B: headOperand(2*d, 3*d, seq, hs), C: scoreOperand(h, seq)})
		r.b.GEMM(a.dqkv, l.scores, a.dAttOut, gpu.GEMM{M: seq, N: hs, K: seq, TransA: true, Batch: batch * h, Inner: h,
			A: scoreOperand(h, seq), B: headOperand(0, d, seq, hs), C: headOperand(2*d, 3*d, seq, hs)})
		r.mark("bwd.attn.dscores+dv")
		// softmax backward (in place, includes the 1/sqrt(hs) scale), then
		// score = scale * q k^T  =>  dQ = dS k ; dK = dS^T q
		r.b.SoftmaxBackward(a.dScores, l.scores, scale, true)
		r.mark("bwd.attn.softmax")
		r.b.GEMM(a.dqkv, a.dScores, l.qkv, gpu.GEMM{M: seq, N: hs, K: seq, Batch: batch * h, Inner: h,
			A: scoreOperand(h, seq), B: headOperand(d, 3*d, seq, hs), C: headOperand(0, 3*d, seq, hs)})
		r.b.GEMM(a.dqkv, a.dScores, l.qkv, gpu.GEMM{M: seq, N: hs, K: seq, TransA: true, Batch: batch * h, Inner: h,
			A: scoreOperand(h, seq), B: headOperand(0, 3*d, seq, hs), C: headOperand(d, 3*d, seq, hs)})
		r.mark("bwd.attn.dq+dk")
		// qkv = h1 @ Wqkv + b
		r.b.MatMulTA(blk.qkvW.G, l.h1, a.dqkv, true)
		r.b.ColSum(blk.qkvB.G, a.dqkv, true)
		r.b.MatMulT(a.dLN, a.dqkv, blk.qkvW.W, false)
		r.mark("bwd.qkv")
		r.b.LayerNormBackwardAdd(dx, blk.ln1g.G, blk.ln1b.G, a.dLN, l.xhat1, l.rstd1, blk.ln1g.W, dh)
		r.mark("bwd.ln+res")

		dy, dh, dx = dx, dy, dh // dx becomes the next layer's dy
	}
	// embeddings: x0 = M*(tok[ids] + pos[t])
	if drop {
		r.b.Mul(dy, a.embMask)
	}
	r.b.ScatterAdd(m.tokEmb.G, dy, a.ids)
	r.b.ScatterAdd(m.posEmb.G, dy, a.pos)
	r.mark("bwd.embed")
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
	r, err := m.step(x, y, opt, maxNorm, step, false)
	if err != nil {
		return StepResult{}, err
	}
	return r.result, nil
}

// stepRec is a rec plus the step's readbacks.
type stepRec struct {
	*rec
	result StepResult
}

func (m *Model) step(x, y [][]int, opt train.AdamWConfig, maxNorm float64, step int, profile bool) (*stepRec, error) {
	ids, batch, seq, err := m.flatten(x)
	if err != nil {
		return nil, err
	}
	targets, tb, ts, err := m.flatten(y)
	if err != nil {
		return nil, err
	}
	if tb != batch || ts != seq {
		return nil, errShape("targets", tb, ts, batch, seq)
	}
	a := m.ensure(batch, seq)
	a.ids.Set(ids)
	a.targets.Set(targets)
	drop := m.dropping()
	n := batch * seq

	r := m.newRec(profile)
	for _, p := range m.params {
		r.b.Zero(p.G)
	}
	r.mark("zero")
	m.recordForward(r, a, drop, uint32(step))
	r.b.CrossEntropy(a.losses, a.logits, a.logits, a.targets)
	r.b.Sum(a.lossT, 0, a.losses, false, false, 1/float64(n))
	r.mark("loss")
	m.recordBackward(r, a, drop)
	for i, p := range m.params {
		r.b.Sum(a.normT, 0, p.G, true, i > 0, 1)
	}
	r.mark("opt.norm")
	for _, p := range m.params {
		r.b.AdamW(p.W, p.G, p.M, p.V, a.normT, gpu.AdamWStep{
			LR: opt.LR, Beta1: opt.Beta1, Beta2: opt.Beta2, Eps: opt.Eps,
			WeightDecay: opt.WeightDecay, Decay: p.Decay, Step: step, MaxNorm: maxNorm,
		})
	}
	r.mark("opt.adamw")
	rbLoss, rbNorm := r.b.Read(a.lossT), r.b.Read(a.normT)
	if err := r.b.Submit(); err != nil {
		return nil, err
	}
	return &stepRec{rec: r, result: StepResult{Loss: float64(rbLoss.Values()[0]), GradNorm: math.Sqrt(float64(rbNorm.Values()[0]))}}, nil
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
	r := m.newRec(false)
	m.recordForward(r, a, false, 0)
	r.b.CrossEntropy(a.losses, a.logits, a.logits, a.targets)
	r.b.Sum(a.lossT, 0, a.losses, false, false, 1/float64(batch*seq))
	rb := r.b.Read(a.lossT)
	if err := r.b.Submit(); err != nil {
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
	r := m.newRec(false)
	m.recordForward(r, a, m.dropping(), 0)
	if err := r.b.Submit(); err != nil {
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
	r := m.newRec(false)
	for _, p := range m.params {
		r.b.Zero(p.G)
	}
	m.recordBackward(r, a, m.dropping())
	return r.b.Submit()
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
