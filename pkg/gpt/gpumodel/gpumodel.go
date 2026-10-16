// Package gpumodel runs the GPT training step from pkg/gpt/model on a GPU,
// using the float32 tensor layer and kernels in pkg/gpt/gpu.
//
// A Model holds the same parameters as a model.GPT, in the same order and
// under the same names, so weights and optimizer moments round-trip through
// the CPU types and the existing float64 checkpoints. Every activation of
// the forward pass stays resident on the device for the backward pass, and
// TrainStep records forward, loss, backward, gradient norm and the AdamW
// update into one CommandBatch, submits once, and reads back two scalars.
//
// Layout follows the CPU model: activations are [B*T, D] with row b*T + t,
// qkv is [B*T, 3D] with q, k and v heads side by side, and attention
// weights are [B*H*T, T] with row (b*H + h)*T + t.
package gpumodel

import (
	"fmt"
	"math"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/gpu"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

// Param is one trainable matrix on the device with its gradient and AdamW
// moments.
type Param struct {
	Name  string
	Decay bool
	W     *gpu.Tensor // value
	G     *gpu.Tensor // gradient, accumulated during Backward
	M, V  *gpu.Tensor // AdamW first and second moments
}

func (p *Param) release() {
	for _, t := range []*gpu.Tensor{p.W, p.G, p.M, p.V} {
		t.Release()
	}
}

// block holds one transformer layer's parameters.
type block struct {
	ln1g, ln1b   *Param
	qkvW, qkvB   *Param
	projW, projB *Param
	ln2g, ln2b   *Param
	fcW, fcB     *Param
	mprW, mprB   *Param
}

// Model is a GPT resident on a GPU.
type Model struct {
	Config model.Config
	dev    *gpu.Device

	params []*Param // model.GPT.Params() order
	tokEmb *Param
	posEmb *Param
	blocks []*block
	lnfG   *Param
	lnfB   *Param

	act      *activations
	training bool
	seed     uint32
}

// New allocates a zero-initialized model for cfg on dev. Call LoadFrom to
// fill it from a CPU model (which is where random initialization lives).
func New(dev *gpu.Device, cfg model.Config) (*Model, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	m := &Model{Config: cfg, dev: dev, seed: 0x9E3779B9}
	d := cfg.EmbedDim
	add := func(name string, rows, cols int, decay bool) *Param {
		p := &Param{Name: name, Decay: decay,
			W: dev.NewTensor(rows, cols), G: dev.NewTensor(rows, cols),
			M: dev.NewTensor(rows, cols), V: dev.NewTensor(rows, cols)}
		m.params = append(m.params, p)
		return p
	}
	m.tokEmb = add("tok_emb", cfg.VocabSize, d, true)
	m.posEmb = add("pos_emb", cfg.ContextLen, d, true)
	for i := 0; i < cfg.NumLayers; i++ {
		n := fmt.Sprintf("blocks.%d", i)
		b := &block{}
		b.ln1g = add(n+".ln1.g", 1, d, false)
		b.ln1b = add(n+".ln1.b", 1, d, false)
		b.qkvW = add(n+".attn.qkv.w", d, 3*d, true)
		b.qkvB = add(n+".attn.qkv.b", 1, 3*d, false)
		b.projW = add(n+".attn.proj.w", d, d, true)
		b.projB = add(n+".attn.proj.b", 1, d, false)
		b.ln2g = add(n+".ln2.g", 1, d, false)
		b.ln2b = add(n+".ln2.b", 1, d, false)
		b.fcW = add(n+".mlp.fc.w", d, 4*d, true)
		b.fcB = add(n+".mlp.fc.b", 1, 4*d, false)
		b.mprW = add(n+".mlp.proj.w", 4*d, d, true)
		b.mprB = add(n+".mlp.proj.b", 1, d, false)
		m.blocks = append(m.blocks, b)
	}
	m.lnfG = add("ln_f.g", 1, d, false)
	m.lnfB = add("ln_f.b", 1, d, false)
	return m, nil
}

// Params lists the parameters in model.GPT.Params() order.
func (m *Model) Params() []*Param { return m.params }

// NumParams returns the total number of scalar parameters.
func (m *Model) NumParams() int {
	n := 0
	for _, p := range m.params {
		n += p.W.Len()
	}
	return n
}

// SetTraining turns dropout on or off for Forward and TrainStep, as
// model.GPT.SetTraining does. Eval and Logits never drop.
func (m *Model) SetTraining(on bool) { m.training = on }

// SetSeed changes the dropout seed; masks are a function of (seed, site,
// step), so two models with the same seed draw the same masks.
func (m *Model) SetSeed(seed uint32) { m.seed = seed }

// Close releases every device buffer the model owns.
func (m *Model) Close() {
	for _, p := range m.params {
		p.release()
	}
	m.params = nil
	if m.act != nil {
		m.act.release()
		m.act = nil
	}
}

// LoadFrom uploads the weights of g, which must have the same Config. The
// optimizer moments are zeroed; see LoadOptimizer.
func (m *Model) LoadFrom(g *model.GPT) error {
	if g.Config != m.Config {
		return fmt.Errorf("gpumodel: config mismatch: gpu %+v, cpu %+v", m.Config, g.Config)
	}
	cpu := g.Params()
	if len(cpu) != len(m.params) {
		return fmt.Errorf("gpumodel: %d cpu params, %d gpu params", len(cpu), len(m.params))
	}
	for i, p := range m.params {
		if cpu[i].Name != p.Name {
			return fmt.Errorf("gpumodel: param %d is %q on the cpu, %q here", i, cpu[i].Name, p.Name)
		}
		p.W.Upload(cpu[i].Value)
	}
	return m.dev.Run(func(b *gpu.CommandBatch) {
		for _, p := range m.params {
			b.Zero(p.M)
			b.Zero(p.V)
		}
	})
}

// LoadOptimizer uploads the moment estimates of opt, which must have been
// created over the params of a model with the same Config.
func (m *Model) LoadOptimizer(opt *train.AdamW) {
	for i, p := range m.params {
		mo, vo := opt.Moments(i)
		p.M.Upload(mo)
		p.V.Upload(vo)
	}
}

// StoreTo downloads the weights into g and, when opt is non-nil, the
// moments and update count into opt, so train.Snapshot can checkpoint them.
func (m *Model) StoreTo(g *model.GPT, opt *train.AdamW, step int) error {
	cpu := g.Params()
	if len(cpu) != len(m.params) {
		return fmt.Errorf("gpumodel: %d cpu params, %d gpu params", len(cpu), len(m.params))
	}
	for i, p := range m.params {
		if err := p.W.Download(cpu[i].Value); err != nil {
			return err
		}
		if opt != nil {
			mo, vo := opt.Moments(i)
			if err := p.M.Download(mo); err != nil {
				return err
			}
			if err := p.V.Download(vo); err != nil {
				return err
			}
		}
	}
	if opt != nil {
		opt.SetStepCount(step)
	}
	return nil
}

// activations is every buffer a forward/backward pass of one batch shape
// touches. It is allocated once per (batch, seq) and reused across steps.
type activations struct {
	batch, seq int
	ids, pos   *gpu.Index // token ids over VocabSize, positions over ContextLen
	targets    *gpu.Index // next-token targets over VocabSize

	x0      *gpu.Tensor // embedding sum (after dropout), [N, D]
	embMask *gpu.Tensor // [N, D]
	layers  []*layerAct
	xhatF   *gpu.Tensor // [N, D]
	rstdF   *gpu.Tensor // [N, 1]
	hF      *gpu.Tensor // [N, D], LNF output
	logits  *gpu.Tensor // [N, V]; becomes dLogits in TrainStep
	losses  *gpu.Tensor // [N, 1]
	lossT   *gpu.Tensor // [1, 1]: mean loss
	lastRow *gpu.Index  // one row of logits, for Generate
	lastLog *gpu.Tensor // [1, V]: that row
	normT   *gpu.Tensor // [1, 1]: squared gradient norm

	// backward scratch, [N, D] unless noted
	dA, dB, dC *gpu.Tensor // rotating dy / dh / dx
	dMasked    *gpu.Tensor // dropout-masked copy of the incoming gradient
	dLN        *gpu.Tensor // gradient into a LayerNorm (dh2 / dh1)
	dAttOut    *gpu.Tensor
	dqkv       *gpu.Tensor // [N, 3D]
	dAct       *gpu.Tensor // [N, 4D]
	dScores    *gpu.Tensor // [B*H*T, T]
}

type layerAct struct {
	xhat1, rstd1, h1 *gpu.Tensor // LN1: [N,D], [N,1], [N,D]
	qkv              *gpu.Tensor // [N, 3D]
	scores           *gpu.Tensor // [B*H*T, T], attention weights after softmax
	attOut           *gpu.Tensor // [N, D], concatenated heads
	h                *gpu.Tensor // [N, D], residual after attention
	attnMask         *gpu.Tensor // [N, D]
	xhat2, rstd2, h2 *gpu.Tensor // LN2
	pre, actv        *gpu.Tensor // [N, 4D]: fc output and gelu(fc output)
	out              *gpu.Tensor // [N, D], block output
	mlpMask          *gpu.Tensor // [N, D]
}

func (a *activations) release() {
	a.ids.Release()
	a.pos.Release()
	a.targets.Release()
	a.lastRow.Release()
	ts := []*gpu.Tensor{a.x0, a.embMask, a.xhatF, a.rstdF, a.hF, a.logits, a.losses, a.lossT, a.normT, a.lastLog,
		a.dA, a.dB, a.dC, a.dMasked, a.dLN, a.dAttOut, a.dqkv, a.dAct, a.dScores}
	for _, l := range a.layers {
		ts = append(ts, l.xhat1, l.rstd1, l.h1, l.qkv, l.scores, l.attOut, l.h, l.attnMask,
			l.xhat2, l.rstd2, l.h2, l.pre, l.actv, l.out, l.mlpMask)
	}
	for _, t := range ts {
		t.Release()
	}
}

// ensure (re)allocates activations for a batch x seq input.
func (m *Model) ensure(batch, seq int) *activations {
	if m.act != nil && m.act.batch == batch && m.act.seq == seq {
		return m.act
	}
	if m.act != nil {
		m.act.release()
	}
	cfg := m.Config
	dev := m.dev
	n, d, h, v := batch*seq, cfg.EmbedDim, cfg.NumHeads, cfg.VocabSize
	zeros := make([]int, n)
	pos := make([]int, n)
	for i := range pos {
		pos[i] = i % seq
	}
	a := &activations{
		batch: batch, seq: seq,
		ids:     dev.NewIndex(zeros, v),
		pos:     dev.NewIndex(pos, cfg.ContextLen),
		targets: dev.NewIndex(zeros, v),
		lastRow: dev.NewIndex([]int{0}, n), lastLog: dev.NewTensor(1, v),
		x0: dev.NewTensor(n, d), embMask: dev.NewTensor(n, d),
		xhatF: dev.NewTensor(n, d), rstdF: dev.NewTensor(n, 1), hF: dev.NewTensor(n, d),
		logits: dev.NewTensor(n, v), losses: dev.NewTensor(n, 1), lossT: dev.NewTensor(1, 1), normT: dev.NewTensor(1, 1),
		dA: dev.NewTensor(n, d), dB: dev.NewTensor(n, d), dC: dev.NewTensor(n, d),
		dMasked: dev.NewTensor(n, d), dLN: dev.NewTensor(n, d), dAttOut: dev.NewTensor(n, d),
		dqkv: dev.NewTensor(n, 3*d), dAct: dev.NewTensor(n, 4*d), dScores: dev.NewTensor(batch*h*seq, seq),
	}
	for i := 0; i < cfg.NumLayers; i++ {
		a.layers = append(a.layers, &layerAct{
			xhat1: dev.NewTensor(n, d), rstd1: dev.NewTensor(n, 1), h1: dev.NewTensor(n, d),
			qkv: dev.NewTensor(n, 3*d), scores: dev.NewTensor(batch*h*seq, seq),
			attOut: dev.NewTensor(n, d), h: dev.NewTensor(n, d), attnMask: dev.NewTensor(n, d),
			xhat2: dev.NewTensor(n, d), rstd2: dev.NewTensor(n, 1), h2: dev.NewTensor(n, d),
			pre: dev.NewTensor(n, 4*d), actv: dev.NewTensor(n, 4*d),
			out: dev.NewTensor(n, d), mlpMask: dev.NewTensor(n, d),
		})
	}
	m.act = a
	return a
}

// flatten validates a batch and returns its ids as one slice.
func (m *Model) flatten(tokens [][]int) ([]int, int, int, error) {
	batch := len(tokens)
	if batch == 0 {
		return nil, 0, 0, fmt.Errorf("gpumodel: empty batch")
	}
	seq := len(tokens[0])
	if seq == 0 || seq > m.Config.ContextLen {
		return nil, 0, 0, fmt.Errorf("gpumodel: sequence length %d not in [1, %d]", seq, m.Config.ContextLen)
	}
	ids := make([]int, 0, batch*seq)
	for b, row := range tokens {
		if len(row) != seq {
			return nil, 0, 0, fmt.Errorf("gpumodel: ragged batch: row %d has length %d, want %d", b, len(row), seq)
		}
		for _, id := range row {
			if id < 0 || id >= m.Config.VocabSize {
				return nil, 0, 0, fmt.Errorf("gpumodel: token id %d out of range [0, %d)", id, m.Config.VocabSize)
			}
			ids = append(ids, id)
		}
	}
	return ids, batch, seq, nil
}

// dropout site seeds, so each site draws an independent mask per step.
const (
	seedEmb  = 1
	seedAttn = 100 // + 2*layer
	seedMLP  = 101 // + 2*layer
)

func (m *Model) dropping() bool { return m.training && m.Config.Dropout > 0 }

func attnScale(cfg model.Config) float64 { return 1 / math.Sqrt(float64(cfg.HeadDim())) }
