package train

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

var tinyCfg = model.Config{VocabSize: 5, ContextLen: 4, NumLayers: 1, NumHeads: 1, EmbedDim: 4}

func TestCrossEntropyValueAndGrad(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	logits := tensor.Randn(3, 5, 2, rng)
	targets := []int{4, 0, 2}
	loss, d := CrossEntropy(logits, targets)

	// value: average of -log softmax[target]
	var want float64
	for i, tgt := range targets {
		row := logits.Row(i)
		var sum float64
		for _, v := range row {
			sum += math.Exp(v)
		}
		want += math.Log(sum) - row[tgt]
	}
	assert.InDelta(t, want/3, loss, 1e-12)

	// gradient: finite differences
	const eps = 1e-6
	for idx := range logits.Data {
		orig := logits.Data[idx]
		logits.Data[idx] = orig + eps
		lp, _ := CrossEntropy(logits, targets)
		logits.Data[idx] = orig - eps
		lm, _ := CrossEntropy(logits, targets)
		logits.Data[idx] = orig
		assert.InDelta(t, (lp-lm)/(2*eps), d.Data[idx], 1e-8)
	}
	// each row of dLogits sums to zero
	for i := 0; i < 3; i++ {
		var s float64
		for _, v := range d.Row(i) {
			s += v
		}
		assert.InDelta(t, 0, s, 1e-12)
	}
	assert.Panics(t, func() { CrossEntropy(logits, []int{0}) })
	assert.Panics(t, func() { CrossEntropy(logits, []int{0, 1, 9}) })
}

// repeating stream: a fixed pattern the model should memorize quickly
type patternBatcher struct{ pattern []int }

func (p patternBatcher) Batch(rng *rand.Rand, b, ctx int) (x, y [][]int) {
	for i := 0; i < b; i++ {
		start := rng.Intn(len(p.pattern))
		xs := make([]int, ctx)
		ys := make([]int, ctx)
		for j := 0; j < ctx; j++ {
			xs[j] = p.pattern[(start+j)%len(p.pattern)]
			ys[j] = p.pattern[(start+j+1)%len(p.pattern)]
		}
		x = append(x, xs)
		y = append(y, ys)
	}
	return x, y
}

func TestRunOverfitsPattern(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	cfg := model.Config{VocabSize: 4, ContextLen: 8, NumLayers: 2, NumHeads: 2, EmbedDim: 16}
	g := model.New(cfg, rng)
	oc := DefaultAdamW()
	oc.LR = 1e-2
	opt := NewAdamW(g.Params(), oc)
	set := patternBatcher{pattern: []int{0, 1, 2, 3, 2, 1}}

	var first, last Metrics
	err := Run(g, opt, set, set, Options{Steps: 250, BatchSize: 8, GradClip: 1, EvalEvery: 250, EvalBatches: 2}, rng,
		func(m Metrics) {
			if m.Step == 1 {
				first = m
			}
			last = m
		})
	require.NoError(t, err)
	assert.Greater(t, first.Loss, 1.0, "initial loss near log(4)")
	assert.Less(t, last.Loss, 0.1, "should memorize the pattern")
	assert.Less(t, last.ValLoss, 0.1)
	assert.Equal(t, 250, opt.StepCount())

	// greedy continuation reproduces the pattern
	out := g.Generate([]int{0, 1, 2}, 6, model.GenerateOptions{}, rng)
	assert.Equal(t, []int{0, 1, 2, 3, 2, 1, 0, 1, 2}, out)
}

// TestRunWithDropout is the pattern-memorization smoke test with dropout on:
// training loss is noisier but must still fall well below its start, and the
// dropout-free validation loss should be low.
func TestRunWithDropout(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	cfg := model.Config{VocabSize: 4, ContextLen: 8, NumLayers: 2, NumHeads: 2, EmbedDim: 16, Dropout: 0.1}
	g := model.New(cfg, rng)
	oc := DefaultAdamW()
	oc.LR = 1e-2
	opt := NewAdamW(g.Params(), oc)
	set := patternBatcher{pattern: []int{0, 1, 2, 3, 2, 1}}

	var first, last Metrics
	err := Run(g, opt, set, set, Options{Steps: 250, BatchSize: 8, GradClip: 1, EvalEvery: 250, EvalBatches: 2}, rng,
		func(m Metrics) {
			if m.Step == 1 {
				first = m
			}
			last = m
		})
	require.NoError(t, err)
	t.Logf("loss %.3f -> %.3f, val %.4f", first.Loss, last.Loss, last.ValLoss)
	assert.False(t, g.Training(), "Run leaves the model in inference mode")
	assert.Greater(t, first.Loss, 1.0, "initial loss near log(4)")
	assert.Less(t, last.Loss, 0.3, "loss should fall despite dropout")
	assert.Less(t, last.ValLoss, 0.1)
	out := g.Generate([]int{0, 1, 2}, 6, model.GenerateOptions{}, rng)
	assert.Equal(t, []int{0, 1, 2, 3, 2, 1, 0, 1, 2}, out)
}

func TestClipGradNorm(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	g := model.New(tinyCfg, rng)
	for _, p := range g.Params() {
		p.Grad.Fill(1)
	}
	before := ClipGradNorm(g.Params(), 1)
	assert.InDelta(t, math.Sqrt(float64(g.NumParams())), before, 1e-9)
	assert.InDelta(t, 1, ClipGradNorm(g.Params(), 100), 1e-5, "after clipping norm is maxNorm")
}

func TestCosineSchedule(t *testing.T) {
	s := Cosine(10, 110, 1.0, 0.1)
	assert.InDelta(t, 0.1, s(0), 1e-12, "warmup starts at max/warmup")
	assert.InDelta(t, 1.0, s(9), 1e-12, "warmup ends at max")
	assert.InDelta(t, 0.55, s(60), 1e-12, "midpoint of cosine")
	assert.InDelta(t, 0.1, s(110), 1e-12)
	assert.InDelta(t, 0.1, s(5000), 1e-12)
	assert.InDelta(t, 0.5, Constant(0.5)(123), 0)
}

func TestCheckpointRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	g := model.New(tinyCfg, rng)
	opt := NewAdamW(g.Params(), DefaultAdamW())
	set := patternBatcher{pattern: []int{0, 1, 2, 3, 4}}
	require.NoError(t, Run(g, opt, set, nil, Options{Steps: 3, BatchSize: 2}, rng, nil))

	path := filepath.Join(t.TempDir(), "sub", "model.ckpt")
	tokBytes := []byte(`{"kind":"char"}`)
	require.NoError(t, Save(path, Snapshot(g, opt, 3, tokBytes)))

	ck, err := Load(path)
	require.NoError(t, err)
	assert.Equal(t, 3, ck.Step)
	assert.Equal(t, tinyCfg, ck.Config)
	assert.Equal(t, tokBytes, ck.Tokenizer)

	g2, err := ck.Model()
	require.NoError(t, err)
	for i, p := range g.Params() {
		assert.Equal(t, p.Value.Data, g2.Params()[i].Value.Data, p.Name)
	}
	opt2, err := ck.OptimizerFor(g2, DefaultAdamW())
	require.NoError(t, err)
	assert.Equal(t, opt.StepCount(), opt2.StepCount())
	for i := range opt.m {
		assert.Equal(t, opt.m[i].Data, opt2.m[i].Data)
		assert.Equal(t, opt.v[i].Data, opt2.v[i].Data)
	}

	// identical forward outputs, and identical next step when resumed
	x := [][]int{{0, 1, 2, 3}}
	assert.Equal(t, g.Forward(x).Data, g2.Forward(x).Data)
	r1, r2 := rand.New(rand.NewSource(9)), rand.New(rand.NewSource(9))
	require.NoError(t, Run(g, opt, set, nil, Options{Steps: 4, StartStep: 3, BatchSize: 2}, r1, nil))
	require.NoError(t, Run(g2, opt2, set, nil, Options{Steps: 4, StartStep: 3, BatchSize: 2}, r2, nil))
	assert.InDeltaSlice(t, g.Forward(x).Data, g2.Forward(x).Data, 1e-12)

	_, err = Load(filepath.Join(t.TempDir(), "missing.ckpt"))
	assert.Error(t, err)
}

func TestCheckpointDropout(t *testing.T) {
	cfg := tinyCfg
	cfg.Dropout = 0.2
	g := model.New(cfg, rand.New(rand.NewSource(4)))
	path := filepath.Join(t.TempDir(), "model.ckpt")
	require.NoError(t, Save(path, Snapshot(g, nil, 0, nil)))
	ck, err := Load(path)
	require.NoError(t, err)
	assert.Equal(t, cfg, ck.Config)
	g2, err := ck.Model()
	require.NoError(t, err)
	assert.Equal(t, 0.2, g2.Config.Dropout)
	assert.False(t, g2.Training())

	// A checkpoint written before Config had Dropout decodes with Dropout 0.
	// gob matches fields by name, so a struct without the field stands in
	// for the old format.
	type oldConfig struct{ VocabSize, ContextLen, NumLayers, NumHeads, EmbedDim int }
	type oldCheckpoint struct {
		Config oldConfig
		Step   int
		Params map[string]MatData
	}
	old := oldCheckpoint{Config: oldConfig{tinyCfg.VocabSize, tinyCfg.ContextLen, tinyCfg.NumLayers, tinyCfg.NumHeads, tinyCfg.EmbedDim}, Step: 7, Params: ck.Params}
	f, err := os.Create(path)
	require.NoError(t, err)
	require.NoError(t, gob.NewEncoder(f).Encode(old))
	require.NoError(t, f.Close())
	ck, err = Load(path)
	require.NoError(t, err)
	assert.Equal(t, tinyCfg, ck.Config)
	assert.Equal(t, 0.0, ck.Config.Dropout)
	_, err = ck.Model()
	require.NoError(t, err)
}

func TestRunRejectsBadOptions(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	g := model.New(tinyCfg, rng)
	opt := NewAdamW(g.Params(), DefaultAdamW())
	set := patternBatcher{pattern: []int{0, 1}}
	assert.Error(t, Run(g, opt, set, nil, Options{Steps: 1}, rng, nil), "batch size 0")
	assert.Error(t, Run(g, opt, set, nil, Options{Steps: 1, BatchSize: 1, ContextLen: 99}, rng, nil), "ctx too long")
}
