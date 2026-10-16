package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

var tinyCfg = Config{VocabSize: 7, ContextLen: 5, NumLayers: 2, NumHeads: 2, EmbedDim: 4}

// testLoss is a stand-in for cross-entropy: mean over all logits of
// logit * weight, with a fixed random weight matrix, so that dLogits is just
// weights/N. This keeps the gradient check independent of the train package.
func testLoss(logits *tensor.Mat, w *tensor.Mat) (float64, *tensor.Mat) {
	n := float64(len(logits.Data))
	var loss float64
	d := tensor.New(logits.Rows, logits.Cols)
	for i, v := range logits.Data {
		loss += v * w.Data[i] / n
		d.Data[i] = w.Data[i] / n
	}
	return loss, d
}

func randomTokens(rng *rand.Rand, b, t, vocab int) [][]int {
	out := make([][]int, b)
	for i := range out {
		out[i] = make([]int, t)
		for j := range out[i] {
			out[i][j] = rng.Intn(vocab)
		}
	}
	return out
}

func TestConfigValidate(t *testing.T) {
	assert.NoError(t, tinyCfg.Validate())
	bad := tinyCfg
	bad.NumHeads = 3
	assert.Error(t, bad.Validate())
	bad = tinyCfg
	bad.VocabSize = 0
	assert.Error(t, bad.Validate())
}

// TestGradientCheck compares analytic gradients of every parameter against
// central finite differences on a tiny model. Because everything is float64
// the agreement should be very tight.
func TestGradientCheck(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	g := New(tinyCfg, rng)
	// Scale weights up so the nonlinearities (GELU, softmax) are exercised
	// away from their near-linear regime around zero.
	for _, p := range g.Params() {
		if p.Decay {
			p.Value.Scale(10)
		}
	}
	tokens := randomTokens(rng, 2, 4, tinyCfg.VocabSize)
	w := tensor.Randn(2*4, tinyCfg.VocabSize, 1, rng)

	g.ZeroGrads()
	_, dLogits := testLoss(g.Forward(tokens), w)
	g.Backward(dLogits)

	const eps = 1e-5
	checked := 0
	for _, p := range g.Params() {
		// check up to 6 random entries per parameter
		n := p.Size()
		for k := 0; k < 6 && k < n; k++ {
			idx := rng.Intn(n)
			orig := p.Value.Data[idx]
			p.Value.Data[idx] = orig + eps
			lp, _ := testLoss(g.Forward(tokens), w)
			p.Value.Data[idx] = orig - eps
			lm, _ := testLoss(g.Forward(tokens), w)
			p.Value.Data[idx] = orig
			numeric := (lp - lm) / (2 * eps)
			analytic := p.Grad.Data[idx]
			tol := 1e-7 + 1e-5*math.Abs(analytic)
			assert.InDeltaf(t, numeric, analytic, tol, "param %s[%d]: numeric %g analytic %g", p.Name, idx, numeric, analytic)
			checked++
		}
	}
	require.Greater(t, checked, 50)
}

// TestCausalMask checks that logits at position t do not depend on tokens
// after t.
func TestCausalMask(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	g := New(tinyCfg, rng)
	a := randomTokens(rng, 1, 5, tinyCfg.VocabSize)
	b := [][]int{append([]int(nil), a[0]...)}
	b[0][3] = (b[0][3] + 1) % tinyCfg.VocabSize
	b[0][4] = (b[0][4] + 2) % tinyCfg.VocabSize

	la, lb := g.Forward(a), g.Forward(b)
	for pos := 0; pos < 3; pos++ {
		assert.InDeltaSlice(t, la.Row(pos), lb.Row(pos), 1e-12, "position %d leaked future tokens", pos)
	}
	assert.NotEqual(t, la.Row(3), lb.Row(3), "changed token should change its own logits")
}

func TestForwardShapeAndPanics(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	g := New(tinyCfg, rng)
	logits := g.Forward(randomTokens(rng, 3, 5, tinyCfg.VocabSize))
	assert.Equal(t, 15, logits.Rows)
	assert.Equal(t, tinyCfg.VocabSize, logits.Cols)
	assert.Panics(t, func() { g.Forward(randomTokens(rng, 1, 6, tinyCfg.VocabSize)) }, "too long")
	assert.Panics(t, func() { g.Forward([][]int{{0, 1}, {0}}) }, "ragged")
	assert.Panics(t, func() { g.Forward([][]int{{tinyCfg.VocabSize}}) }, "id out of range")
	assert.Panics(t, func() { New(Config{VocabSize: 1, ContextLen: 1, NumLayers: 1, NumHeads: 2, EmbedDim: 3}, rng) })
}

func TestNumParamsAndOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	a, b := New(tinyCfg, rng), New(tinyCfg, rng)
	pa, pb := a.Params(), b.Params()
	require.Equal(t, len(pa), len(pb))
	for i := range pa {
		assert.Equal(t, pa[i].Name, pb[i].Name, "Params order must be deterministic")
	}
	// tok_emb + pos_emb + per-block (2 LN * 2 + qkv w,b + proj w,b + fc w,b + proj w,b) + ln_f
	d := tinyCfg.EmbedDim
	perBlock := 2*(2*d) + (d*3*d + 3*d) + (d*d + d) + (d*4*d + 4*d) + (4*d*d + d)
	want := tinyCfg.VocabSize*d + tinyCfg.ContextLen*d + tinyCfg.NumLayers*perBlock + 2*d
	assert.Equal(t, want, a.NumParams())
}

func TestGenerate(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	g := New(tinyCfg, rng)
	out := g.Generate([]int{1, 2}, 10, GenerateOptions{Temperature: 1, TopK: 3}, rng)
	require.Len(t, out, 12)
	assert.Equal(t, []int{1, 2}, out[:2])
	for _, id := range out {
		assert.GreaterOrEqual(t, id, 0)
		assert.Less(t, id, tinyCfg.VocabSize)
	}
	greedy1 := g.Generate([]int{1}, 8, GenerateOptions{}, rng)
	greedy2 := g.Generate([]int{1}, 8, GenerateOptions{}, rng)
	assert.Equal(t, greedy1, greedy2, "greedy decoding is deterministic")
}

func TestSampleTopKAndGreedy(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	logits := []float64{0, 5, 1, 4}
	assert.Equal(t, 1, sample(logits, GenerateOptions{}, rng))
	assert.Equal(t, 1, Sample(logits, GenerateOptions{Temperature: 0}, rng), "exported Sample is greedy at temperature 0")
	for i := 0; i < 50; i++ {
		id := Sample(logits, GenerateOptions{Temperature: 1, TopK: 2}, rng)
		assert.Contains(t, []int{1, 3}, id)
	}
	seen := map[int]bool{}
	for i := 0; i < 200; i++ {
		seen[Sample(logits, GenerateOptions{Temperature: 1}, rng)] = true
	}
	assert.True(t, seen[1] && seen[3], "temperature 1 without top-k draws the likely ids")
}

func TestGeluGrad(t *testing.T) {
	for _, x := range []float64{-3, -0.5, 0, 0.7, 2.5} {
		const eps = 1e-6
		num := (gelu(x+eps) - gelu(x-eps)) / (2 * eps)
		assert.InDelta(t, num, geluGrad(x), 1e-7)
	}
}

func TestConfigValidateDropout(t *testing.T) {
	for _, p := range []float64{0, 0.1, 0.5, 0.999} {
		c := tinyCfg
		c.Dropout = p
		assert.NoError(t, c.Validate(), "dropout %g", p)
	}
	for _, p := range []float64{-0.1, 1, 1.5, math.NaN()} {
		c := tinyCfg
		c.Dropout = p
		assert.Error(t, c.Validate(), "dropout %g", p)
	}
}

// TestGradientCheckDropout is TestGradientCheck with dropout active. The rng
// is reseeded identically before every Forward so each pass draws the same
// masks, which makes the loss a smooth function of the weights again.
func TestGradientCheckDropout(t *testing.T) {
	cfg := tinyCfg
	cfg.Dropout = 0.3
	rng := rand.New(rand.NewSource(43))
	g := New(cfg, rng)
	for _, p := range g.Params() {
		if p.Decay {
			p.Value.Scale(10)
		}
	}
	g.SetTraining(true)
	tokens := randomTokens(rng, 2, 4, cfg.VocabSize)
	w := tensor.Randn(2*4, cfg.VocabSize, 1, rng)
	forward := func() *tensor.Mat {
		g.SetRNG(rand.New(rand.NewSource(1234)))
		return g.Forward(tokens)
	}

	g.ZeroGrads()
	_, dLogits := testLoss(forward(), w)
	g.Backward(dLogits)

	// the masks must actually drop something, or this is just TestGradientCheck
	zeros := 0
	for _, m := range g.embDrop.mask {
		if m == 0 {
			zeros++
		}
	}
	require.Greater(t, zeros, 0)

	const eps = 1e-5
	checked := 0
	for _, p := range g.Params() {
		n := p.Size()
		for k := 0; k < 6 && k < n; k++ {
			idx := rng.Intn(n)
			orig := p.Value.Data[idx]
			p.Value.Data[idx] = orig + eps
			lp, _ := testLoss(forward(), w)
			p.Value.Data[idx] = orig - eps
			lm, _ := testLoss(forward(), w)
			p.Value.Data[idx] = orig
			numeric := (lp - lm) / (2 * eps)
			analytic := p.Grad.Data[idx]
			tol := 1e-7 + 1e-5*math.Abs(analytic)
			assert.InDeltaf(t, numeric, analytic, tol, "param %s[%d]: numeric %g analytic %g", p.Name, idx, numeric, analytic)
			checked++
		}
	}
	require.Greater(t, checked, 50)
}

// TestDropoutTrainingMode checks that dropout only acts in training mode and
// that its masks come from the model's rng.
func TestDropoutTrainingMode(t *testing.T) {
	cfg := tinyCfg
	cfg.Dropout = 0.5
	rng := rand.New(rand.NewSource(5))
	g := New(cfg, rng)
	plain := New(tinyCfg, rand.New(rand.NewSource(5)))
	tokens := randomTokens(rng, 2, 5, cfg.VocabSize)

	assert.False(t, g.Training(), "models start in inference mode")
	eval := g.Forward(tokens)
	assert.Equal(t, plain.Forward(tokens).Data, eval.Data, "inference mode ignores Dropout")

	g.SetTraining(true)
	g.SetRNG(rand.New(rand.NewSource(1)))
	a := g.Forward(tokens)
	b := g.Forward(tokens)
	g.SetRNG(rand.New(rand.NewSource(1)))
	c := g.Forward(tokens)
	assert.NotEqual(t, eval.Data, a.Data, "training mode drops activations")
	assert.NotEqual(t, a.Data, b.Data, "fresh masks each Forward")
	assert.Equal(t, a.Data, c.Data, "same seed, same masks")

	g.SetTraining(false)
	assert.Equal(t, eval.Data, g.Forward(tokens).Data)
}
