package gpumodel

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/gpu"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

var (
	testDev    *gpu.Device
	testDevErr error
)

func TestMain(m *testing.M) {
	testDev, testDevErr = gpu.Open()
	if testDevErr == nil {
		fmt.Fprintf(os.Stderr, "gpumodel: testing on %s\n", testDev.Name())
	}
	code := m.Run()
	if testDev != nil {
		testDev.Close()
	}
	os.Exit(code)
}

func dev(tb testing.TB) *gpu.Device {
	tb.Helper()
	if testDevErr != nil {
		if errors.Is(testDevErr, gpu.ErrNoAdapter) {
			tb.Skipf("skipping GPU model test: %v", testDevErr)
		}
		tb.Fatalf("gpu.Open: %v", testDevErr)
	}
	return testDev
}

// Two shapes: a tiny one and one with several heads, a ragged vocabulary
// and more than one row block, so the batched attention layout is exercised.
var (
	tinyCfg = model.Config{VocabSize: 11, ContextLen: 8, NumLayers: 2, NumHeads: 2, EmbedDim: 16}
	midCfg  = model.Config{VocabSize: 100, ContextLen: 40, NumLayers: 3, NumHeads: 4, EmbedDim: 64}
)

// newPair builds a CPU model with weights scaled up so the nonlinearities
// are exercised away from zero, and a GPU copy of it.
func newPair(tb testing.TB, d *gpu.Device, cfg model.Config, seed int64) (*model.GPT, *Model) {
	tb.Helper()
	rng := rand.New(rand.NewSource(seed))
	g := model.New(cfg, rng)
	for _, p := range g.Params() {
		if p.Decay {
			p.Value.Scale(8)
		}
	}
	m, err := New(d, cfg)
	require.NoError(tb, err)
	require.NoError(tb, m.LoadFrom(g))
	return g, m
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

func flatten(rows [][]int) []int {
	var out []int
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// assertClose checks max|got-want| relative to max|want|.
func assertClose(tb testing.TB, name string, got, want *tensor.Mat, tol float64) {
	tb.Helper()
	require.Truef(tb, got.SameShape(want), "%s: shape %s, want %s", name, got.Shape(), want.Shape())
	var maxDiff, maxRef float64
	worst := 0
	for i, w := range want.Data {
		gv := got.Data[i]
		require.Falsef(tb, math.IsNaN(gv) || math.IsInf(gv, 0), "%s: element %d is %v", name, i, gv)
		if d := math.Abs(gv - w); d > maxDiff {
			maxDiff, worst = d, i
		}
		maxRef = max(maxRef, math.Abs(w))
	}
	rel := maxDiff / max(maxRef, 1e-30)
	assert.LessOrEqualf(tb, rel, tol, "%s: max |gpu-cpu| %.3g at %d (gpu %v, cpu %v), relative %.2e", name, maxDiff, worst, got.Data[worst], want.Data[worst], rel)
}

func TestParamOrderMatchesCPU(t *testing.T) {
	d := dev(t)
	g := model.New(midCfg, rand.New(rand.NewSource(1)))
	m, err := New(d, midCfg)
	require.NoError(t, err)
	defer m.Close()
	cpu := g.Params()
	require.Len(t, m.Params(), len(cpu))
	for i, p := range m.Params() {
		assert.Equal(t, cpu[i].Name, p.Name)
		assert.Equal(t, cpu[i].Decay, p.Decay)
		assert.Equal(t, cpu[i].Value.Rows, p.W.Rows)
		assert.Equal(t, cpu[i].Value.Cols, p.W.Cols)
	}
	assert.Equal(t, g.NumParams(), m.NumParams())
	bad := midCfg
	bad.NumHeads = 3
	_, err = New(d, bad)
	assert.Error(t, err)
}

func TestForwardMatchesCPU(t *testing.T) {
	d := dev(t)
	for _, c := range []struct {
		name  string
		cfg   model.Config
		b, tt int
	}{{"tiny", tinyCfg, 2, 8}, {"mid", midCfg, 3, 40}, {"short-seq", midCfg, 2, 7}} {
		t.Run(c.name, func(t *testing.T) {
			g, m := newPair(t, d, c.cfg, 2)
			defer m.Close()
			rng := rand.New(rand.NewSource(3))
			tokens := randomTokens(rng, c.b, c.tt, c.cfg.VocabSize)
			want := g.Forward(tokens)
			got, err := m.Forward(tokens)
			require.NoError(t, err)
			assertClose(t, "logits", got, want, 2e-4)
			// a second call with new tokens reuses the activations
			tokens = randomTokens(rng, c.b, c.tt, c.cfg.VocabSize)
			got, err = m.Forward(tokens)
			require.NoError(t, err)
			assertClose(t, "logits (2nd batch)", got, g.Forward(tokens), 2e-4)
		})
	}
}

func TestBackwardMatchesCPU(t *testing.T) {
	d := dev(t)
	for _, c := range []struct {
		name  string
		cfg   model.Config
		b, tt int
	}{{"tiny", tinyCfg, 2, 8}, {"mid", midCfg, 3, 40}} {
		t.Run(c.name, func(t *testing.T) {
			g, m := newPair(t, d, c.cfg, 4)
			defer m.Close()
			rng := rand.New(rand.NewSource(5))
			tokens := randomTokens(rng, c.b, c.tt, c.cfg.VocabSize)
			logits := g.Forward(tokens)
			_, err := m.Forward(tokens)
			require.NoError(t, err)
			dLogits := tensor.Randn(logits.Rows, logits.Cols, 1e-2, rng)
			g.ZeroGrads()
			g.Backward(dLogits)
			require.NoError(t, m.Backward(dLogits))
			grads, err := m.Grads()
			require.NoError(t, err)
			for i, p := range g.Params() {
				assertClose(t, p.Name, grads[i], p.Grad, 2e-3)
			}
		})
	}
}

// cpuStep is the CPU training step TrainStep is compared against.
func cpuStep(g *model.GPT, opt *train.AdamW, x, y [][]int, maxNorm float64) (loss, norm float64) {
	g.ZeroGrads()
	loss, dl := train.CrossEntropy(g.Forward(x), flatten(y))
	g.Backward(dl)
	norm = train.ClipGradNorm(g.Params(), maxNorm)
	opt.Step()
	return loss, norm
}

func TestTrainStepMatchesCPU(t *testing.T) {
	d := dev(t)
	for _, maxNorm := range []float64{0, 0.5} {
		t.Run(fmt.Sprintf("clip=%g", maxNorm), func(t *testing.T) {
			g, m := newPair(t, d, midCfg, 6)
			defer m.Close()
			oc := train.DefaultAdamW()
			oc.LR = 1e-3
			opt := train.NewAdamW(g.Params(), oc)
			rng := rand.New(rand.NewSource(7))
			const steps = 5
			for step := 1; step <= steps; step++ {
				x := randomTokens(rng, 4, 40, midCfg.VocabSize)
				y := randomTokens(rng, 4, 40, midCfg.VocabSize)
				wantLoss, wantNorm := cpuStep(g, opt, x, y, maxNorm)
				got, err := m.TrainStep(x, y, oc, maxNorm, step)
				require.NoError(t, err)
				assert.InDeltaf(t, wantLoss, got.Loss, 2e-3*wantLoss, "loss at step %d", step)
				assert.InDeltaf(t, wantNorm, got.GradNorm, 5e-3*wantNorm, "grad norm at step %d", step)
			}
			// Adam normalizes each element's update to about lr, so an element
			// whose float32 gradient differs in sign from the float64 one moves
			// 2*lr the wrong way; a few such elements are expected, most of
			// them in the bias vectors whose gradients are small column sums.
			// Bound the worst element loosely and the mean more tightly.
			maxTol := 3 * oc.LR * steps
			meanTol := oc.LR / 5
			for i, p := range g.Params() {
				got, err := m.Params()[i].W.ToMat()
				require.NoError(t, err)
				var maxDiff, sum float64
				for j, w := range p.Value.Data {
					dd := math.Abs(got.Data[j] - w)
					maxDiff = max(maxDiff, dd)
					sum += dd
				}
				assert.LessOrEqualf(t, maxDiff, maxTol, "%s: max weight diff %.3g", p.Name, maxDiff)
				assert.LessOrEqualf(t, sum/float64(len(p.Value.Data)), meanTol, "%s: mean weight diff %.3g", p.Name, sum/float64(len(p.Value.Data)))
			}
		})
	}
}

// pattern yields windows of a fixed repeating sequence, as in the train tests.
type pattern struct{ seq []int }

func (p pattern) batch(rng *rand.Rand, b, ctx int) (x, y [][]int) {
	for i := 0; i < b; i++ {
		start := rng.Intn(len(p.seq))
		xs, ys := make([]int, ctx), make([]int, ctx)
		for j := 0; j < ctx; j++ {
			xs[j] = p.seq[(start+j)%len(p.seq)]
			ys[j] = p.seq[(start+j+1)%len(p.seq)]
		}
		x, y = append(x, xs), append(y, ys)
	}
	return x, y
}

func TestOverfitsPatternWithDropout(t *testing.T) {
	d := dev(t)
	cfg := model.Config{VocabSize: 4, ContextLen: 8, NumLayers: 2, NumHeads: 2, EmbedDim: 16, Dropout: 0.1}
	g := model.New(cfg, rand.New(rand.NewSource(8)))
	m, err := New(d, cfg)
	require.NoError(t, err)
	defer m.Close()
	require.NoError(t, m.LoadFrom(g))
	m.SetTraining(true)
	oc := train.DefaultAdamW()
	oc.LR = 1e-2
	rng := rand.New(rand.NewSource(9))
	set := pattern{seq: []int{0, 1, 2, 3, 2, 1}}
	var first, last StepResult
	for step := 1; step <= 250; step++ {
		x, y := set.batch(rng, 8, 8)
		r, err := m.TrainStep(x, y, oc, 1, step)
		require.NoError(t, err)
		if step == 1 {
			first = r
		}
		last = r
	}
	assert.Greater(t, first.Loss, 1.0, "initial loss near ln(4)")
	assert.Less(t, last.Loss, 0.15, "memorizes the pattern")
	x, y := set.batch(rng, 8, 8)
	ev, err := m.Eval(x, y)
	require.NoError(t, err)
	assert.Less(t, ev, 0.15, "eval loss without dropout")

	// eval-mode logits are the same whether or not training mode is on,
	// once training mode is off
	m.SetTraining(false)
	l1, err := m.Forward(x)
	require.NoError(t, err)
	l2, err := m.Forward(x)
	require.NoError(t, err)
	assert.Equal(t, l1.Data, l2.Data)
}

func TestStoreToRoundTrip(t *testing.T) {
	d := dev(t)
	g, m := newPair(t, d, tinyCfg, 10)
	defer m.Close()
	oc := train.DefaultAdamW()
	rng := rand.New(rand.NewSource(11))
	for step := 1; step <= 3; step++ {
		x := randomTokens(rng, 2, 8, tinyCfg.VocabSize)
		y := randomTokens(rng, 2, 8, tinyCfg.VocabSize)
		_, err := m.TrainStep(x, y, oc, 1, step)
		require.NoError(t, err)
	}
	// store into a fresh CPU model + optimizer, then compare forward passes
	g2 := model.New(tinyCfg, rand.New(rand.NewSource(99)))
	opt2 := train.NewAdamW(g2.Params(), oc)
	require.NoError(t, m.StoreTo(g2, opt2, 3))
	assert.Equal(t, 3, opt2.StepCount())
	x := randomTokens(rng, 2, 8, tinyCfg.VocabSize)
	got, err := m.Forward(x)
	require.NoError(t, err)
	assertClose(t, "stored model logits", got, g2.Forward(x), 2e-4)
	m1, _ := opt2.Moments(2)
	assert.NotEqual(t, 0.0, m1.Norm(), "moments were downloaded")
	_ = g

	// and the reverse: load the optimizer state back and keep training
	m2, err := New(d, tinyCfg)
	require.NoError(t, err)
	defer m2.Close()
	require.NoError(t, m2.LoadFrom(g2))
	m2.LoadOptimizer(opt2)
	y := randomTokens(rng, 2, 8, tinyCfg.VocabSize)
	r1, err := m.TrainStep(x, y, oc, 1, 4)
	require.NoError(t, err)
	r2, err := m2.TrainStep(x, y, oc, 1, 4)
	require.NoError(t, err)
	assert.InDelta(t, r1.Loss, r2.Loss, 1e-5)
	w1, _ := m.Params()[0].W.ToMat()
	w2, _ := m2.Params()[0].W.ToMat()
	assertClose(t, "weights after restored step", w2, w1, 1e-5)
}

func TestInputErrors(t *testing.T) {
	d := dev(t)
	_, m := newPair(t, d, tinyCfg, 12)
	defer m.Close()
	_, err := m.Forward(nil)
	assert.Error(t, err)
	_, err = m.Forward([][]int{{0, 1}, {0}})
	assert.Error(t, err)
	_, err = m.Forward([][]int{{tinyCfg.VocabSize}})
	assert.Error(t, err)
	_, err = m.Forward([][]int{make([]int, tinyCfg.ContextLen+1)})
	assert.Error(t, err)
	_, err = m.TrainStep([][]int{{0, 1}}, [][]int{{0, 1, 2}}, train.DefaultAdamW(), 0, 1)
	assert.Error(t, err)
}

// BenchmarkCPUTrainStep is the CPU step at the same shape, for the readme.
func BenchmarkCPUTrainStep(b *testing.B) {
	cfg := model.Config{VocabSize: 2000, ContextLen: 256, NumLayers: 6, NumHeads: 6, EmbedDim: 384, Dropout: 0.1}
	g := model.New(cfg, rand.New(rand.NewSource(1)))
	g.SetTraining(true)
	opt := train.NewAdamW(g.Params(), train.DefaultAdamW())
	rng := rand.New(rand.NewSource(2))
	x := randomTokens(rng, 16, 256, cfg.VocabSize)
	y := randomTokens(rng, 16, 256, cfg.VocabSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpuStep(g, opt, x, y, 1)
	}
	b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N), "ms/step")
}

// BenchmarkTrainStep is the BPE Lovecraft run's shape: 6 layers, width 384,
// 6 heads, vocabulary 2000, context 256, batch 16, dropout 0.1.
func BenchmarkTrainStep(b *testing.B) {
	d := dev(b)
	cfg := model.Config{VocabSize: 2000, ContextLen: 256, NumLayers: 6, NumHeads: 6, EmbedDim: 384, Dropout: 0.1}
	g := model.New(cfg, rand.New(rand.NewSource(1)))
	m, err := New(d, cfg)
	require.NoError(b, err)
	defer m.Close()
	require.NoError(b, m.LoadFrom(g))
	m.SetTraining(true)
	rng := rand.New(rand.NewSource(2))
	x := randomTokens(rng, 16, 256, cfg.VocabSize)
	y := randomTokens(rng, 16, 256, cfg.VocabSize)
	oc := train.DefaultAdamW()
	if _, err := m.TrainStep(x, y, oc, 1, 1); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.TrainStep(x, y, oc, 1, i+2); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N), "ms/step")
}

func TestGenerateMatchesCPUGreedy(t *testing.T) {
	d := dev(t)
	g, m := newPair(t, d, midCfg, 31)
	defer m.Close()
	m.SetTraining(true) // Generate must switch dropout off and restore the mode
	for _, prompt := range [][]int{{1}, {5, 9, 2}, {7, 7, 7, 7, 7, 7}} {
		want := g.Generate(prompt, 12, model.GenerateOptions{}, nil)
		got, err := m.Generate(prompt, 12, model.GenerateOptions{}, nil)
		require.NoError(t, err)
		assert.Equal(t, want, got, "greedy continuation of %v", prompt)
	}
	assert.True(t, m.training, "training mode restored")
	// the window exceeds ContextLen and gets truncated like the CPU
	long := make([]int, midCfg.ContextLen+5)
	for i := range long {
		long[i] = i % midCfg.VocabSize
	}
	want := g.Generate(long, 3, model.GenerateOptions{}, nil)
	got, err := m.Generate(long, 3, model.GenerateOptions{}, nil)
	require.NoError(t, err)
	assert.Equal(t, want, got)
	// sampling returns ids in range of the right length
	rng := rand.New(rand.NewSource(3))
	got, err = m.Generate([]int{1, 2}, 10, model.GenerateOptions{Temperature: 1, TopK: 5}, rng)
	require.NoError(t, err)
	require.Len(t, got, 12)
	for _, id := range got {
		assert.GreaterOrEqual(t, id, 0)
		assert.Less(t, id, midCfg.VocabSize)
	}
	_, err = m.Generate(nil, 1, model.GenerateOptions{}, rng)
	assert.Error(t, err)
}
