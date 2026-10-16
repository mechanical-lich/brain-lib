package train

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
)

// fakeStepper records what RunStepper asks of it.
type fakeStepper struct {
	steps []int
	lrs   []float64
	evals int
	syncs []int
}

func (f *fakeStepper) TrainStep(x, y [][]int, lr float64, step int) (float64, float64, error) {
	f.steps = append(f.steps, step)
	f.lrs = append(f.lrs, lr)
	return 1.5, 0.25, nil
}
func (f *fakeStepper) Eval(x, y [][]int) (float64, error) { f.evals++; return 0.75, nil }
func (f *fakeStepper) Sync(step int) error                { f.syncs = append(f.syncs, step); return nil }

func TestRunStepperCadence(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	g := model.New(tinyCfg, rng)
	opt := NewAdamW(g.Params(), DefaultAdamW())
	set := patternBatcher{pattern: []int{0, 1, 2, 3, 4}}
	f := &fakeStepper{}
	path := filepath.Join(t.TempDir(), "m.ckpt")
	var metrics []Metrics
	err := RunStepper(f, g, opt, set, set, Options{
		Steps: 10, StartStep: 3, BatchSize: 2, LR: Constant(0.5),
		EvalEvery: 4, EvalBatches: 3, CheckpointEvery: 5, CheckpointPath: path,
	}, rng, func(m Metrics) { metrics = append(metrics, m) })
	require.NoError(t, err)

	assert.Equal(t, []int{4, 5, 6, 7, 8, 9, 10}, f.steps, "1-based global steps from StartStep")
	for _, lr := range f.lrs {
		assert.Equal(t, 0.5, lr)
	}
	// eval at steps 4, 8 and the last (10): 3 evals x 3 batches
	assert.Equal(t, 9, f.evals)
	// sync before each eval (4, 8, 10) and each checkpoint (5, 10), deduplicated per step
	assert.Equal(t, []int{4, 5, 8, 10}, f.syncs)
	require.Len(t, metrics, 7)
	assert.Equal(t, 4, metrics[0].Step)
	assert.Equal(t, 1.5, metrics[0].Loss)
	assert.Equal(t, 0.25, metrics[0].GradNorm)
	assert.False(t, math.IsNaN(metrics[0].ValLoss), "step 4 evaluated")
	assert.Equal(t, 0.75, metrics[0].ValLoss)
	assert.True(t, math.IsNaN(metrics[1].ValLoss), "step 5 not evaluated")
	ck, err := Load(path)
	require.NoError(t, err)
	assert.Equal(t, 10, ck.Step)
}

func TestCPUStepperMatchesRun(t *testing.T) {
	// Run and a hand-driven CPU stepper must produce identical models.
	set := patternBatcher{pattern: []int{0, 1, 2, 3, 4}}
	g1 := model.New(tinyCfg, rand.New(rand.NewSource(2)))
	g2 := model.New(tinyCfg, rand.New(rand.NewSource(2)))
	o1 := NewAdamW(g1.Params(), DefaultAdamW())
	o2 := NewAdamW(g2.Params(), DefaultAdamW())
	r1, r2 := rand.New(rand.NewSource(3)), rand.New(rand.NewSource(3))
	require.NoError(t, Run(g1, o1, set, nil, Options{Steps: 4, BatchSize: 2, GradClip: 1, LR: Cosine(1, 4, 1e-2, 1e-3)}, r1, nil))
	s := NewCPUStepper(g2, o2, 1)
	sched := Cosine(1, 4, 1e-2, 1e-3)
	for step := 0; step < 4; step++ {
		x, y := set.Batch(r2, 2, tinyCfg.ContextLen)
		_, _, err := s.TrainStep(x, y, sched(step), step+1)
		require.NoError(t, err)
	}
	for i, p := range g1.Params() {
		assert.Equal(t, p.Value.Data, g2.Params()[i].Value.Data, p.Name)
	}
	assert.Equal(t, o1.StepCount(), o2.StepCount())
}
