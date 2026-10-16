package gpumodel

import (
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

// TestStepperThroughRunStepper drives the GPU stepper with the shared
// training loop, checks it tracks a CPU run, and that the checkpoint it
// writes restores a CPU model and optimizer matching the GPU state.
func TestStepperThroughRunStepper(t *testing.T) {
	d := dev(t)
	cfg := model.Config{VocabSize: 6, ContextLen: 8, NumLayers: 2, NumHeads: 2, EmbedDim: 16}
	set := pattern{seq: []int{0, 1, 2, 3, 4, 5, 2}}
	batcher := batcherFunc(func(rng *rand.Rand, b, ctx int) ([][]int, [][]int) { return set.batch(rng, b, ctx) })
	oc := train.DefaultAdamW()
	oc.LR = 5e-3
	opts := train.Options{Steps: 12, BatchSize: 4, GradClip: 1, LR: train.Cosine(2, 12, 5e-3, 5e-4), EvalEvery: 6, EvalBatches: 2}

	// CPU reference run
	gc := model.New(cfg, rand.New(rand.NewSource(1)))
	oc1 := train.NewAdamW(gc.Params(), oc)
	var cpuLosses []float64
	require.NoError(t, train.Run(gc, oc1, batcher, batcher, opts, rand.New(rand.NewSource(2)), func(m train.Metrics) { cpuLosses = append(cpuLosses, m.Loss) }))

	// GPU run through the same loop, with a checkpoint at the end
	gg := model.New(cfg, rand.New(rand.NewSource(1)))
	og := train.NewAdamW(gg.Params(), oc)
	m, err := New(d, cfg)
	require.NoError(t, err)
	defer m.Close()
	st, err := NewStepper(m, gg, og, 1)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "gpu.ckpt")
	opts.CheckpointPath, opts.CheckpointEvery = path, 12
	opts.Tokenizer = []byte("tok")
	var gpuLosses, valLosses []float64
	require.NoError(t, train.RunStepper(st, gg, og, batcher, batcher, opts, rand.New(rand.NewSource(2)), func(mt train.Metrics) {
		gpuLosses = append(gpuLosses, mt.Loss)
		if mt.ValLoss == mt.ValLoss { // not NaN
			valLosses = append(valLosses, mt.ValLoss)
		}
	}))
	require.Len(t, gpuLosses, 12)
	for i := range cpuLosses {
		assert.InDeltaf(t, cpuLosses[i], gpuLosses[i], 2e-2*cpuLosses[i], "loss at step %d", i+1)
	}
	assert.Len(t, valLosses, 2)

	// the checkpoint restores to the GPU's weights and step count
	ck, err := train.Load(path)
	require.NoError(t, err)
	assert.Equal(t, 12, ck.Step)
	assert.Equal(t, []byte("tok"), ck.Tokenizer)
	gl, err := ck.Model()
	require.NoError(t, err)
	ol, err := ck.OptimizerFor(gl, oc)
	require.NoError(t, err)
	assert.Equal(t, 12, ol.StepCount())
	x, _ := set.batch(rand.New(rand.NewSource(3)), 2, 8)
	m.SetTraining(false)
	got, err := m.Forward(x)
	require.NoError(t, err)
	assertClose(t, "restored checkpoint logits", got, gl.Forward(x), 2e-4)

	// and a CPU resume from it keeps training without error
	require.NoError(t, train.Run(gl, ol, batcher, nil, train.Options{Steps: 14, StartStep: 12, BatchSize: 4, GradClip: 1}, rand.New(rand.NewSource(4)), nil))
	assert.Equal(t, 14, ol.StepCount())
}

type batcherFunc func(rng *rand.Rand, b, ctx int) ([][]int, [][]int)

func (f batcherFunc) Batch(rng *rand.Rand, b, ctx int) ([][]int, [][]int) { return f(rng, b, ctx) }
