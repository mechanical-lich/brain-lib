package train

import (
	"fmt"
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
)

// Batcher yields training batches: x and y are each batchSize sequences of
// contextLen token ids, with y[i][t] the target for x[i][t] (usually the
// next token).
type Batcher interface {
	Batch(rng *rand.Rand, batchSize, contextLen int) (x, y [][]int)
}

// Options controls Run.
type Options struct {
	Steps      int // total optimizer steps to run (including StartStep)
	StartStep  int // step to resume from (0 for a fresh run)
	BatchSize  int
	ContextLen int // sequence length per sample; 0 means the model's ContextLen

	GradClip float64  // max global grad norm; 0 disables clipping
	LR       Schedule // nil means the optimizer's configured LR

	EvalEvery   int // run validation every N steps; 0 disables
	EvalBatches int // batches averaged per validation

	CheckpointEvery int    // save every N steps (and at the end); 0 disables
	CheckpointPath  string // where to save
	Tokenizer       []byte // opaque tokenizer bytes stored in checkpoints
}

// Metrics is reported after each step.
type Metrics struct {
	Step     int
	Loss     float64
	ValLoss  float64 // NaN unless validation ran this step
	LR       float64
	GradNorm float64
}

// Run trains g with opt for opts.Steps steps, calling onStep after each.
// val may be nil to disable validation. The caller owns rng so runs are
// reproducible. g is in training mode (dropout on) only for each step's
// forward/backward and is left in inference mode.
func Run(g *model.GPT, opt *AdamW, trainSet, val Batcher, opts Options, rng *rand.Rand, onStep func(Metrics)) error {
	if opts.BatchSize <= 0 {
		return fmt.Errorf("train: BatchSize must be > 0")
	}
	ctx := opts.ContextLen
	if ctx <= 0 {
		ctx = g.Config.ContextLen
	}
	if ctx > g.Config.ContextLen {
		return fmt.Errorf("train: ContextLen %d exceeds model ContextLen %d", ctx, g.Config.ContextLen)
	}
	lr := opts.LR
	if lr == nil {
		lr = Constant(opt.Config.LR)
	}
	nan := func() float64 { var z float64; return z / z }

	for step := opts.StartStep; step < opts.Steps; step++ {
		opt.Config.LR = lr(step)

		x, y := trainSet.Batch(rng, opts.BatchSize, ctx)
		g.ZeroGrads()
		// Dropout only for the training forward/backward; validation and
		// any sampling in onStep see the deterministic model.
		g.SetTraining(true)
		loss, dLogits := CrossEntropy(g.Forward(x), flatten(y))
		g.Backward(dLogits)
		g.SetTraining(false)
		norm := ClipGradNorm(g.Params(), opts.GradClip)
		opt.Step()

		m := Metrics{Step: step + 1, Loss: loss, ValLoss: nan(), LR: opt.Config.LR, GradNorm: norm}
		last := step+1 == opts.Steps
		if val != nil && opts.EvalEvery > 0 && ((step+1)%opts.EvalEvery == 0 || last) {
			m.ValLoss = Evaluate(g, val, opts.EvalBatches, opts.BatchSize, ctx, rng)
		}
		if opts.CheckpointPath != "" && opts.CheckpointEvery > 0 && ((step+1)%opts.CheckpointEvery == 0 || last) {
			if err := Save(opts.CheckpointPath, Snapshot(g, opt, step+1, opts.Tokenizer)); err != nil {
				return err
			}
		}
		if onStep != nil {
			onStep(m)
		}
	}
	return nil
}

// Evaluate returns the mean loss over batches batches drawn from set.
func Evaluate(g *model.GPT, set Batcher, batches, batchSize, contextLen int, rng *rand.Rand) float64 {
	if batches <= 0 {
		batches = 1
	}
	var total float64
	for i := 0; i < batches; i++ {
		x, y := set.Batch(rng, batchSize, contextLen)
		loss, _ := CrossEntropy(g.Forward(x), flatten(y))
		total += loss
	}
	return total / float64(batches)
}

func flatten(rows [][]int) []int {
	out := make([]int, 0, len(rows)*len(rows[0]))
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

func newRand(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }
