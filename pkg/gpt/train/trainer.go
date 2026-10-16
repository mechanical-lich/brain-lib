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

// Run trains g with opt on the CPU for opts.Steps steps, calling onStep
// after each. val may be nil to disable validation. The caller owns rng so
// runs are reproducible. g is in training mode (dropout on) only for each
// step's forward/backward and is left in inference mode. It is RunStepper
// over NewCPUStepper(g, opt, opts.GradClip).
func Run(g *model.GPT, opt *AdamW, trainSet, val Batcher, opts Options, rng *rand.Rand, onStep func(Metrics)) error {
	return RunStepper(NewCPUStepper(g, opt, opts.GradClip), g, opt, trainSet, val, opts, rng, onStep)
}

// RunStepper is the training loop over any Stepper: the learning-rate
// schedule, validation every EvalEvery steps, checkpoints every
// CheckpointEvery steps and at the end, and the onStep callback. g and opt
// are the CPU model and optimizer the checkpoints are written from; the
// stepper's Sync is called before each validation and checkpoint so they
// are current. opts.GradClip is the stepper's business and is ignored here.
func RunStepper(s Stepper, g *model.GPT, opt *AdamW, trainSet, val Batcher, opts Options, rng *rand.Rand, onStep func(Metrics)) error {
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
		rate := lr(step)
		x, y := trainSet.Batch(rng, opts.BatchSize, ctx)
		loss, norm, err := s.TrainStep(x, y, rate, step+1)
		if err != nil {
			return fmt.Errorf("train: step %d: %w", step+1, err)
		}

		m := Metrics{Step: step + 1, Loss: loss, ValLoss: nan(), LR: rate, GradNorm: norm}
		last := step+1 == opts.Steps
		doEval := val != nil && opts.EvalEvery > 0 && ((step+1)%opts.EvalEvery == 0 || last)
		doSave := opts.CheckpointPath != "" && opts.CheckpointEvery > 0 && ((step+1)%opts.CheckpointEvery == 0 || last)
		if doEval || doSave {
			if err := s.Sync(step + 1); err != nil {
				return fmt.Errorf("train: sync at step %d: %w", step+1, err)
			}
		}
		if doEval {
			if m.ValLoss, err = evaluate(s, val, opts.EvalBatches, opts.BatchSize, ctx, rng); err != nil {
				return fmt.Errorf("train: eval at step %d: %w", step+1, err)
			}
		}
		if doSave {
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

// evaluate averages s.Eval over batches batches from set.
func evaluate(s Stepper, set Batcher, batches, batchSize, contextLen int, rng *rand.Rand) (float64, error) {
	if batches <= 0 {
		batches = 1
	}
	var total float64
	for i := 0; i < batches; i++ {
		x, y := set.Batch(rng, batchSize, contextLen)
		loss, err := s.Eval(x, y)
		if err != nil {
			return 0, err
		}
		total += loss
	}
	return total / float64(batches), nil
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
