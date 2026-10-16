package train

import "github.com/mechanical-lich/brain-lib/pkg/gpt/model"

// Stepper is one backend's implementation of a training step. RunStepper
// drives it; the CPU implementation wraps a model.GPT and an AdamW, and
// pkg/gpt/gpumodel provides one that runs the step on a GPU.
type Stepper interface {
	// TrainStep runs one optimizer update on batch x with targets y at
	// learning rate lr. step is the 1-based global update number (used
	// for AdamW's bias correction and dropout masks where the backend
	// needs it). It returns the mean loss before the update and the global
	// gradient norm before clipping.
	TrainStep(x, y [][]int, lr float64, step int) (loss, gradNorm float64, err error)
	// Eval returns the mean loss on x, y without dropout and without
	// changing any state.
	Eval(x, y [][]int) (float64, error)
	// Sync makes the CPU model and optimizer the stepper was created over
	// reflect its current state, so checkpoints and sampling see the
	// trained weights. step is the number of updates applied so far. It is
	// called before every validation and checkpoint, and is a no-op for
	// the CPU stepper.
	Sync(step int) error
}

// cpuStepper is the float64 CPU training step: the loop body Run has
// always executed.
type cpuStepper struct {
	g    *model.GPT
	opt  *AdamW
	clip float64
}

// NewCPUStepper wraps g and opt as a Stepper, clipping the global gradient
// norm to clip (0 disables).
func NewCPUStepper(g *model.GPT, opt *AdamW, clip float64) Stepper {
	return &cpuStepper{g: g, opt: opt, clip: clip}
}

func (s *cpuStepper) TrainStep(x, y [][]int, lr float64, _ int) (float64, float64, error) {
	s.opt.Config.LR = lr
	s.g.ZeroGrads()
	// Dropout only for the training forward/backward; validation and any
	// sampling in onStep see the deterministic model.
	s.g.SetTraining(true)
	loss, dLogits := CrossEntropy(s.g.Forward(x), flatten(y))
	s.g.Backward(dLogits)
	s.g.SetTraining(false)
	norm := ClipGradNorm(s.g.Params(), s.clip)
	s.opt.Step()
	return loss, norm, nil
}

func (s *cpuStepper) Eval(x, y [][]int) (float64, error) {
	loss, _ := CrossEntropy(s.g.Forward(x), flatten(y))
	return loss, nil
}

func (s *cpuStepper) Sync(int) error { return nil }
