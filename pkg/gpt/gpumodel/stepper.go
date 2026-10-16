package gpumodel

import (
	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

// Stepper runs a Model as a train.Stepper, so train.RunStepper drives the
// GPU with the same schedule, validation, checkpoint and logging code as
// the CPU. Checkpoints are still written from the CPU model and optimizer
// it was created over: Sync downloads the device state into them first.
type Stepper struct {
	Model *Model
	g     *model.GPT
	opt   *train.AdamW
	clip  float64
}

// NewStepper uploads g's weights and opt's moments into m, turns training
// mode on, and returns the stepper. opt must have been created over
// g.Params(); its Config supplies the AdamW hyperparameters (the learning
// rate is overridden per step by the schedule).
func NewStepper(m *Model, g *model.GPT, opt *train.AdamW, clip float64) (*Stepper, error) {
	if err := m.LoadFrom(g); err != nil {
		return nil, err
	}
	m.LoadOptimizer(opt)
	m.SetTraining(true)
	return &Stepper{Model: m, g: g, opt: opt, clip: clip}, nil
}

// TrainStep implements train.Stepper.
func (s *Stepper) TrainStep(x, y [][]int, lr float64, step int) (float64, float64, error) {
	cfg := s.opt.Config
	cfg.LR = lr
	r, err := s.Model.TrainStep(x, y, cfg, s.clip, step)
	if err != nil {
		return 0, 0, err
	}
	return r.Loss, r.GradNorm, nil
}

// Eval implements train.Stepper.
func (s *Stepper) Eval(x, y [][]int) (float64, error) { return s.Model.Eval(x, y) }

// Sync implements train.Stepper: it downloads the weights and optimizer
// moments into the CPU model and optimizer, and sets the optimizer's update
// count to step, so train.Snapshot and model.Generate see the trained
// state.
func (s *Stepper) Sync(step int) error { return s.Model.StoreTo(s.g, s.opt, step) }
