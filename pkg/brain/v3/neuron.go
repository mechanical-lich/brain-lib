package brain

import (
	"math"
)

// ActivationFunc is a simple enum for neuron activation types
type ActivationFunc int

const (
	ActLinear ActivationFunc = iota
	ActSigmoid
	ActTanh
)

func applyActivation(a ActivationFunc, x float64) float64 {
	switch a {
	case ActSigmoid:
		return 1.0 / (1.0 + math.Exp(-x))
	case ActTanh:
		return math.Tanh(x)
	default:
		return x
	}
}

// Neuron represents a compact unit that accumulates inputs and produces an output.
type Neuron struct {
	ID            string         `json:"id"`
	Bias          float64        `json:"bias"`
	Activation    ActivationFunc `json:"activation"`
	ActivationVal float64        `json:"activation_val"`
	ExternalInput float64        `json:"external_lobe_input"`
	LearningRate  float64        `json:"learning_rate"`
	DecayRate     float64        `json:"decay_rate"`
	Threshold     float64        `json:"threshold"` // Optional threshold value, activationVal must be this high otherwise it is 0.0
	// Whether this neuron persists its activation between steps (overrides lobe-level persists if needed)
	Persists bool `json:"persists,omitempty"`
	// Optional visual position for editor/UI
	X float64 `json:"x,omitempty"`
	Y float64 `json:"y,omitempty"`
}

func (n *Neuron) Compute(sum float64) float64 {
	v := sum + n.Bias + n.ExternalInput
	out := applyActivation(n.Activation, v)

	if n.ActivationVal < out && out > n.Threshold {
		n.ActivationVal = out
	}

	n.ActivationVal -= n.ActivationVal * n.DecayRate

	n.ExternalInput = 0.0 // reset after use
	return n.ActivationVal
}
