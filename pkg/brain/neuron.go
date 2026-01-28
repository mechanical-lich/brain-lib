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
	Max           float64        `json:"max_value"` // Optional max value clamp
	Min           float64        `json:"min_value"` // Optional min value clamp
	Threshold     float64        `json:"threshold"` // Optional threshold value, activationVal must be this high otherwise it is 0.0
	// Whether this neuron persists its activation between steps (overrides lobe-level persists if needed)
	Persists bool `json:"persists,omitempty"`
	// Capacitor mode: when true, input is ADDED to activation rather than replacing it (only if above threshold).
	// This allows the neuron to accumulate charge over time, like a capacitor.
	Capacitor bool `json:"capacitor,omitempty"`
	// Optional visual position for editor/UI
	X      float64 `json:"x,omitempty"`
	Y      float64 `json:"y,omitempty"`
	Hidden bool    `json:"hidden,omitempty"`

	// GroupID is an integer group identifier (0 = no-group). Used to constrain rewiring targets.
	GroupID int `json:"group_id,omitempty"`
}

func (n *Neuron) Compute(sum float64, dt float64) float64 {
	v := sum + n.Bias + n.ExternalInput
	out := applyActivation(n.Activation, v)

	if n.Capacitor {
		// Capacitor mode: add input to activation if above threshold
		if out >= n.Threshold {
			n.ActivationVal += out
		}
	} else {
		// Standard mode: only update if new value is higher
		if n.ActivationVal < out && out >= n.Threshold {
			n.ActivationVal = out
		}
	}

	// Scale decay by time delta so it's frame-rate independent
	n.ActivationVal -= n.ActivationVal * n.DecayRate * dt

	n.ExternalInput = 0.0 // reset after use

	if n.Max != 0 && n.ActivationVal > n.Max {
		n.ActivationVal = n.Max
	}
	if n.Min != 0 && n.ActivationVal < n.Min {
		n.ActivationVal = n.Min
	}

	return n.ActivationVal
}
