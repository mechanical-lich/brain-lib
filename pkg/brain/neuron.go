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

// Compute updates the neuron's activation from its summed synaptic input.
//
// persists says whether this neuron retains state between steps, which is the
// lobe's Persists OR'd with the neuron's own.
//
// The activation used to only ever ratchet *upward* within a step
// (`if n.ActivationVal < out`), then decay. That had three consequences:
// a persisting neuron driven high never followed its input back down (measured:
// driven to 1.98, input dropped to 0.5, activation stayed 1.97); negative
// activations were unreachable, so Neuron.Min was dead and inhibitory synapses
// could only cancel excitation rather than invert it (measured: an input of -5
// against Min=-2 produced 0.0); and Threshold skipped the update instead of
// suppressing the signal, contradicting its own documentation.
func (n *Neuron) Compute(sum float64, dt float64, persists bool) float64 {
	v := sum + n.Bias + n.ExternalInput
	out := applyActivation(n.Activation, v)
	n.ExternalInput = 0.0 // reset after use

	// Threshold gates on magnitude, so it suppresses weak signals of either
	// sign rather than discarding every negative one.
	if math.Abs(out) < n.Threshold {
		out = 0
	}

	switch {
	case n.Capacitor:
		// Integrate input over time, then leak. Scaled by dt so charge
		// accumulates per second rather than per frame.
		n.ActivationVal += out * dt
		n.ActivationVal *= decayFactor(n.DecayRate, dt)

	case persists:
		// Relax toward the current input, with DecayRate as the time constant.
		// This is what makes a persisting neuron a low-pass filter of its input
		// rather than a max-hold: it rises *and* falls toward what it is being
		// told, just slowly.
		f := decayFactor(n.DecayRate, dt)
		n.ActivationVal = out + (n.ActivationVal-out)*f

	default:
		// Stateless neuron: activation is purely a function of this step's
		// input.
		n.ActivationVal = out
	}

	if n.Max != 0 && n.ActivationVal > n.Max {
		n.ActivationVal = n.Max
	}

	// Rectify at zero unless a negative Min is explicitly configured.
	//
	// Neurons are used as logic gates throughout the brain — an AND is two
	// weight-1 inputs and a bias of -1, and the lab's concept builder generates
	// exactly that shape — so a sub-threshold sum *must* rectify to 0 or every
	// gate inverts. Setting Min below zero opts an individual neuron into
	// signed activation, which is what makes Min meaningful and lets an
	// inhibitory synapse invert rather than merely cancel.
	floor := 0.0
	if n.Min < 0 {
		floor = n.Min
	}
	if n.ActivationVal < floor {
		n.ActivationVal = floor
	}

	return n.ActivationVal
}
