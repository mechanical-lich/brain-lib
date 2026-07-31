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
	// LearningRate is serialized but never read by this library. Plasticity is
	// driven entirely by the lobe's PlasticityConfig; per-neuron rates were
	// never implemented. Kept so existing files round-trip without losing data.
	LearningRate float64 `json:"learning_rate"`
	DecayRate    float64 `json:"decay_rate"`
	Max          float64 `json:"max_value"` // Optional max value clamp
	Min          float64 `json:"min_value"` // Optional min value clamp
	Threshold    float64 `json:"threshold"` // Optional threshold value, activationVal must be this high otherwise it is 0.0
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

// EffectiveDecayRate resolves this neuron's decay rate against its lobe's.
//
// A lobe's DecayRate is a *default* for the neurons that do not set one of their
// own, not an extra decay applied on top. It used to be both: Compute applied
// the neuron's rate and Lobe.Step then multiplied by the lobe's, so the two
// stacked and a neuron could only ever decay faster than its lobe, never slower.
// Setting `decay_rate: 0.005` on a neuron in a lobe running 0.05 bought a 10%
// change instead of the 10x it reads as, which made per-neuron rates all but
// useless — and made a long-lived memory impossible to express in a lobe of
// short-lived ones.
//
// Zero means "unset". A neuron that genuinely should never decay needs a lobe
// that does not either; the validator reports that pairing as
// Frozen_Persisting_Neuron.
func (n *Neuron) EffectiveDecayRate(lobeRate float64) float64 {
	if n.DecayRate > 0 {
		return n.DecayRate
	}
	return lobeRate
}

// Compute updates the neuron's activation from its summed synaptic input.
//
// persists says whether this neuron retains state between steps, which is the
// lobe's Persists OR'd with the neuron's own. decayRate is the resolved rate
// from EffectiveDecayRate; a caller holding a lobe should pass that rather than
// the neuron's raw DecayRate.
//
// The activation used to only ever ratchet *upward* within a step
// (`if n.ActivationVal < out`), then decay. That had three consequences:
// a persisting neuron driven high never followed its input back down (measured:
// driven to 1.98, input dropped to 0.5, activation stayed 1.97); negative
// activations were unreachable, so Neuron.Min was dead and inhibitory synapses
// could only cancel excitation rather than invert it (measured: an input of -5
// against Min=-2 produced 0.0); and Threshold skipped the update instead of
// suppressing the signal, contradicting its own documentation.
func (n *Neuron) Compute(sum float64, dt float64, persists bool, decayRate float64) float64 {
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
		n.ActivationVal *= decayFactor(decayRate, dt)

	case persists:
		// Asymmetric: a stronger signal is taken immediately, a weaker one is
		// relaxed toward with DecayRate as the time constant.
		//
		// Symmetric relaxation was measurably wrong for the signals these
		// neurons carry. The shipped brain's attention slots feed
		// `proximity -> focus` at weight 0.40, the largest single term in focus
		// selection, and `decay_rate` there is 0.05. With both directions
		// smoothed, a near object appearing beside an established far one took
		// **5.75 simulated seconds** to win focus, and proximity never reached
		// its true value at all (0.233 after 30s of a constant 0.3 input) — so
		// the critter under-reported distance and stayed fixated on whatever it
		// had been looking at.
		//
		// Rising instantly restores that: the near object leads from tick 1.
		// Falling still relaxes, which is the point of the rewrite — the old
		// code could only ratchet upward and then decay toward zero, so it
		// could never follow its input down to a lower non-zero value.
		//
		// Compared by magnitude rather than sign, so a neuron opted into signed
		// activation (Min < 0) responds to a strong inhibitory signal just as
		// promptly as to a strong excitatory one.
		//
		// A resolved rate of 0 therefore means "rise instantly, never fall" —
		// coherent, but a neuron that can only climb. That requires both the
		// neuron and its lobe to be unset; the validator reports it as
		// Frozen_Persisting_Neuron.
		if math.Abs(out) >= math.Abs(n.ActivationVal) {
			n.ActivationVal = out
		} else {
			f := decayFactor(decayRate, dt)
			n.ActivationVal = out + (n.ActivationVal-out)*f
		}

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
