package pnn

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
	Incoming      []string       `json:"incoming"` // synapse IDs
	Outgoing      []string       `json:"outgoing"` // synapse IDs (optional)
	LearningRate  float64        `json:"learning_rate"`
}

func (n *Neuron) Compute(sum float64) float64 {
	v := sum + n.Bias
	out := applyActivation(n.Activation, v)
	n.ActivationVal = out
	return out
}
