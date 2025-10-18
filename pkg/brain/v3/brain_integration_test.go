package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Basic brain with a single lobe and simple neuron propagation
func TestBrainSingleLobe(t *testing.T) {
	br := NewBrain()
	l := NewLobe("L1", nil)

	// neurons: input (bias), mid, out
	in := &Neuron{ID: "in", Bias: 1.0, Activation: ActLinear}
	mid := &Neuron{ID: "mid", Bias: 0.0, Activation: ActLinear}
	out := &Neuron{ID: "out", Bias: 0.0, Activation: ActLinear}

	l.AddNeuron(in)
	l.AddNeuron(mid)
	l.AddNeuron(out)

	l.Connect("in", "mid", 0.5, false)
	l.Connect("mid", "out", 0.5, false)

	br.AddLobe(l)

	// step the brain
	br.Step(0.1)

	assert.InDelta(t, 1.0, br.GetLobe("L1").GetNeuron("in").ActivationVal, 1e-6)
	assert.InDelta(t, 0.5, br.GetLobe("L1").GetNeuron("mid").ActivationVal, 1e-6)
	assert.InDelta(t, 0.25, br.GetLobe("L1").GetNeuron("out").ActivationVal, 1e-6)
}

// Two lobes, feed-forward from L1 to L2 via cross-lobe synapse
func TestBrainTwoLobeFeedForward(t *testing.T) {
	br := NewBrain()

	l1 := NewLobe("L1", nil)
	l2 := NewLobe("L2", nil)

	in := &Neuron{ID: "in", Bias: 1.0, Activation: ActLinear}
	mid := &Neuron{ID: "mid", Bias: 0.0, Activation: ActLinear}
	out := &Neuron{ID: "out", Bias: 0.0, Activation: ActLinear}

	l1.AddNeuron(in)
	l2.AddNeuron(mid)
	l2.AddNeuron(out)

	// Connect across lobes: L1.in -> L2.mid, L2.mid -> L2.out
	br.AddLobe(l1)
	br.AddLobe(l2)
	br.Connect("L1", "in", "L2", "mid", 0.5, false)
	// intra-lobe connection should be added directly to the lobe (no Brain mirroring)
	l2.Connect("mid", "out", 0.5, false)

	// step should run lobes in topological order and propagate activation
	br.Step(0.1)

	// After step: mid should be 0.5, out should be 0.25
	assert.InDelta(t, 1.0, br.GetLobe("L1").GetNeuron("in").ActivationVal, 1e-6)
	assert.InDelta(t, 0.5, br.GetLobe("L2").GetNeuron("mid").ActivationVal, 1e-6)
	assert.InDelta(t, 0.25, br.GetLobe("L2").GetNeuron("out").ActivationVal, 1e-6)
}
