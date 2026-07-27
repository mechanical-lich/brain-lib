package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewLobe(t *testing.T) {
	l := NewLobe("foo", map[string]*Modulator{Dopamine: {Name: Dopamine, Level: 0.5, DecayRate: 0.1}})
	assert.NotNil(t, l)
	assert.Equal(t, "foo", l.Name)
	assert.NotEmpty(t, l.Modulators)
}

func TestUpdateNeuronOrder(t *testing.T) {
	l := NewLobe("test", nil)

	// Create neurons A, B, C, D
	a := &Neuron{ID: "A"}
	b := &Neuron{ID: "B"}
	c := &Neuron{ID: "C"}
	d := &Neuron{ID: "D"}

	l.AddNeuron(a)
	l.AddNeuron(b)
	l.AddNeuron(c)
	l.AddNeuron(d)

	// Connect A->B, A->C, B->D, C->D
	l.Connect("A", "B", 1.0, false)
	l.Connect("A", "C", 1.0, false)
	l.Connect("B", "D", 1.0, false)
	l.Connect("C", "D", 1.0, false)

	// After connections, updateOrder should be set
	// Ensure that A comes before B and C, and B and C come before D
	order := l.updateOrder
	assert.NotEmpty(t, order)

	pos := func(id string) int {
		for i, v := range order {
			if v == id {
				return i
			}
		}
		return -1
	}

	assert.True(t, pos("A") >= 0)
	assert.True(t, pos("B") >= 0)
	assert.True(t, pos("C") >= 0)
	assert.True(t, pos("D") >= 0)

	assert.True(t, pos("A") < pos("B"), "A should come before B")
	assert.True(t, pos("A") < pos("C"), "A should come before C")
	assert.True(t, pos("B") < pos("D"), "B should come before D")
	assert.True(t, pos("C") < pos("D"), "C should come before D")
}

func TestStepPropagation(t *testing.T) {
	l := NewLobe("prop", nil)

	// Create neurons A, B, C. A will be driven by bias so it acts as input.
	a := &Neuron{ID: "A", Bias: 1.0, Activation: ActLinear}
	b := &Neuron{ID: "B", Bias: 0.0, Activation: ActLinear}
	c := &Neuron{ID: "C", Bias: 0.0, Activation: ActLinear}

	l.AddNeuron(a)
	l.AddNeuron(b)
	l.AddNeuron(c)

	// Connect A->B (0.5), B->C (0.5)
	l.Connect("A", "B", 0.5, false)
	l.Connect("B", "C", 0.5, false)

	// Run one step
	l.Step(0.1, 0)

	// After step: A should be 1.0 (from bias), B should be 0.5, C should be 0.25
	assert.InDelta(t, 1.0, l.GetNeuron("A").ActivationVal, 1e-6)
	assert.InDelta(t, 0.5, l.GetNeuron("B").ActivationVal, 1e-6)
	assert.InDelta(t, 0.25, l.GetNeuron("C").ActivationVal, 1e-6)
}

func TestPlasticityHebbian(t *testing.T) {
	l := NewLobe("plastic", nil)

	// Create neurons pre and post
	pre := &Neuron{ID: "pre", Bias: 1.0, Activation: ActLinear}
	post := &Neuron{ID: "post", Bias: 0.0, Activation: ActLinear}
	l.AddNeuron(pre)
	l.AddNeuron(post)

	// Connect pre->post with small initial weight and plastic enabled
	syn := l.Connect("pre", "post", 0.1, true)

	// Run one step to build eligibility trace
	l.Step(0.1, 0)

	// Apply reward and run another step
	l.Reinforce(Dopamine, 1.0)
	l.Step(0.1, 1)

	after := l.GetSynapse(syn.ID).Weight
	assert.Greater(t, after, 0.1, "expected synapse weight to increase due to reinforcement plasticity")
}

func TestPlasticityPunishment(t *testing.T) {
	l := NewLobe("plastic", nil)

	// Create neurons pre and post
	pre := &Neuron{ID: "pre", Bias: 1.0, Activation: ActLinear}
	post := &Neuron{ID: "post", Bias: 0.0, Activation: ActLinear}
	l.AddNeuron(pre)
	l.AddNeuron(post)

	// Connect pre->post with initial weight
	syn := l.Connect("pre", "post", 0.5, true)

	// Run one step to build eligibility trace
	l.Step(0.1, 0)

	// Apply punishment (serotonin) and run another step
	l.Reinforce(Serotonin, 1.0)
	l.Step(0.1, 1)

	after := l.GetSynapse(syn.ID).Weight
	assert.Less(t, after, 0.5, "expected synapse weight to decrease due to punishment")
}

func TestPlasticityConfig(t *testing.T) {
	l := NewLobe("plastic", nil)

	// Check default config is set. These are per-second rates.
	assert.Equal(t, DefaultPlasticityConfig(), l.PlasticityConfig)
	assert.Greater(t, l.PlasticityConfig.TraceDecay, 0.0, "a zero trace decay disables the fallback check in applyInterLobePlasticity")

	// Test custom config
	l.PlasticityConfig.RewardLearningRate = 0.5
	assert.Equal(t, 0.5, l.PlasticityConfig.RewardLearningRate)
}
