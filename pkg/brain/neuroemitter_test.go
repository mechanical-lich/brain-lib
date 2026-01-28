package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNeuroemitter_BelowThreshold(t *testing.T) {
	emitter := NewNeuroemitter("e1", "dopamine", 0.5, 1.0)

	// Send input below threshold
	emitter.ReceiveInput(0.5)

	// Step should produce no emission
	emission := emitter.Step(0.1)
	assert.Equal(t, 0.0, emission, "Should not emit below threshold")
}

func TestNeuroemitter_AboveThreshold(t *testing.T) {
	emitter := NewNeuroemitter("e1", "dopamine", 0.5, 1.0)

	// Send input above threshold
	emitter.ReceiveInput(2.0)

	// Step should produce emission = activation * rate * dt = 2.0 * 0.5 * 0.1 = 0.1
	emission := emitter.Step(0.1)
	assert.Equal(t, 0.1, emission, "Should emit when above threshold")
}

func TestNeuroemitter_AtThreshold(t *testing.T) {
	emitter := NewNeuroemitter("e1", "serotonin", 0.3, 1.0)

	// Send input exactly at threshold
	emitter.ReceiveInput(1.0)

	// Step should produce emission (threshold is inclusive)
	emission := emitter.Step(0.1)
	assert.Equal(t, 0.03, emission, "Should emit when at threshold")
}

func TestNeuroemitter_AccumulatesInput(t *testing.T) {
	emitter := NewNeuroemitter("e1", "dopamine", 0.5, 1.0)

	// Send multiple inputs that accumulate
	emitter.ReceiveInput(0.3)
	emitter.ReceiveInput(0.4)
	emitter.ReceiveInput(0.5)

	// Total = 1.2, above threshold of 1.0
	emission := emitter.Step(0.1)
	assert.Equal(t, 0.06, emission, "Should accumulate all inputs")
}

func TestNeuroemitter_ResetsAfterStep(t *testing.T) {
	emitter := NewNeuroemitter("e1", "dopamine", 0.5, 1.0)

	// First step with input above threshold
	emitter.ReceiveInput(2.0)
	emission1 := emitter.Step(0.1)
	assert.Equal(t, 0.1, emission1, "First emission should be non-zero")

	// Second step without new input should emit nothing
	emission2 := emitter.Step(0.1)
	assert.Equal(t, 0.0, emission2, "Should reset after step")
}

func TestNeuroemitter_InLobe(t *testing.T) {
	lobe := NewLobe("test", nil)
	lobe.Persists = true // Make sure neurons keep their activation values

	// Add neurons
	n1 := &Neuron{ID: "n1", ActivationVal: 0.8, Bias: 0.0, Threshold: 0.0}
	n2 := &Neuron{ID: "n2", ActivationVal: 0.6, Bias: 0.0, Threshold: 0.0}
	lobe.AddNeuron(n1)
	lobe.AddNeuron(n2)

	// Add neuroemitter with lower threshold
	lobe.AddEmitter("e1", "dopamine", 0.5, 0.5) // threshold 0.5 instead of 1.0

	// Connect neurons to emitter via synapses
	syn1 := &Synapse{ID: "n1_to_e1", From: "n1", To: "e1", FromLobe: "test", ToLobe: "test", Weight: 0.5}
	syn2 := &Synapse{ID: "n2_to_e1", From: "n2", To: "e1", FromLobe: "test", ToLobe: "test", Weight: 0.5}
	lobe.AddSynapse(syn1)
	lobe.AddSynapse(syn2)

	// Step the lobe (this should send neuron activations to emitter)
	lobe.Step(0.1, 0)

	// Update neuroemitters and check chemical production
	chemicals := make(map[string]float64)
	chemicals = lobe.UpdateNeuroemitters(0.1, chemicals)

	// Expected: (0.8 * 0.5 + 0.6 * 0.5) * 0.5 * 0.1 = 0.7 * 0.5 * 0.1 = 0.035
	assert.InDelta(t, 0.035, chemicals["dopamine"], 0.001, "Should produce correct dopamine amount")
}

func TestNeuroemitter_MultipleEmittersToSameChemical(t *testing.T) {
	lobe := NewLobe("test", nil)
	lobe.Persists = true // Make sure neurons keep their activation values

	// Add neurons
	n1 := &Neuron{ID: "n1", ActivationVal: 1.0, Bias: 0.0, Threshold: 0.0}
	n2 := &Neuron{ID: "n2", ActivationVal: 1.0, Bias: 0.0, Threshold: 0.0}
	lobe.AddNeuron(n1)
	lobe.AddNeuron(n2)

	// Add two emitters for the same chemical
	lobe.AddEmitter("e1", "dopamine", 0.5, 0.5)
	lobe.AddEmitter("e2", "dopamine", 0.3, 0.5)

	// Connect each neuron to a different emitter
	syn1 := &Synapse{ID: "n1_to_e1", From: "n1", To: "e1", FromLobe: "test", ToLobe: "test", Weight: 1.0}
	syn2 := &Synapse{ID: "n2_to_e2", From: "n2", To: "e2", FromLobe: "test", ToLobe: "test", Weight: 1.0}
	lobe.AddSynapse(syn1)
	lobe.AddSynapse(syn2)

	// Step the lobe
	lobe.Step(0.1, 0)

	// Update neuroemitters
	chemicals := make(map[string]float64)
	chemicals = lobe.UpdateNeuroemitters(0.1, chemicals)

	// Expected: e1 emits 1.0 * 0.5 * 0.1 = 0.05, e2 emits 1.0 * 0.3 * 0.1 = 0.03
	// Total = 0.08
	assert.InDelta(t, 0.08, chemicals["dopamine"], 0.001, "Multiple emitters should accumulate")
}

func TestNeuroemitter_RemoveEmitter(t *testing.T) {
	lobe := NewLobe("test", nil)

	// Add emitter
	lobe.AddEmitter("e1", "dopamine", 0.5, 1.0)
	assert.NotNil(t, lobe.GetEmitter("e1"), "Emitter should exist")

	// Add synapse to emitter
	syn := &Synapse{ID: "test_syn", From: "n1", To: "e1", FromLobe: "test", ToLobe: "test"}
	lobe.AddSynapse(syn)

	// Remove emitter
	lobe.RemoveEmitter("e1")
	assert.Nil(t, lobe.GetEmitter("e1"), "Emitter should be removed")
	assert.Nil(t, lobe.GetSynapse("test_syn"), "Synapse to emitter should be removed")
}
