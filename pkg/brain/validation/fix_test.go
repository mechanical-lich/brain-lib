package validation

import (
	"testing"

	"github.com/mechanical-lich/brain-lib/pkg/brain"
	"github.com/stretchr/testify/assert"
)

func TestFix_MoveIntraFromBrainToLobe(t *testing.T) {
	br := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	a := &brain.Neuron{ID: "a", Bias: 0.0, Activation: brain.ActLinear}
	b := &brain.Neuron{ID: "b", Bias: 0.0, Activation: brain.ActLinear}
	l.AddNeuron(a)
	l.AddNeuron(b)
	br.AddLobe(l)

	syn := &brain.Synapse{ID: "s", FromLobe: "L", From: "a", ToLobe: "L", To: "b", Weight: 0.1}
	br.Synapses = map[string]*brain.Synapse{"s": syn}

	complaints := []Complaint{{ComplaintType: Inter_Synapse_To_Same_Lobe, ID: "s"}}

	err := FixSynapses(br, complaints)
	assert.NoError(t, err)
	_, inBrain := br.Synapses["s"]
	assert.False(t, inBrain)
	_, inLobe := l.Synapses["s"]
	assert.True(t, inLobe)
}

func TestFix_DeleteMissingNeuron(t *testing.T) {
	br := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	// only add 'to' neuron
	to := &brain.Neuron{ID: "to", Bias: 0.0, Activation: brain.ActLinear}
	l.AddNeuron(to)
	br.AddLobe(l)

	syn := &brain.Synapse{ID: "bad", FromLobe: "L", From: "missing", ToLobe: "L", To: "to", Weight: 0.2}
	br.Synapses = map[string]*brain.Synapse{"bad": syn}

	complaints := []Complaint{{ComplaintType: Missing_Neuron_From, ID: "bad"}}

	err := FixSynapses(br, complaints)
	assert.NoError(t, err)
	_, exists := br.Synapses["bad"]
	assert.False(t, exists)
}

func TestFix_MoveLobeToBrain(t *testing.T) {
	br := brain.NewBrain()
	l1 := brain.NewLobe("L1", nil)
	l2 := brain.NewLobe("L2", nil)
	a := &brain.Neuron{ID: "a", Bias: 0.0, Activation: brain.ActLinear}
	b := &brain.Neuron{ID: "b", Bias: 0.0, Activation: brain.ActLinear}
	l1.AddNeuron(a)
	l2.AddNeuron(b)
	br.AddLobe(l1)
	br.AddLobe(l2)

	// store inter-lobe synapse incorrectly in lobe map
	l1.Synapses = map[string]*brain.Synapse{"s2": {ID: "s2", FromLobe: "L1", From: "a", ToLobe: "L2", To: "b", Weight: 0.3}}

	complaints := []Complaint{{ComplaintType: Intra_Synapse_To_Different_Lobe, ID: "s2"}}

	err := FixSynapses(br, complaints)
	assert.NoError(t, err)
	_, inLobe := l1.Synapses["s2"]
	assert.False(t, inLobe)
	_, inBrain := br.Synapses["s2"]
	assert.True(t, inBrain)
}

func TestFix_InferToLobe(t *testing.T) {
	br := brain.NewBrain()
	l1 := brain.NewLobe("L1", nil)
	l2 := brain.NewLobe("L2", nil)
	a := &brain.Neuron{ID: "a", Bias: 0.0, Activation: brain.ActLinear}
	b := &brain.Neuron{ID: "b", Bias: 0.0, Activation: brain.ActLinear}
	l1.AddNeuron(a)
	l2.AddNeuron(b)
	br.AddLobe(l1)
	br.AddLobe(l2)

	// inter-lobe synapse in lobe map missing ToLobe
	l1.Synapses = map[string]*brain.Synapse{"s3": {ID: "s3", From: "a", To: "b", FromLobe: "L1", ToLobe: "", Weight: 0.4}}

	complaints := []Complaint{{ComplaintType: Intra_Synapse_To_Different_Lobe, ID: "s3"}}

	err := FixSynapses(br, complaints)
	assert.NoError(t, err)
	s, ok := br.Synapses["s3"]
	assert.True(t, ok)
	assert.Equal(t, "L2", s.ToLobe)
}

func TestFix_DeleteWhenCannotInferToLobe(t *testing.T) {
	br := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	a := &brain.Neuron{ID: "a", Bias: 0.0, Activation: brain.ActLinear}
	l.AddNeuron(a)
	br.AddLobe(l)

	// lobe-level synapse missing ToLobe and target neuron doesn't exist anywhere
	l.Synapses = map[string]*brain.Synapse{"s_del": {ID: "s_del", From: "a", To: "missing", FromLobe: "L", ToLobe: "", Weight: 0.5}}

	complaints := []Complaint{{ComplaintType: Intra_Synapse_To_Different_Lobe, ID: "s_del"}}

	err := FixSynapses(br, complaints)
	assert.NoError(t, err)

	// Synapse should be deleted (not present in lobe or brain)
	_, inBrain := br.Synapses["s_del"]
	assert.False(t, inBrain)
	_, inLobe := l.Synapses["s_del"]
	assert.False(t, inLobe)
}
