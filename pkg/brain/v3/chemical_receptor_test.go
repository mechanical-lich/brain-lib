package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestApplyChemical_SingleReceptor(t *testing.T) {
	l := NewLobe("test", nil)

	n := &Neuron{ID: "n1"}
	l.AddNeuron(n)
	l.AddReceptor("chemA", 0.2, "n1", TargetTypeNeuron, ReceptorTypeExcitatory)

	// Apply 1.0 concentration
	rem := l.ApplyChemical("chemA", 1.0)

	// neuron should have received concentration * uptake
	assert.InDelta(t, 0.2, l.GetNeuron("n1").ExternalInput, 1e-9)
	// returned remaining concentration should be 0.8
	assert.InDelta(t, 0.8, rem, 1e-9)
}

func TestApplyChemical_InhibitoryReceptor(t *testing.T) {
	l := NewLobe("inhib", nil)
	n := &Neuron{ID: "n2", ExternalInput: 0.5}
	l.AddNeuron(n)

	// inhibitory receptor should subtract from ExternalInput
	l.AddReceptor("chemB", 0.2, "n2", TargetTypeNeuron, ReceptorTypeInhibitory)

	rem := l.ApplyChemical("chemB", 1.0)
	// amount = 0.2 taken -> ExternalInput should decrease from 0.5 to 0.3
	assert.InDelta(t, 0.3, l.GetNeuron("n2").ExternalInput, 1e-9)
	assert.InDelta(t, 0.8, rem, 1e-9)
}

func TestUpdateChemicalReceptors_and_BrainIntegration(t *testing.T) {
	// Lobe with one receptor targeting an existing neuron
	l := NewLobe("chemLobe", nil)
	n := &Neuron{ID: "target"}
	l.AddNeuron(n)
	l.AddReceptor("food", 0.5, "target", TargetTypeNeuron, ReceptorTypeExcitatory)

	chemicals := map[string]float64{"food": 2.0}

	// dt = 0.1 -> amount = 2.0 * 0.5 * 0.1 = 0.1
	updated := l.UpdateChemicalReceptors(0.1, chemicals)

	assert.InDelta(t, 0.1, l.GetNeuron("target").ExternalInput, 1e-9)
	assert.InDelta(t, 1.9, updated["food"], 1e-9)

	// If the receptor points at a missing neuron, chemicals should be unchanged for that receptor
	l2 := NewLobe("missing", nil)
	l2.AddReceptor("x", 1.0, "nope", TargetTypeNeuron, ReceptorTypeExcitatory)
	ch2 := map[string]float64{"x": 1.0}
	after := l2.UpdateChemicalReceptors(1.0, ch2)
	assert.InDelta(t, 1.0, after["x"], 1e-9)

	// Brain-level integration: Brain.ApplyChemistry should dispatch to lobes
	b := NewBrain()
	b.AddLobe(l)
	// Provide a fresh chemicals map
	chem := map[string]float64{"food": 1.0}
	b.ApplyChemistry(chem, 0.5) // amount = 1.0 * 0.5 * 0.5 = 0.25
	assert.InDelta(t, 0.75, chem["food"], 1e-9)
	// target neuron on l should have received the input
	assert.Greater(t, l.GetNeuron("target").ExternalInput, 0.0)

	// Test modulator-targeting receptor
	lm := NewLobe("modLobe", map[string]*Modulator{"m": {Name: "m", Level: 0.1}})
	lm.AddReceptor("sugar", 1.0, "m", TargetTypeModulator, ReceptorTypeExcitatory)
	chm := map[string]float64{"sugar": 1.0}
	updatedMod := lm.UpdateChemicalReceptors(0.5, chm) // amount = 1.0 * 1.0 * 0.5 = 0.5
	assert.InDelta(t, 0.6, lm.Modulators["m"].Level, 1e-9)
	assert.InDelta(t, 0.5, updatedMod["sugar"], 1e-9)
}
