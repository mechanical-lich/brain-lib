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
	rem := l.ApplyChemical("chemA", 1.0, 1.0)

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

	rem := l.ApplyChemical("chemB", 1.0, 1.0)
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

func TestChemicalReceptor_NormalizationMinMax(t *testing.T) {
	cr := &ChemicalReceptor{
		ChemicalName:           "nutr",
		UptakeRate:             1.0,
		OutputNormalizationMin: 10.0,
		OutputNormalizationMax: 20.0,
		Threshold:              0.0,
		ReceptorType:           ReceptorTypeExcitatory,
	}

	// At min -> output should be 0.0, consumed = conc * uptakeRate * dt
	ch := map[string]float64{"nutr": 10.0}
	out, consumed := cr.GetOutput(ch, 1.0)
	assert.InDelta(t, 0.0, out, 1e-9)
	assert.InDelta(t, 10.0, consumed, 1e-9)

	// Midpoint -> normalized to 0.5
	ch["nutr"] = 15.0
	out, consumed = cr.GetOutput(ch, 1.0)
	assert.InDelta(t, 0.5, out, 1e-9)
	assert.InDelta(t, 15.0, consumed, 1e-9)

	// At max -> normalized to 1.0
	ch["nutr"] = 20.0
	out, consumed = cr.GetOutput(ch, 1.0)
	assert.InDelta(t, 1.0, out, 1e-9)
	assert.InDelta(t, 20.0, consumed, 1e-9)
}

func TestChemicalReceptor_ThresholdBehavior(t *testing.T) {
	cr := &ChemicalReceptor{
		ChemicalName: "fuel",
		UptakeRate:   1.0,
		Threshold:    1.0,
		ReceptorType: ReceptorTypeExcitatory,
	}

	// Below threshold -> no output, no consumption
	ch := map[string]float64{"fuel": 0.5}
	out, consumed := cr.GetOutput(ch, 1.0)
	assert.InDelta(t, 0.0, out, 1e-9)
	assert.InDelta(t, 0.0, consumed, 1e-9)

	// At threshold -> produces output and consumes according to uptake
	ch["fuel"] = 1.0
	out, consumed = cr.GetOutput(ch, 1.0)
	// uptake = concentration * uptakeRate * dt = 1.0
	assert.InDelta(t, 1.0, out, 1e-9)
	assert.InDelta(t, 1.0, consumed, 1e-9)

	// Above threshold with smaller dt
	ch["fuel"] = 2.0
	out, consumed = cr.GetOutput(ch, 0.5)
	// uptake = 2.0 * 1.0 * 0.5 = 1.0
	assert.InDelta(t, 1.0, out, 1e-9)
	assert.InDelta(t, 1.0, consumed, 1e-9)
}

// A normalized receptor must deliver a full 0..1 signal to a neuron.
//
// It used to scale the normalized value by dt for every target. A neuron's
// ExternalInput is consumed and zeroed on each Compute, so that capped the
// signal at dt — 0.1 at the game's step, against drive neurons that all run
// 0..1. A chemical drive could not compete with hunger no matter how strong the
// concentration.
func TestChemicalReceptor_NormalizedNeuronOutputIsALevelNotARate(t *testing.T) {
	cr := &ChemicalReceptor{
		ChemicalName:           "testosterone",
		UptakeRate:             0.01,
		Target:                 "libido",
		TargetType:             TargetTypeNeuron,
		ReceptorType:           ReceptorTypeExcitatory,
		OutputNormalizationMin: 0.0,
		OutputNormalizationMax: 50.0,
	}

	ch := map[string]float64{"testosterone": 50.0}
	out, consumed := cr.GetOutput(ch, 0.1)

	assert.InDelta(t, 1.0, out, 1e-9, "a saturated receptor should deliver a full signal")
	// Draining stays a rate: 50 * 0.01 * 0.1.
	assert.InDelta(t, 0.05, consumed, 1e-9, "consumption should still scale with dt")

	// Half the range, half the signal, regardless of dt.
	ch["testosterone"] = 25.0
	out, _ = cr.GetOutput(ch, 0.1)
	assert.InDelta(t, 0.5, out, 1e-9)
	out, _ = cr.GetOutput(ch, 1.0)
	assert.InDelta(t, 0.5, out, 1e-9, "a neuron's signal must not depend on the step size")
}

// A modulator integrates what it is handed and decays on its own, so it must
// keep receiving a rate — otherwise its level would depend on how often the
// brain is stepped.
func TestChemicalReceptor_NormalizedModulatorOutputStaysARate(t *testing.T) {
	cr := &ChemicalReceptor{
		ChemicalName:           "dopamine",
		UptakeRate:             0.2,
		Target:                 "dopamine",
		TargetType:             TargetTypeModulator,
		ReceptorType:           ReceptorTypeExcitatory,
		OutputNormalizationMin: 0.0,
		OutputNormalizationMax: 10.0,
	}

	ch := map[string]float64{"dopamine": 10.0}
	out, _ := cr.GetOutput(ch, 0.1)
	assert.InDelta(t, 0.1, out, 1e-9)

	ch["dopamine"] = 10.0
	out, _ = cr.GetOutput(ch, 0.5)
	assert.InDelta(t, 0.5, out, 1e-9)
}
