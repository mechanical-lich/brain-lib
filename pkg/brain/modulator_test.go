package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestModulatorAdd_ClampsAtZero(t *testing.T) {
	m := &Modulator{Name: Dopamine, Level: 0.3, DecayRate: 0.5}

	m.Add(0.2)
	assert.InDelta(t, 0.5, m.Level, 1e-9)

	// Overshooting downward floors at zero rather than going negative
	m.Add(-2.0)
	assert.InDelta(t, 0.0, m.Level, 1e-9)
}

func TestModulatorStep_ClampsLegacyNegativeLevel(t *testing.T) {
	// A brain saved before the clamp existed could carry a negative level
	m := &Modulator{Name: Serotonin, Level: -0.4, DecayRate: 0.5}

	m.Step(0.1)

	assert.InDelta(t, 0.0, m.Level, 1e-9)
}

// An inhibitory receptor pointed at a modulator must not be able to drive its
// level negative. If it could, applyReinforcementLearning's punishment term
// (-trace * rate * serotonin) would flip sign and *strengthen* the synapses it
// is meant to weaken.
func TestInhibitoryReceptor_CannotInvertPunishment(t *testing.T) {
	l := NewLobe("antagonism", nil)
	l.PlasticityConfig = PlasticityConfig{
		TraceDecay:         0.9,
		RewardLearningRate: 0.0, // isolate the punishment term
		PunishLearningRate: 0.5,
		InactivityDecay:    0.0,
	}

	if l.Modulators == nil {
		l.Modulators = make(map[string]*Modulator)
	}
	l.Modulators[Serotonin] = &Modulator{Name: Serotonin, Level: 0.3, DecayRate: 0.5}

	// Dopamine inhibits serotonin, mirroring the cross-receptors in the game's
	// brain data. A large dopamine surge would previously push serotonin below
	// zero.
	l.AddReceptor(Dopamine, 1.0, Serotonin, TargetTypeModulator, ReceptorTypeInhibitory)

	chemicals := map[string]float64{Dopamine: 50.0}
	l.UpdateChemicalReceptors(1.0, chemicals)

	assert.GreaterOrEqual(t, l.Modulators[Serotonin].Level, 0.0,
		"inhibition must not drive a modulator negative")

	// With serotonin floored at zero, a plastic synapse carrying a positive
	// trace must not gain weight from the punishment term.
	from := &Neuron{ID: "a"}
	to := &Neuron{ID: "b"}
	l.AddNeuron(from)
	l.AddNeuron(to)
	l.Connect("a", "b", 0.5, true)

	var syn *Synapse
	for _, s := range l.Synapses {
		syn = s
	}
	syn.Trace = 1.0
	before := syn.Weight

	l.applyReinforcementLearning()

	assert.LessOrEqual(t, syn.Weight, before,
		"punishment must never strengthen a synapse")
}
