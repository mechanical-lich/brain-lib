package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// A lobe's DecayRate is a default, not an extra decay stacked on top. The two
// used to multiply, so a neuron could only ever decay *faster* than its lobe —
// which made a long-lived memory impossible to express in a lobe of short-lived
// ones, and reduced a 10x per-neuron change to a 10% one.
func TestNeuron_DecayRateOverridesTheLobeDefault(t *testing.T) {
	n := &Neuron{ID: "n", DecayRate: 0.005}
	assert.Equal(t, 0.005, n.EffectiveDecayRate(0.05), "the neuron's own rate should win")

	unset := &Neuron{ID: "unset"}
	assert.Equal(t, 0.05, unset.EffectiveDecayRate(0.05), "an unset neuron should inherit the lobe's")
	assert.Equal(t, 0.0, unset.EffectiveDecayRate(0), "with neither set there is no decay")
}

// The end-to-end version: a slow neuron in a fast lobe has to actually outlast
// its neighbours.
func TestLobe_SlowNeuronOutlastsAFastLobe(t *testing.T) {
	l := NewLobe("memory", nil)
	l.Persists = true
	l.DecayRate = 1.0
	l.AddNeuron(&Neuron{ID: "fast"}) // inherits the lobe's 1.0
	l.AddNeuron(&Neuron{ID: "slow", DecayRate: 0.01})

	l.SetNeuronInput("fast", 1.0)
	l.SetNeuronInput("slow", 1.0)
	l.Step(0.1, 0, nil)

	for i := 0; i < 100; i++ {
		l.Step(0.1, 0, nil)
	}

	fast := l.GetNeuronOutput("fast")
	slow := l.GetNeuronOutput("slow")
	assert.Less(t, fast, 0.01, "the neuron inheriting the lobe's fast decay should be gone")
	assert.Greater(t, slow, 0.9, "the neuron with its own slow rate should still be holding")
}
