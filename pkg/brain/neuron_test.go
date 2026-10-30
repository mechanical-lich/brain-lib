package brain

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

// memoryLobe is a persisting lobe holding one memory neuron that decays at 1/s.
func memoryLobe(threshold float64) *Lobe {
	l := NewLobe("memory", nil)
	l.Persists = true
	l.AddNeuron(&Neuron{ID: "mem", DecayRate: 1.0, Max: 1, Threshold: threshold})
	return l
}

// stepsUntilZero pulses the lobe's memory once and counts the steps until its
// activation is exactly 0, or returns -1 if it is still non-zero after limit.
func stepsUntilZero(l *Lobe, dt float64, limit int) int {
	l.SetNeuronInput("mem", l.Neurons["mem"].Max)
	l.Step(dt, 0, nil)
	for i := 1; i <= limit; i++ {
		l.Step(dt, i, nil)
		if l.Neurons["mem"].ActivationVal == 0 {
			return i
		}
	}
	return -1
}

// Threshold used to gate only a step's input, never the activation a
// persisting neuron carries over, so a memory pulsed once faded toward 0
// forever without reaching it. It now ends exactly once it decays below
// Threshold: at rate 1/s from 1.0 under a 0.05 threshold that is ln(20) ≈ 3.0s.
func TestNeuron_PersistingMemoryEndsAtThreshold(t *testing.T) {
	steps := stepsUntilZero(memoryLobe(0.05), 0.1, 100)
	assert.GreaterOrEqual(t, steps, 29, "should not end before it decays below threshold")
	assert.LessOrEqual(t, steps, 31, "should end within a step of ln(20) seconds")
}

// The point of ending exactly: inactivity decay only treats a synapse as unused
// when its source is exactly 0, so the plastic link a faded memory feeds could
// never extinguish. This is the critters recently_ate_* -> craving shape, an
// inter-lobe link out of a persisting memory.
func TestNeuron_EndedMemoryLetsItsLinkDecay(t *testing.T) {
	b := NewBrain()
	mem := memoryLobe(0.05)
	dst := NewLobe("craving", nil)
	dst.AddNeuron(&Neuron{ID: "want"})
	dst.PlasticityConfig.InactivityDecay = 1.0
	b.AddLobe(mem)
	b.AddLobe(dst)
	syn := b.Connect("memory", "mem", "craving", "want", 1.0, true)
	syn.InitWeight = 0

	mem.SetNeuronInput("mem", 1.0)
	held, ended := 0.0, false
	for i := 0; i < 100 && !ended; i++ {
		held = b.Synapses[syn.ID].Weight
		b.Step(0.1)
		ended = mem.Neurons["mem"].ActivationVal == 0
	}
	require.True(t, ended, "the memory should end")
	assert.Equal(t, 1.0, held, "the link should hold for as long as the memory is active")

	runFor(b, 2.0, 0.1)
	assert.Less(t, b.Synapses[syn.ID].Weight, 0.2, "once the memory has ended its link should decay toward InitWeight")
}

// Threshold 0 means no threshold: the carry-over still decays asymptotically,
// exactly as before.
func TestNeuron_ZeroThresholdMemoryIsUnchanged(t *testing.T) {
	l := memoryLobe(0)
	assert.Equal(t, -1, stepsUntilZero(l, 0.1, 100))
	assert.InEpsilon(t, math.Exp(-10.0), l.Neurons["mem"].ActivationVal, 1e-9,
		"10s at rate 1/s should leave exp(-10) of the pulse")
}

// A signed neuron (Min < 0) ends from below the same way.
func TestNeuron_SignedMemoryEndsFromBelow(t *testing.T) {
	l := NewLobe("memory", nil)
	l.Persists = true
	l.AddNeuron(&Neuron{ID: "mem", DecayRate: 1.0, Min: -1, Max: 1, Threshold: 0.05})
	l.SetNeuronInput("mem", -1)
	l.Step(0.1, 0, nil)
	require.Equal(t, -1.0, l.Neurons["mem"].ActivationVal)
	for i := 1; i <= 31; i++ {
		l.Step(0.1, i, nil)
	}
	assert.Equal(t, 0.0, l.Neurons["mem"].ActivationVal)
}

// A capacitor integrates below its threshold by design — the critters
// cant_reach_* capacitors gain 0.2 a step under a 0.3 threshold — so a charge
// that has not yet reached Threshold is kept, even with the input gone.
func TestNeuron_CapacitorKeepsSubThresholdCharge(t *testing.T) {
	n := &Neuron{ID: "cap", Capacitor: true, Max: 2, Threshold: 0.3}
	n.ExternalInput = 2.0
	n.Compute(0, 0.1, true, 0.005)
	require.InDelta(t, 0.2, n.ActivationVal, 1e-3)

	n.Compute(0, 0.1, true, 0.005)
	assert.Greater(t, n.ActivationVal, 0.19, "a gap in the input must not discard the charge")

	n.ExternalInput = 2.0
	n.Compute(0, 0.1, true, 0.005)
	assert.Greater(t, n.ActivationVal, 0.3, "intermittent input should still accumulate past threshold")
}

// Once charged past Threshold, a capacitor that leaks back down through it with
// nothing driving it ends exactly: from 1.0 at 0.5/s under a 0.3 threshold that
// is ln(1/0.3)/0.5 ≈ 2.4s.
func TestNeuron_DischargedCapacitorEndsAtThreshold(t *testing.T) {
	n := &Neuron{ID: "cap", Capacitor: true, Max: 2, Threshold: 0.3, ActivationVal: 1.0}
	steps := -1
	for i := 1; i <= 100; i++ {
		n.Compute(0, 0.1, true, 0.5)
		if n.ActivationVal == 0 {
			steps = i
			break
		}
	}
	assert.GreaterOrEqual(t, steps, 24)
	assert.LessOrEqual(t, steps, 25)
}
