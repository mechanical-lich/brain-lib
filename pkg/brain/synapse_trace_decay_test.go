package brain

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Synapse.TraceDecay lets a few synapses carry a wider reinforcement window
// than the rest of their lobe. Before it, the window was lobe-wide: widening it
// for the teachable synapses into a lobe widened it for every plastic synapse
// into that lobe, so pathways nobody meant to make teachable learned from the
// creature's own behaviour several times faster too.
//
// Each fixture below primes two traces to 1.0 behind a silent source neuron
// (bias 0, so the input term pre*post is exactly zero) and lets them decay for
// one second. One synapse sets its own rate, the other leaves it at 0, so after
// a second they should read exp(-own) and exp(-lobe's).

const (
	lobeTraceDecay = 2.0
	ownTraceDecay  = 0.15
)

// decayOneSecond runs one simulated second at 60 steps per second.
func decayOneSecond(b *Brain) {
	runFor(b, 1.0, 1.0/60)
}

func TestSynapseTraceDecayOverridesLobe_IntraLobe(t *testing.T) {
	b := NewBrain()
	l := NewLobe("L", nil)
	l.PlasticityConfig.TraceDecay = lobeTraceDecay
	l.AddNeuron(&Neuron{ID: "a", Activation: ActLinear})
	l.AddNeuron(&Neuron{ID: "b", Activation: ActLinear})
	l.AddNeuron(&Neuron{ID: "c", Activation: ActLinear})
	own := l.Connect("a", "b", 0.5, true)
	inherited := l.Connect("a", "c", 0.5, true)
	own.TraceDecay = ownTraceDecay
	own.Trace, inherited.Trace = 1.0, 1.0
	b.AddLobe(l)

	decayOneSecond(b)

	assert.InDelta(t, math.Exp(-ownTraceDecay), own.Trace, 1e-9, "own TraceDecay should set the window")
	assert.InDelta(t, math.Exp(-lobeTraceDecay), inherited.Trace, 1e-9, "zero should fall back to the lobe's")
}

func TestSynapseTraceDecayOverridesLobe_EmitterTarget(t *testing.T) {
	b := NewBrain()
	l := NewLobe("L", nil)
	l.PlasticityConfig.TraceDecay = lobeTraceDecay
	l.AddNeuron(&Neuron{ID: "a", Activation: ActLinear})
	l.AddEmitter("e1", "dopamine", 0.5, 0.0)
	l.AddEmitter("e2", "serotonin", 0.5, 0.0)
	own := l.Connect("a", "e1", 0.5, true)
	inherited := l.Connect("a", "e2", 0.5, true)
	own.TraceDecay = ownTraceDecay
	own.Trace, inherited.Trace = 1.0, 1.0
	b.AddLobe(l)

	decayOneSecond(b)

	assert.InDelta(t, math.Exp(-ownTraceDecay), own.Trace, 1e-9, "own TraceDecay should set the window")
	assert.InDelta(t, math.Exp(-lobeTraceDecay), inherited.Trace, 1e-9, "zero should fall back to the lobe's")
}

func TestSynapseTraceDecayOverridesLobe_InterLobe(t *testing.T) {
	b := NewBrain()
	src := NewLobe("src", nil)
	dst := NewLobe("dst", nil)
	// The source lobe's rate must not matter: an inter-lobe synapse falls back
	// to its target lobe's config.
	src.PlasticityConfig.TraceDecay = 9.0
	dst.PlasticityConfig.TraceDecay = lobeTraceDecay
	src.AddNeuron(&Neuron{ID: "a", Activation: ActLinear})
	dst.AddNeuron(&Neuron{ID: "b", Activation: ActLinear})
	dst.AddNeuron(&Neuron{ID: "c", Activation: ActLinear})
	b.AddLobe(src)
	b.AddLobe(dst)
	own := b.Connect("src", "a", "dst", "b", 0.5, true)
	inherited := b.Connect("src", "a", "dst", "c", 0.5, true)
	own.TraceDecay = ownTraceDecay
	own.Trace, inherited.Trace = 1.0, 1.0

	decayOneSecond(b)

	assert.InDelta(t, math.Exp(-ownTraceDecay), own.Trace, 1e-9, "own TraceDecay should set the window")
	assert.InDelta(t, math.Exp(-lobeTraceDecay), inherited.Trace, 1e-9, "zero should fall back to the target lobe's")
}

// The field is additive: written when set, omitted when zero, and read back
// unchanged, so CurrentFormatVersion did not need to move.
func TestSynapseTraceDecayRoundTrips(t *testing.T) {
	b := fullyFeaturedBrain()
	require.Len(t, b.Synapses, 1, "fixture has one inter-lobe synapse")
	var id string
	for id = range b.Synapses {
	}
	b.Synapses[id].TraceDecay = ownTraceDecay
	// The fixture's intra-lobe synapse keeps the default of 0.

	data, err := b.ToJSON()
	require.NoError(t, err)

	loaded, err := LoadFromJson(data)
	require.NoError(t, err)
	assert.Equal(t, CurrentFormatVersion, loaded.FormatVersion)
	require.Contains(t, loaded.Synapses, id)
	assert.InDelta(t, ownTraceDecay, loaded.Synapses[id].TraceDecay, 1e-12)
	for sid, s := range loaded.Lobes["src"].Synapses {
		assert.Zero(t, s.TraceDecay, "intra-lobe synapse %s should still use its lobe's", sid)
	}

	var raw struct {
		Synapses map[string]map[string]any `json:"synapses"`
		Lobes    map[string]struct {
			Synapses map[string]map[string]any `json:"synapses"`
		} `json:"lobes"`
	}
	require.NoError(t, json.Unmarshal(data, &raw))
	assert.Contains(t, raw.Synapses[id], "trace_decay")
	for sid, s := range raw.Lobes["src"].Synapses {
		assert.NotContains(t, s, "trace_decay", "zero TraceDecay on %s should be omitted", sid)
	}
}

// A per-synapse rate runs through the same exact decayFactor as the lobe's, so
// the trace it produces over a simulated second must not depend on frame rate.
func TestSynapseTraceDecayIsFrameRateIndependent(t *testing.T) {
	var traces []float64
	for _, dt := range []float64{1.0 / 15, 1.0 / 60, 1.0 / 240} {
		b, syn := learningBrain()
		b.Lobes["dst"].PlasticityConfig.TraceDecay = 1.0
		syn.TraceDecay = ownTraceDecay
		runFor(b, 1.0, dt)
		traces = append(traces, b.Synapses[syn.ID].Trace)
	}

	reference := traces[len(traces)-1]
	require.Greater(t, reference, 0.0)
	for i, got := range traces {
		rel := math.Abs(got-reference) / reference
		assert.Less(t, rel, 0.25, "trace %d (%.4f) vs %.4f", i, got, reference)
	}

	// And the override is actually in effect: the lobe's narrower window
	// gives a visibly smaller trace over the same second.
	b, syn := learningBrain()
	b.Lobes["dst"].PlasticityConfig.TraceDecay = 1.0
	runFor(b, 1.0, 1.0/240)
	assert.Less(t, b.Synapses[syn.ID].Trace, reference,
		"the lobe's TraceDecay 1.0 should leave less trace than a synapse's own 0.15")
}
