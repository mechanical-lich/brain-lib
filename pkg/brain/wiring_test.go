package brain

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Brain.Reinforce's doc comment has always promised that a brain-level signal
// affects "any lobes that don't have their own modulators", but only the
// inter-lobe path implemented it: lobe.applyReinforcementLearning read
// l.Modulators alone, so a global dopamine signal left intra-lobe weights
// untouched however large the trace. Two reinforcement policies in one brain.
func TestBrainReinforce_ReachesIntraLobeSynapses(t *testing.T) {
	b := NewBrain()
	l := NewLobe("L", nil) // deliberately no lobe-level modulators
	l.AddNeuron(&Neuron{ID: "a", Bias: 1.0, Activation: ActLinear})
	l.AddNeuron(&Neuron{ID: "b", Bias: 0.5, Activation: ActLinear})
	syn := l.Connect("a", "b", 0.5, true)
	b.AddLobe(l)

	b.Reinforce(Dopamine, 5.0)
	b.Modulators[Dopamine].DecayRate = 0 // hold it steady

	before := syn.Weight
	for i := 0; i < 60; i++ {
		b.Step(1.0 / 60)
	}

	assert.Greater(t, syn.Weight, before,
		"a brain-level dopamine signal should reach a lobe with no modulators of its own")
}

// A lobe's own modulator still wins over the brain's.
func TestLobeModulatorOverridesBrainFallback(t *testing.T) {
	build := func(withLobeModulator bool) float64 {
		b := NewBrain()
		l := NewLobe("L", nil)
		l.AddNeuron(&Neuron{ID: "a", Bias: 1.0, Activation: ActLinear})
		l.AddNeuron(&Neuron{ID: "b", Bias: 0.5, Activation: ActLinear})
		syn := l.Connect("a", "b", 0.0, true)
		b.AddLobe(l)

		b.Reinforce(Dopamine, 0.1)
		b.Modulators[Dopamine].DecayRate = 0
		if withLobeModulator {
			l.Modulators[Dopamine] = &Modulator{Name: Dopamine, Level: 5.0, DecayRate: 0}
		}
		for i := 0; i < 60; i++ {
			b.Step(1.0 / 60)
		}
		return syn.Weight
	}

	assert.Greater(t, build(true), build(false),
		"the lobe's own modulator (5.0) should outweigh the brain fallback (0.1)")
}

// A lobe whose JSON omits plasticity_config used to load with an all-zero
// config, so its intra-lobe learning was silently off while inter-lobe learning
// still ran through the brain-level fallback.
func TestLoad_DefaultsLobePlasticityConfig(t *testing.T) {
	raw := []byte(`{
		"lobes": {"L": {"name": "L", "neurons": {"a": {"id": "a"}}, "synapses": {}}},
		"synapses": {},
		"modulators": {},
		"plasticity_config": {"trace_decay": 2.0, "reward_learning_rate": 7.0, "punish_learning_rate": 8.0, "inactivity_decay": 0.1}
	}`)

	b, err := LoadFromJson(raw)
	require.NoError(t, err)

	got := b.Lobes["L"].PlasticityConfig
	assert.NotZero(t, got.TraceDecay, "lobe plasticity config should not load as all zeros")
	assert.Equal(t, b.PlasticityConfig, got,
		"a lobe without its own config should inherit the brain's")
}

// Step matched ToLobe against lobe.Name but FromLobe against the map key. A file
// where the two disagreed gave a lobe that worked as a synapse source but was
// silently inert as a target.
func TestLoad_ReconcilesLobeNameWithMapKey(t *testing.T) {
	raw := []byte(`{
		"lobes": {"real_key": {"name": "stale_name", "neurons": {"a": {"id": "a"}}, "synapses": {}}},
		"synapses": {},
		"modulators": {},
		"plasticity_config": {"trace_decay": 1.0}
	}`)

	b, err := LoadFromJson(raw)
	require.NoError(t, err)

	require.Contains(t, b.Lobes, "real_key")
	assert.Equal(t, "real_key", b.Lobes["real_key"].Name,
		"the map key is the lobe's identity; a disagreeing name field makes it inert as a synapse target")
}

// applyInactivityDecay pulls an unused weight toward InitWeight. If InitWeight
// sits outside [MinWeight, MaxWeight] that pull points away from the legal
// range, so idling *strengthens* the synapse until it pins against the clip.
func TestInitWeightIsClampedIntoBounds(t *testing.T) {
	t.Run("Connect", func(t *testing.T) {
		l := NewLobe("L", nil)
		l.AddNeuron(&Neuron{ID: "a"})
		l.AddNeuron(&Neuron{ID: "b"})

		syn := l.Connect("a", "b", 50.0, true) // far outside the default [-2,2]

		assert.LessOrEqual(t, syn.InitWeight, syn.MaxWeight,
			"InitWeight outside the bounds inverts inactivity decay")
		assert.LessOrEqual(t, syn.Weight, syn.MaxWeight)
	})

	t.Run("Load", func(t *testing.T) {
		raw := []byte(`{
			"lobes": {},
			"synapses": {"s": {"id":"s","from":"a","to":"b","from_lobe":"X","to_lobe":"Y",
				"weight": 0.5, "init_weight": 50, "min_weight": -2, "max_weight": 2, "plastic": true}},
			"modulators": {},
			"plasticity_config": {"trace_decay": 1.0}
		}`)
		b, err := LoadFromJson(raw)
		require.NoError(t, err)
		assert.LessOrEqual(t, b.Synapses["s"].InitWeight, 2.0,
			"a loaded InitWeight outside the bounds should be clamped")
	})
}

// Idling must never push a weight away from its permitted range.
func TestInactivityDecayNeverStrengthensPastBounds(t *testing.T) {
	b := NewBrain()
	src, dst := NewLobe("src", nil), NewLobe("dst", nil)
	src.AddNeuron(&Neuron{ID: "a"})
	dst.AddNeuron(&Neuron{ID: "b"})
	b.AddLobe(src)
	b.AddLobe(dst)

	syn := b.Connect("src", "a", "dst", "b", 0.0, true)
	syn.MinWeight, syn.MaxWeight = -2, 2
	syn.InitWeight = 50 // the hazard, set after construction
	normalizeSynapse(syn)
	syn.Weight = 0.0

	b.Lobes["dst"].PlasticityConfig.InactivityDecay = 1.0
	for i := 0; i < 600; i++ {
		b.Step(1.0 / 60)
	}

	assert.LessOrEqual(t, syn.Weight, syn.MaxWeight)
	assert.GreaterOrEqual(t, syn.Weight, syn.MinWeight)
}

// A plastic synapse targeting a neuroemitter got no LastUsed, no UsageCount and
// no trace, because the input and trace loops only walk l.Neurons — yet
// applyInactivityDecay still saw it as permanently unused and dragged it back
// to InitWeight forever. It could only decay, never learn.
func TestEmitterTargetedSynapseCanLearn(t *testing.T) {
	b := NewBrain()
	l := NewLobe("L", nil)
	l.AddNeuron(&Neuron{ID: "a", Bias: 1.0, Activation: ActLinear})
	l.AddEmitter("e1", "dopamine", 0.5, 0.0)
	syn := l.Connect("a", "e1", 0.5, true)
	b.AddLobe(l)

	b.Reinforce(Dopamine, 5.0)
	b.Modulators[Dopamine].DecayRate = 0

	before := syn.Weight
	for i := 0; i < 60; i++ {
		b.Step(1.0 / 60)
	}

	assert.NotZero(t, syn.LastUsed, "an active emitter-targeted synapse should be marked used")
	assert.Greater(t, syn.Trace, 0.0, "it should accumulate an eligibility trace")
	assert.Greater(t, syn.Weight, before, "and therefore be able to learn")
}

// The shipped brain must survive a load/save round-trip through the new
// normalization without drifting.
func TestLoadNormalizationIsIdempotent(t *testing.T) {
	raw := []byte(`{
		"lobes": {"L": {"name": "L", "neurons": {"a": {"id":"a"}, "b": {"id":"b"}},
			"synapses": {"s": {"id":"s","from":"a","to":"b","from_lobe":"L","to_lobe":"L",
				"weight":0.5,"init_weight":0.5,"min_weight":-2,"max_weight":2}},
			"plasticity_config": {"trace_decay": 1.0, "reward_learning_rate": 10, "punish_learning_rate": 15, "inactivity_decay": 0.00001}}},
		"synapses": {},
		"modulators": {},
		"plasticity_config": {"trace_decay": 1.0, "reward_learning_rate": 10, "punish_learning_rate": 15, "inactivity_decay": 0.00001}
	}`)

	first, err := LoadFromJson(raw)
	require.NoError(t, err)
	out1, err := first.ToJSON()
	require.NoError(t, err)

	second, err := LoadFromJson(out1)
	require.NoError(t, err)
	out2, err := second.ToJSON()
	require.NoError(t, err)

	var a, c map[string]any
	require.NoError(t, json.Unmarshal(out1, &a))
	require.NoError(t, json.Unmarshal(out2, &c))
	assert.Equal(t, a, c, "loading a normalized brain should not change it again")
}

// A persisting neuron must follow its input back down, not hold its peak.
// Measured before the fix: driven to 1.98, input dropped to 0.5, activation
// stayed at 1.97.
func TestPersistingNeuronTracksInputDownward(t *testing.T) {
	n := &Neuron{ID: "n", Persists: true, Activation: ActLinear, DecayRate: 5.0}

	for i := 0; i < 60; i++ {
		n.Compute(2.0, 1.0/60, true)
	}
	peak := n.ActivationVal
	require.Greater(t, peak, 1.5)

	for i := 0; i < 60; i++ {
		n.Compute(0.5, 1.0/60, true)
	}

	assert.Less(t, n.ActivationVal, peak*0.75,
		"a persisting neuron should relax toward a lower input, not hold its peak")
	assert.InDelta(t, 0.5, n.ActivationVal, 0.15, "it should settle at the new input")
}

// Neuron.Min was unreachable: activation could only ratchet upward from zero,
// so an inhibitory synapse could cancel excitation but never invert it.
func TestNegativeActivationRequiresExplicitMin(t *testing.T) {
	t.Run("opted in", func(t *testing.T) {
		n := &Neuron{ID: "n", Activation: ActLinear, Min: -2}
		n.Compute(-5.0, 1.0/60, false)
		assert.Equal(t, -2.0, n.ActivationVal, "Min should be reachable when set below zero")
	})

	// Rectification is the default, because the brain's concept layer is built
	// from logic gates that need a sub-threshold sum to land on 0.
	t.Run("default rectifies", func(t *testing.T) {
		n := &Neuron{ID: "n", Activation: ActLinear}
		n.Compute(-5.0, 1.0/60, false)
		assert.Equal(t, 0.0, n.ActivationVal, "without an explicit Min a neuron stays rectified")
	})
}

// Winner-take-all must gate learning, not just the reported output. Measured
// before the fix: a loser reported activation 0.000 while accumulating a trace
// of 1.498 and gaining +1.5 of weight — reinforced for an action the lobe
// rejected.
func TestWinnerTakesAllGatesLearning(t *testing.T) {
	b := NewBrain()
	// Realistic shape: a source lobe feeding a winner-take-all decision lobe.
	// (Intra-lobe synapses inside a WTA lobe are a degenerate case — WTA
	// silences the presynaptic neuron too, so nothing can transmit.)
	src := NewLobe("src", nil)
	src.AddNeuron(&Neuron{ID: "s", Bias: 1.0, Activation: ActLinear})

	dec := NewLobe("dec", nil)
	dec.WinnerTakesAll = true
	dec.AddNeuron(&Neuron{ID: "winner", Bias: 2.0, Activation: ActLinear})
	dec.AddNeuron(&Neuron{ID: "loser", Bias: 0.1, Activation: ActLinear})

	b.AddLobe(src)
	b.AddLobe(dec)
	winSyn := b.Connect("src", "s", "dec", "winner", 0.5, true)
	loseSyn := b.Connect("src", "s", "dec", "loser", 0.5, true)

	b.Reinforce(Dopamine, 5.0)
	b.Modulators[Dopamine].DecayRate = 0

	for i := 0; i < 60; i++ {
		b.Step(1.0 / 60)
	}

	assert.Zero(t, dec.Neurons["loser"].ActivationVal, "the loser should be silenced")
	assert.InDelta(t, 0.0, loseSyn.Trace, 1e-9,
		"a silenced neuron must not accumulate eligibility trace")
	assert.InDelta(t, 0.5, loseSyn.Weight, 1e-9,
		"a synapse onto a rejected neuron must not be reinforced")
	assert.Greater(t, winSyn.Weight, 0.5, "the winner should still learn")
}

// Intra-lobe learning inside a WTA lobe is a consequence worth pinning: WTA
// silences every neuron but one, including presynaptic ones, so an intra-lobe
// synapse has nothing to transmit. WTA lobes are output layers of competing
// alternatives; internal wiring belongs upstream.
func TestWinnerTakesAllSilencesIntraLobeSources(t *testing.T) {
	b := NewBrain()
	l := NewLobe("wta", nil)
	l.WinnerTakesAll = true
	l.AddNeuron(&Neuron{ID: "src", Bias: 1.0, Activation: ActLinear})
	l.AddNeuron(&Neuron{ID: "winner", Bias: 5.0, Activation: ActLinear})
	syn := l.Connect("src", "winner", 0.5, true)
	b.AddLobe(l)
	b.Reinforce(Dopamine, 5.0)
	b.Modulators[Dopamine].DecayRate = 0

	for i := 0; i < 60; i++ {
		b.Step(1.0 / 60)
	}

	assert.Zero(t, l.Neurons["src"].ActivationVal, "WTA silences the source too")
	assert.InDelta(t, 0.5, syn.Weight, 1e-9, "so an intra-lobe synapse cannot learn here")
}

// Ties must not depend on Go's randomised map iteration.
func TestWinnerTakesAllTieIsDeterministic(t *testing.T) {
	winners := map[string]int{}
	for i := 0; i < 50; i++ {
		l := NewLobe("wta", nil)
		l.WinnerTakesAll = true
		for _, id := range []string{"a", "b", "c"} {
			l.AddNeuron(&Neuron{ID: id, Bias: 1.0, Activation: ActLinear})
		}
		l.Step(1.0/60, 0, nil)
		for id, n := range l.Neurons {
			if n.ActivationVal != 0 {
				winners[id]++
			}
		}
	}
	require.Len(t, winners, 1, "a tie should always resolve to the same neuron, got %v", winners)
}
