package brain

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Two-factor (Hebbian) plasticity by the instar rule: a synapse with Hebbian
// set moves by
//
//	dW/dt = HebbianRate·post·(pre − W) + HebbianDecay·(InitWeight − W)
//
// with no modulator. Before it, every plastic change in the library needed
// dopamine or serotonin, so "these two things happened together" —
// familiarity, feature conjunctions — could not be learned without the game
// handing down a reward.

const (
	hebbRate  = 0.5 // r, per second
	hebbDecay = 0.2 // δ, per second
)

// hebbianLobe builds a lobe where "a" feeds a Hebbian-only synapse into "b"
// and a Plastic-only control into "c". b and c clamp at 1, so while all three
// are driven at 1 the synapses cannot change their own postsynaptic activity
// and pre = post = 1 exactly — the weight then has a closed form.
func hebbianLobe() (*Brain, *Lobe, *Synapse, *Synapse) {
	b := NewBrain()
	l := NewLobe("L", nil)
	l.PlasticityConfig.HebbianRate = hebbRate
	l.PlasticityConfig.HebbianDecay = hebbDecay
	l.AddNeuron(&Neuron{ID: "a", Activation: ActLinear})
	l.AddNeuron(&Neuron{ID: "b", Activation: ActLinear, Max: 1})
	l.AddNeuron(&Neuron{ID: "c", Activation: ActLinear, Max: 1})
	hebb := l.Connect("a", "b", 0.0, false)
	hebb.Hebbian = true
	hebb.MinWeight, hebb.MaxWeight = 0, 1
	control := l.Connect("a", "c", 0.3, true)
	b.AddLobe(l)
	return b, l, hebb, control
}

// drive runs `seconds` of simulated time with the named neurons held at an
// external input of 1.
func drive(b *Brain, l *Lobe, seconds, dt float64, ids ...string) {
	for i := 0; i < int(math.Round(seconds/dt)); i++ {
		for _, id := range ids {
			l.Neurons[id].ExternalInput = 1
		}
		b.Step(dt)
	}
}

// instarTarget and instarRate give the fixed point and relaxation rate of the
// rule for activations held constant.
func instarTarget(r, d, pre, post, w0 float64) float64 { return (r*post*pre + d*w0) / (r*post + d) }
func instarRate(r, d, post float64) float64            { return r*post + d }

// (a) Conditioning. With pre = post = 1 from the first step, W relaxes from
// W0 = 0 toward r/(r+δ) at rate r+δ:
//
//	W(t) = r/(r+δ) · (1 − e^{−(r+δ)t})
//
// No modulator exists anywhere in the brain, so the Plastic-only control has
// a trace but no reason to move.
func TestHebbianConditioningReachesPredictedWeight(t *testing.T) {
	b, l, hebb, control := hebbianLobe()
	const seconds = 3.0
	drive(b, l, seconds, 1.0/60, "a", "b", "c")

	target := instarTarget(hebbRate, hebbDecay, 1, 1, 0)
	predicted := target * (1 - math.Exp(-instarRate(hebbRate, hebbDecay, 1)*seconds))
	t.Logf("conditioning after %.0fs: predicted %.6f, measured %.6f", seconds, predicted, hebb.Weight)

	assert.InDelta(t, predicted, hebb.Weight, 1e-9, "Hebbian weight should follow the instar rule with no modulator")
	assert.Zero(t, hebb.Trace, "the Hebbian rule does not use, or keep, an eligibility trace")
	assert.Greater(t, control.Trace, 0.0, "the control was co-active too")
	assert.Equal(t, 0.3, control.Weight, "a Plastic-only synapse must not learn without a modulator")
}

// Depression: while the target fires and the source is silent, the weight
// falls toward the source's activation (0). This is the half of the rule that
// lets a winner-takes-all lobe recruit fresh slots.
func TestHebbianDepressesWeightFromAbsentInput(t *testing.T) {
	b, l, hebb, _ := hebbianLobe()
	hebb.InitWeight, hebb.Weight = 0.3, 0.3
	const seconds = 2.0
	drive(b, l, seconds, 1.0/60, "b") // target fires, source silent

	target := instarTarget(hebbRate, hebbDecay, 0, 1, 0.3)
	predicted := target + (0.3-target)*math.Exp(-instarRate(hebbRate, hebbDecay, 1)*seconds)
	t.Logf("depression after %.0fs: predicted %.6f, measured %.6f", seconds, predicted, hebb.Weight)
	assert.InDelta(t, predicted, hebb.Weight, 1e-9)
	assert.Less(t, hebb.Weight, 0.3)
}

// (b) Forgetting. With the target silent only the decay term runs, so a
// Hebbian weight relaxes exactly as W0 + (W - W0)·e^{-δt}. A Plastic-only
// synapse's inactivity decay is unchanged: still the per-step Euler form,
// applied only because it is unused.
func TestHebbianForgettingDecaysTowardInitWeight(t *testing.T) {
	b, l, hebb, control := hebbianLobe()
	l.PlasticityConfig.InactivityDecay = 0.5
	hebb.Weight = 0.9
	control.InitWeight, control.Weight = 0, 1.0

	const seconds, dt = 5.0, 1.0 / 60
	drive(b, l, seconds, dt) // nothing driven

	predicted := 0.9 * math.Exp(-hebbDecay*seconds)
	t.Logf("forgetting after %.0fs: predicted %.6f, measured %.6f", seconds, predicted, hebb.Weight)
	assert.InDelta(t, predicted, hebb.Weight, 1e-9, "Hebbian weight should relax toward InitWeight at HebbianDecay")

	steps := int(math.Round(seconds / dt))
	assert.InDelta(t, math.Pow(1-0.5*dt, float64(steps)), control.Weight, 1e-9,
		"inactivity decay of a Plastic-only synapse should behave as before")
}

// Hebbian forgetting is passive: unlike inactivity decay it also runs while
// the synapse is in use. With the source active but the target silent, the
// weight still relaxes.
func TestHebbianDecayRunsWhileUsed(t *testing.T) {
	b, l, hebb, _ := hebbianLobe()
	hebb.Weight = 1.0
	l.Neurons["b"].Max = 0
	l.Neurons["b"].Bias = -10 // rectified to 0: a is used, but post is 0
	drive(b, l, 1.0, 1.0/60, "a")
	assert.Equal(t, b.Tick, hebb.LastUsed, "the synapse was in use")
	assert.InDelta(t, math.Exp(-hebbDecay), hebb.Weight, 1e-9)
}

// A neuroemitter has no postsynaptic activation, so a Hebbian synapse into
// one never learns; it only forgets toward InitWeight.
func TestHebbianSynapseIntoEmitterOnlyForgets(t *testing.T) {
	b := NewBrain()
	l := NewLobe("L", nil)
	l.PlasticityConfig.HebbianRate = 1.0
	l.PlasticityConfig.HebbianDecay = 0.1
	l.AddNeuron(&Neuron{ID: "a", Bias: 1.0, Activation: ActLinear})
	l.AddEmitter("e", "dopamine", 0.5, 0.0)
	syn := l.Connect("a", "e", 0.2, false)
	syn.Hebbian = true
	syn.Weight = 0.5
	b.AddLobe(l)
	runFor(b, 1.0, 1.0/60)
	assert.InDelta(t, 0.2+0.3*math.Exp(-0.1), syn.Weight, 1e-9)
}

// hebbianInterLobeBrain is learningBrain's shape with no dopamine and a
// Hebbian-only synapse. The target's activity grows with the weight, so this
// exercises a changing post activation rather than a closed form.
func hebbianInterLobeBrain() (*Brain, *Synapse) {
	b := NewBrain()
	src := NewLobe("src", nil)
	dst := NewLobe("dst", nil)
	dst.PlasticityConfig.HebbianRate = 2.0
	dst.PlasticityConfig.HebbianDecay = 0.5
	src.AddNeuron(&Neuron{ID: "a", Bias: 1.0, Activation: ActLinear})
	dst.AddNeuron(&Neuron{ID: "b", Bias: 0.5, Activation: ActLinear})
	b.AddLobe(src)
	b.AddLobe(dst)
	syn := b.Connect("src", "a", "dst", "b", 0.0, false)
	syn.Hebbian = true
	syn.MinWeight, syn.MaxWeight = 0, 1
	return b, syn
}

// hebbianIntraLobeBrain is the same circuit inside one lobe.
func hebbianIntraLobeBrain() (*Brain, *Synapse) {
	b := NewBrain()
	l := NewLobe("L", nil)
	l.PlasticityConfig.HebbianRate = 2.0
	l.PlasticityConfig.HebbianDecay = 0.5
	l.AddNeuron(&Neuron{ID: "a", Bias: 1.0, Activation: ActLinear})
	l.AddNeuron(&Neuron{ID: "b", Bias: 0.5, Activation: ActLinear})
	syn := l.Connect("a", "b", 0.0, false)
	syn.Hebbian = true
	syn.MinWeight, syn.MaxWeight = 0, 1
	b.AddLobe(l)
	return b, syn
}

// (c) The same simulated second must give the same Hebbian weight whatever
// the frame rate, on both plasticity paths. Each step is integrated exactly
// for its activations, so what remains is post lagging W by one step.
func TestHebbianLearningIsFrameRateIndependent(t *testing.T) {
	for name, build := range map[string]func() (*Brain, *Synapse){
		"inter-lobe": hebbianInterLobeBrain,
		"intra-lobe": hebbianIntraLobeBrain,
	} {
		t.Run(name, func(t *testing.T) {
			var weights []float64
			for _, tps := range []float64{15, 60, 600} {
				b, syn := build()
				runFor(b, 1.0, 1/tps)
				weights = append(weights, syn.Weight)
			}
			reference := weights[len(weights)-1]
			t.Logf("weight after 1s at 15/60/600 TPS: %.6f / %.6f / %.6f", weights[0], weights[1], weights[2])
			require.Greater(t, reference, 0.0, "the fixture should learn without any modulator")
			for i, got := range weights {
				rel := math.Abs(got-reference) / reference
				assert.Less(t, rel, 0.02, "weight %d (%.6f) vs %.6f at 600 TPS", i, got, reference)
			}
		})
	}
}

// (d) A synapse with both flags learns from co-activity alone, and from
// reward on top.
func TestHebbianAndPlasticSynapseLearnsFromBoth(t *testing.T) {
	run := func(dopamine bool) (hebbOnly, plasticOnly, both *Synapse) {
		b := NewBrain()
		l := NewLobe("L", nil)
		l.PlasticityConfig.TraceDecay = 1.0
		l.PlasticityConfig.HebbianRate = 0.5
		l.PlasticityConfig.HebbianDecay = 0.1
		l.PlasticityConfig.RewardLearningRate = 0.4
		l.AddNeuron(&Neuron{ID: "a", Activation: ActLinear})
		for _, id := range []string{"b1", "b2", "b3"} {
			l.AddNeuron(&Neuron{ID: id, Activation: ActLinear, Max: 1})
		}
		hebbOnly = l.Connect("a", "b1", 0.2, false)
		hebbOnly.Hebbian = true
		plasticOnly = l.Connect("a", "b2", 0.2, true)
		both = l.Connect("a", "b3", 0.2, true)
		both.Hebbian = true
		b.AddLobe(l)
		if dopamine {
			b.Reinforce(Dopamine, 1.0)
			b.Modulators[Dopamine].DecayRate = 0
		}
		drive(b, l, 1.0, 1.0/60, "a", "b1", "b2", "b3")
		return
	}

	hebb, plastic, both := run(false)
	assert.Greater(t, hebb.Weight, 0.2)
	assert.Equal(t, 0.2, plastic.Weight, "no modulator: the reward term is silent")
	assert.InDelta(t, hebb.Weight, both.Weight, 1e-12, "co-activity alone: both-flags learns like Hebbian-only")

	hebbAlone := hebb.Weight
	hebb, plastic, both = run(true)
	assert.InDelta(t, hebbAlone, hebb.Weight, 1e-12, "dopamine does not touch a Hebbian-only synapse")
	assert.Greater(t, plastic.Weight, 0.2)
	assert.Greater(t, both.Weight, hebb.Weight, "with dopamine: both-flags also gets the reward term")
}

// (e) The new fields are additive: written when set, omitted when zero or
// false, and read back unchanged.
func TestHebbianFieldsRoundTrip(t *testing.T) {
	b := fullyFeaturedBrain()
	var id string
	for id = range b.Synapses {
	}
	b.Synapses[id].Hebbian = true
	b.Lobes["src"].PlasticityConfig.HebbianRate = 0.75
	b.Lobes["src"].PlasticityConfig.HebbianDecay = 0.05

	data, err := b.ToJSON()
	require.NoError(t, err)
	loaded, err := LoadFromJson(data)
	require.NoError(t, err)
	assert.Equal(t, CurrentFormatVersion, loaded.FormatVersion)
	assert.True(t, loaded.Synapses[id].Hebbian)
	assert.Equal(t, 0.75, loaded.Lobes["src"].PlasticityConfig.HebbianRate)
	assert.Equal(t, 0.05, loaded.Lobes["src"].PlasticityConfig.HebbianDecay)
	for sid, s := range loaded.Lobes["src"].Synapses {
		assert.False(t, s.Hebbian, "intra-lobe synapse %s was not Hebbian", sid)
	}

	var raw struct {
		Synapses map[string]map[string]any `json:"synapses"`
		Lobes    map[string]struct {
			Synapses         map[string]map[string]any `json:"synapses"`
			PlasticityConfig map[string]any            `json:"plasticity_config"`
		} `json:"lobes"`
	}
	require.NoError(t, json.Unmarshal(data, &raw))
	assert.Contains(t, raw.Synapses[id], "hebbian")
	assert.Contains(t, raw.Lobes["src"].PlasticityConfig, "hebbian_rate")
	for sid, s := range raw.Lobes["src"].Synapses {
		assert.NotContains(t, s, "hebbian", "false Hebbian on %s should be omitted", sid)
	}
	assert.NotContains(t, raw.Lobes["dst"].PlasticityConfig, "hebbian_rate")
	assert.NotContains(t, raw.Lobes["dst"].PlasticityConfig, "hebbian_decay")
}

// Worked example: a familiarity memory. "familiar" is a conjunction neuron fed
// by feature neurons over Hebbian synapses at 0.3 with a threshold of 0.5, so
// at first it fires only when two features arrive together (0.6) and never
// for one alone. Each co-presentation of f1 and f2 pulls both links toward 1
// while it fires; once they pass 0.5 either feature alone is recognised. f3
// was silent every time "familiar" fired, so the instar rule pulled its link
// down toward 0: still not recognised, and further from it than at the start.
// No reward was involved at any point.
func TestHebbianFamiliarityMemory(t *testing.T) {
	b := NewBrain()
	l := NewLobe("memory", nil)
	l.PlasticityConfig.HebbianRate = 0.5
	l.PlasticityConfig.HebbianDecay = 0.02 // familiarity fades over about a minute
	for _, id := range []string{"f1", "f2", "f3"} {
		l.AddNeuron(&Neuron{ID: id, Activation: ActLinear})
	}
	familiar := &Neuron{ID: "familiar", Activation: ActLinear, Threshold: 0.5, Max: 1}
	l.AddNeuron(familiar)
	for _, id := range []string{"f1", "f2", "f3"} {
		syn := l.Connect(id, "familiar", 0.3, false)
		syn.Hebbian = true
		syn.MinWeight, syn.MaxWeight = 0, 1 // the recommended [0, 1] bounds
	}
	b.AddLobe(l)

	const dt = 1.0 / 60
	probe := func(ids ...string) float64 {
		drive(b, l, dt, dt, ids...)
		got := familiar.ActivationVal
		drive(b, l, 0.5, dt) // let it settle back
		return got
	}

	require.Zero(t, probe("f1"), "one feature alone starts below threshold")
	require.Zero(t, probe("f3"))
	require.Greater(t, probe("f1", "f2"), 0.5, "the conjunction fires from the start")

	presentations := 0
	for ; presentations < 20 && probe("f1") == 0; presentations++ {
		drive(b, l, 0.5, dt, "f1", "f2") // co-present for half a second
		drive(b, l, 0.5, dt)             // then a gap
	}
	f3 := l.Synapses["f3->familiar"].Weight
	t.Logf("f1 alone recognised after %d co-presentations; weights f1 %.3f, f2 %.3f, f3 %.3f",
		presentations, l.Synapses["f1->familiar"].Weight, l.Synapses["f2->familiar"].Weight, f3)

	assert.GreaterOrEqual(t, probe("f1"), 0.5, "f1 alone should now be familiar")
	assert.GreaterOrEqual(t, probe("f2"), 0.5, "and so should f2")
	assert.Zero(t, probe("f3"), "a feature never co-presented stays unfamiliar")
	assert.Less(t, f3, 0.3, "and its link was depressed while the conjunction fired without it")
	assert.Greater(t, presentations, 1, "recognition should take more than one encounter")
}

// recognitionBrain is the critters recognition-lobe shape: feature senses in
// their own lobe (a WTA lobe would zero them along with the losing slots)
// feeding a winner-takes-all lobe of slot neurons over Hebbian synapses from
// every feature. Every synapse starts at a small uniform InitWeight with
// bounds [0, 1], and every slot has the same small bias.
func recognitionBrain() (*Brain, *Lobe, *Lobe) {
	b := NewBrain()
	senses := NewLobe("senses", nil)
	rec := NewLobe("recognition", nil)
	rec.WinnerTakesAll = true
	rec.PlasticityConfig.HebbianRate = 2.0
	rec.PlasticityConfig.HebbianDecay = 0.05
	for _, f := range []string{"f1", "f2", "f3", "f4"} {
		senses.AddNeuron(&Neuron{ID: f, Activation: ActLinear})
	}
	for _, s := range []string{"s1", "s2", "s3"} {
		rec.AddNeuron(&Neuron{ID: s, Bias: 0.1, Activation: ActLinear, Max: 1})
	}
	b.AddLobe(senses)
	b.AddLobe(rec)
	for _, f := range []string{"f1", "f2", "f3", "f4"} {
		for _, s := range []string{"s1", "s2", "s3"} {
			syn := b.Connect("senses", f, "recognition", s, recognitionInit, false)
			syn.Hebbian = true
			syn.MinWeight, syn.MaxWeight = 0, 1
		}
	}
	return b, senses, rec
}

// recognitionInit is the small positive InitWeight recruitment depends on: a
// trained slot is depressed below it on features it never saw, so an
// untrained slot outbids it on a novel bundle.
const recognitionInit = 0.05

// present holds the named features at 1 for `seconds` and returns the slot
// that won the last step.
func present(b *Brain, senses, rec *Lobe, seconds float64, features ...string) string {
	const dt = 1.0 / 60
	for i := 0; i < int(math.Round(seconds/dt)); i++ {
		for _, f := range features {
			senses.Neurons[f].ExternalInput = 1
		}
		b.Step(dt)
	}
	for _, id := range []string{"s1", "s2", "s3"} {
		if rec.Neurons[id].ActivationVal > 0 {
			return id
		}
	}
	return ""
}

func slotSynapse(b *Brain, feature, slot string) *Synapse {
	for _, syn := range b.Synapses {
		if syn.From == feature && syn.To == slot {
			return syn
		}
	}
	return nil
}

// Winner-takes-all resolves before plasticity, so only the winning slot sees a
// postsynaptic activation: the losers are zeroed first and their weights never
// move. The winner's weights move toward the bundle — up from present
// features, down from absent ones.
func TestWinnerTakesAllGatesHebbianLearning(t *testing.T) {
	b, senses, rec := recognitionBrain()

	winner := present(b, senses, rec, 1.0, "f1", "f2")
	require.Equal(t, "s1", winner, "equal biases and weights: the tie breaks by ID")

	for _, f := range []string{"f1", "f2"} {
		assert.Greater(t, slotSynapse(b, f, "s1").Weight, recognitionInit, "the winner binds %s", f)
	}
	for _, f := range []string{"f3", "f4"} {
		assert.Less(t, slotSynapse(b, f, "s1").Weight, recognitionInit, "the winner unbinds absent %s", f)
	}
	for _, s := range []string{"s2", "s3"} {
		for _, f := range []string{"f1", "f2", "f3", "f4"} {
			assert.Equal(t, recognitionInit, slotSynapse(b, f, s).Weight, "loser %s must not learn %s", s, f)
		}
	}
}

// Recruitment: a second, non-overlapping bundle goes to a fresh slot. s1 lost
// its weight from f3/f4 while learning f1+f2, so on f3+f4 it gets little more
// than its bias, while an untrained slot still holds 0.05 from each and wins.
// The first binding survives: f1+f2 still goes to s1 afterwards.
func TestHebbianWTARecruitsAFreshSlot(t *testing.T) {
	b, senses, rec := recognitionBrain()
	require.Equal(t, "s1", present(b, senses, rec, 1.0, "f1", "f2"))
	t.Logf("after f1+f2: s1 weights f1 %.3f, f3 %.3f", slotSynapse(b, "f1", "s1").Weight, slotSynapse(b, "f3", "s1").Weight)

	winner := present(b, senses, rec, 1.0, "f3", "f4")
	t.Logf("after f3+f4: s1 f3 %.3f; s2 f3 %.3f, f1 %.3f", slotSynapse(b, "f3", "s1").Weight,
		slotSynapse(b, "f3", "s2").Weight, slotSynapse(b, "f1", "s2").Weight)
	assert.Equal(t, "s2", winner, "a novel bundle recruits an untrained slot")
	for _, f := range []string{"f3", "f4"} {
		assert.Less(t, slotSynapse(b, f, "s1").Weight, recognitionInit, "s1's link from absent %s stays depressed", f)
		assert.Greater(t, slotSynapse(b, f, "s2").Weight, recognitionInit, "s2 binds %s", f)
	}
	for _, f := range []string{"f1", "f2"} {
		assert.Less(t, slotSynapse(b, f, "s2").Weight, recognitionInit, "s2 unbinds %s", f)
	}
	assert.Equal(t, recognitionInit, slotSynapse(b, "f1", "s3").Weight, "s3 is still free")

	assert.Equal(t, "s1", present(b, senses, rec, 0.2, "f1", "f2"), "f1+f2 is still s1's")
}
