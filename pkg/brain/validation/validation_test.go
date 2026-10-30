package validation

import (
	"math"
	"testing"

	"github.com/mechanical-lich/brain-lib/pkg/brain"
	"github.com/stretchr/testify/assert"
)

func TestInterLobeSynapses(t *testing.T) {
	br := brain.NewBrain()

	l1 := brain.NewLobe("L1", nil)
	l2 := brain.NewLobe("L2", nil)

	in := &brain.Neuron{ID: "in", Bias: 1.0, Activation: brain.ActLinear}
	out := &brain.Neuron{ID: "out", Bias: 0.0, Activation: brain.ActLinear}
	out2 := &brain.Neuron{ID: "out2", Bias: 0.0, Activation: brain.ActLinear}

	l1.AddNeuron(in)
	l1.AddNeuron(out2)
	l2.AddNeuron(out)

	br.AddLobe(l1)
	br.AddLobe(l2)

	// Valid inter-lobe synapse
	br.Connect("L1", "in", "L2", "out", 0.5, false)

	// Invalid intra-lobe synapse added at brain level
	br.Synapses["L1_in->L1_out2"] = &brain.Synapse{
		ID:       "L1_in->L1_out2",
		FromLobe: "L1",
		From:     "in",
		ToLobe:   "L1",
		To:       "out2",
		Weight:   0.5,
	}

	complaints := ValidateBrain(br)
	assert.Len(t, complaints, 1)
	assert.Equal(t, Inter_Synapse_To_Same_Lobe, complaints[0].ComplaintType)
	assert.Equal(t, "L1_in->L1_out2", complaints[0].ID)
}

func TestMissingFromLobeAndNeuron(t *testing.T) {
	br := brain.NewBrain()

	l2 := brain.NewLobe("L2", nil)
	out := &brain.Neuron{ID: "out", Bias: 0.0, Activation: brain.ActLinear}
	l2.AddNeuron(out)
	br.AddLobe(l2)

	// Synapse references a from-lobe that doesn't exist
	br.Synapses["bad1"] = &brain.Synapse{
		ID:       "bad1",
		FromLobe: "L1",
		From:     "missing",
		ToLobe:   "L2",
		To:       "out",
		Weight:   0.1,
	}

	complaints := ValidateBrain(br)
	assert.Len(t, complaints, 1)
	assert.Equal(t, Missing_From_Lobe, complaints[0].ComplaintType)
}

func TestMissingToLobeAndNeuron(t *testing.T) {
	br := brain.NewBrain()

	l1 := brain.NewLobe("L1", nil)
	in := &brain.Neuron{ID: "in", Bias: 1.0, Activation: brain.ActLinear}
	l1.AddNeuron(in)
	br.AddLobe(l1)

	// Synapse references a to-lobe that doesn't exist
	br.Synapses["bad2"] = &brain.Synapse{
		ID:       "bad2",
		FromLobe: "L1",
		From:     "in",
		ToLobe:   "L2",
		To:       "out",
		Weight:   0.1,
	}

	complaints := ValidateBrain(br)
	assert.Len(t, complaints, 1)
	assert.Equal(t, Missing_To_Lobe, complaints[0].ComplaintType)
}

func TestLobeLevelSynapseProblems(t *testing.T) {
	br := brain.NewBrain()
	l1 := brain.NewLobe("L1", nil)
	l2 := brain.NewLobe("L2", nil)

	a := &brain.Neuron{ID: "a", Bias: 0.0, Activation: brain.ActLinear}
	b := &brain.Neuron{ID: "b", Bias: 0.0, Activation: brain.ActLinear}
	l1.AddNeuron(a)
	l2.AddNeuron(b)

	br.AddLobe(l1)
	br.AddLobe(l2)

	// Lobe-level synapse that points to a different lobe (invalid)
	l1.Synapses = map[string]*brain.Synapse{
		"s1": {
			ID:       "s1",
			FromLobe: "L1",
			From:     "a",
			ToLobe:   "L2",
			To:       "b",
			Weight:   0.2,
		},
	}

	// Lobe-level synapse that doesn't belong to this lobe
	l1.Synapses["s2"] = &brain.Synapse{
		ID:       "s2",
		FromLobe: "L3",
		From:     "x",
		ToLobe:   "L3",
		To:       "y",
		Weight:   0.3,
	}

	complaints := ValidateBrain(br)
	// Expect two complaints: one for intra-synapse pointing to different lobe, one for wrong lobe
	types := map[ComplaintType]bool{}
	for _, c := range complaints {
		types[c.ComplaintType] = true
	}

	assert.True(t, types[Intra_Synapse_To_Different_Lobe])
	assert.True(t, types[Wrong_Lobe_Synapse])
}

func TestMissingNeuronsInLobeSynapse(t *testing.T) {
	br := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	n := &brain.Neuron{ID: "n", Bias: 0.0, Activation: brain.ActLinear}
	l.AddNeuron(n)
	br.AddLobe(l)

	// Synapse missing 'from' neuron
	l.Synapses = map[string]*brain.Synapse{
		"s1": {
			ID:       "s1",
			FromLobe: "L",
			From:     "missing",
			ToLobe:   "L",
			To:       "n",
			Weight:   0.1,
		},
	}

	complaints := ValidateBrain(br)
	assert.Len(t, complaints, 1)
	assert.Equal(t, Missing_Neuron_From, complaints[0].ComplaintType)
}

// A per-synapse TraceDecay of 0 means "use the lobe's"; a negative one is
// silently treated the same way, which is never what the author meant.
func TestInvalidSynapseTraceDecay(t *testing.T) {
	br := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	l.AddNeuron(&brain.Neuron{ID: "a", Activation: brain.ActLinear})
	l.AddNeuron(&brain.Neuron{ID: "b", Activation: brain.ActLinear})
	l.AddNeuron(&brain.Neuron{ID: "c", Activation: brain.ActLinear})
	br.AddLobe(l)
	good := l.Connect("a", "b", 0.5, true)
	good.TraceDecay = 0.15
	bad := l.Connect("a", "c", 0.5, true)
	bad.TraceDecay = -0.15

	complaints := ValidateBrain(br)
	assert.Len(t, complaints, 1)
	assert.Equal(t, Invalid_Trace_Decay, complaints[0].ComplaintType)
	assert.Equal(t, bad.ID, complaints[0].ID)
}

// hebbianPairBrain wires a Hebbian synapse from an always-on neuron into one
// with a positive bias.
func hebbianPairBrain(rate, decay float64) (*brain.Brain, *brain.Synapse) {
	br := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	l.PlasticityConfig.HebbianRate = rate
	l.PlasticityConfig.HebbianDecay = decay
	l.AddNeuron(&brain.Neuron{ID: "a", Bias: 1.0, Activation: brain.ActLinear})
	l.AddNeuron(&brain.Neuron{ID: "b", Bias: 0.5, Activation: brain.ActLinear})
	syn := l.Connect("a", "b", 0.05, false)
	syn.Hebbian = true
	syn.MinWeight, syn.MaxWeight = 0, 1
	br.AddLobe(l)
	return br, syn
}

func complaintsOfType(complaints []Complaint, ct ComplaintType) []Complaint {
	var out []Complaint
	for _, c := range complaints {
		if c.ComplaintType == ct {
			out = append(out, c)
		}
	}
	return out
}

// (f) The instar rule is self-limiting: even into a biased neuron with no
// decay, the weight settles at the source's activation (1) rather than running
// to its bound. But with HebbianDecay 0 nothing ever unbinds it once the
// source goes quiet, and the validator says so. Warning only — repair leaves
// it alone. With decay the binding fades and nothing is reported.
func TestHebbianWithoutDecayNeverForgetsAndIsReported(t *testing.T) {
	step := func(br *brain.Brain, n int) {
		for i := 0; i < n; i++ {
			br.Step(1.0 / 60)
		}
	}
	// Both quiet: with the target still firing, instar would (rightly) unbind
	// the absent source; forgetting is about the pair falling silent.
	silence := func(br *brain.Brain) {
		br.Lobes["L"].Neurons["a"].Bias = 0
		br.Lobes["L"].Neurons["b"].Bias = 0
	}

	br, syn := hebbianPairBrain(1.0, 0)
	step(br, 600)
	assert.InDelta(t, 1.0, syn.Weight, 1e-3, "W settles at the source's activation, not beyond")
	silence(br)
	step(br, 600)
	assert.InDelta(t, 1.0, syn.Weight, 1e-3, "with no decay a silent pair keeps its binding forever")

	got := complaintsOfType(ValidateBrain(br), Hebbian_Without_Decay)
	if assert.Len(t, got, 1) {
		assert.Equal(t, syn.ID, got[0].ID)
	}
	report, _ := FixSynapses(br, ValidateBrain(br))
	assert.Contains(t, br.Lobes["L"].Synapses, syn.ID, "a warning must not delete the synapse")
	for _, r := range report.Repairs {
		assert.Equal(t, Skipped, r.Action)
	}

	br, syn = hebbianPairBrain(1.0, 0.5)
	step(br, 600)
	learned := syn.Weight
	silence(br)
	step(br, 600)
	assert.Less(t, syn.Weight, learned/2, "with decay the binding fades once the pair is quiet")
	assert.Empty(t, complaintsOfType(ValidateBrain(br), Hebbian_Without_Decay))
}

// The same check on an inter-lobe synapse, resolved against the target lobe's
// config the way Brain.applyInterLobePlasticity resolves it.
func TestHebbianWithoutDecayIsReportedInterLobe(t *testing.T) {
	br := brain.NewBrain()
	src := brain.NewLobe("src", nil)
	dst := brain.NewLobe("dst", nil)
	src.PlasticityConfig.HebbianDecay = 0.5 // the source's config must not matter
	src.AddNeuron(&brain.Neuron{ID: "a", Activation: brain.ActLinear})
	dst.AddNeuron(&brain.Neuron{ID: "b", Activation: brain.ActLinear})
	br.AddLobe(src)
	br.AddLobe(dst)
	syn := br.Connect("src", "a", "dst", "b", 0.1, false)
	syn.Hebbian = true

	got := complaintsOfType(ValidateBrain(br), Hebbian_Without_Decay)
	if assert.Len(t, got, 1) {
		assert.Equal(t, syn.ID, got[0].ID)
	}

	dst.PlasticityConfig.HebbianDecay = 0.1
	assert.Empty(t, complaintsOfType(ValidateBrain(br), Hebbian_Without_Decay))

	// Non-Hebbian synapses are never reported, whatever the config says.
	syn.Hebbian = false
	dst.PlasticityConfig.HebbianDecay = 0
	assert.Empty(t, complaintsOfType(ValidateBrain(br), Hebbian_Without_Decay))
}

func TestInvalidHebbianRates(t *testing.T) {
	br := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	l.PlasticityConfig.HebbianRate = -0.5
	br.AddLobe(l)
	br.PlasticityConfig.HebbianDecay = math.NaN()

	got := complaintsOfType(ValidateBrain(br), Invalid_Hebbian_Rate)
	ids := []string{}
	for _, c := range got {
		ids = append(ids, c.ID)
	}
	assert.ElementsMatch(t, []string{"L", "brain"}, ids)
}
