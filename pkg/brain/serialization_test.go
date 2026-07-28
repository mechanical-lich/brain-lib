package brain

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fullyFeaturedBrain exercises every part of the format: lobes, neurons with
// every optional field, intra- and inter-lobe synapses, modulators, receptors of
// both target types, emitters, and plasticity configs at both levels.
func fullyFeaturedBrain() *Brain {
	b := NewBrain()
	b.PlasticityConfig = PlasticityConfig{
		TraceDecay: 2.5, RewardLearningRate: 7, PunishLearningRate: 9, InactivityDecay: 0.002,
	}
	b.Modulators[Dopamine] = &Modulator{Name: Dopamine, Level: 0.4, DecayRate: 0.6}
	b.Modulators[Serotonin] = &Modulator{Name: Serotonin, Level: 0.2, DecayRate: 0.3}
	b.AddReceptor("dopamine", 0.25, Dopamine, TargetTypeModulator, ReceptorTypeExcitatory)

	src := NewLobe("src", nil)
	src.X, src.Y, src.Width, src.Height = 1, 2, 3, 4
	src.Persists = true
	src.DecayRate = 0.35
	src.PlasticityConfig = PlasticityConfig{
		TraceDecay: 1.5, RewardLearningRate: 4, PunishLearningRate: 6, InactivityDecay: 0.001,
	}
	src.AddNeuron(&Neuron{
		ID: "a", Bias: 0.7, Activation: ActTanh, DecayRate: 0.2,
		Threshold: 0.05, Min: -1.5, Max: 1.5, Persists: true, GroupID: 3, X: 5, Y: 6,
	})
	src.AddNeuron(&Neuron{ID: "b", Activation: ActSigmoid, Capacitor: true, Hidden: true})
	src.Connect("a", "b", 0.45, true)
	src.Modulators[Dopamine] = &Modulator{Name: Dopamine, Level: 0.9, DecayRate: 0.44}
	src.AddReceptor("serotonin", 0.15, "a", TargetTypeNeuron, ReceptorTypeInhibitory)
	src.AddEmitter("e1", "dopamine", 0.8, 0.1)
	src.Neuroemitters["e1"].ReceiveInput(0.33) // pending charge

	dst := NewLobe("dst", nil)
	dst.WinnerTakesAll = true
	dst.AddNeuron(&Neuron{ID: "c", Bias: -0.2, Activation: ActLinear})
	b.AddLobe(src)
	b.AddLobe(dst)

	syn := b.Connect("src", "a", "dst", "c", 0.6, true)
	syn.Trace = 0.12
	syn.LastUsed = 7
	syn.UsageCount = 21
	b.Tick = 99
	return b
}

// Nothing tested Save/Load/ToJSON before, despite it being the path the game,
// the lab and every snapshot depend on.
func TestBrainRoundTripsThroughJSON(t *testing.T) {
	original := fullyFeaturedBrain()

	data, err := original.ToJSON()
	require.NoError(t, err)

	loaded, err := LoadFromJson(data)
	require.NoError(t, err)

	// Compare through a second serialization so unexported/derived state is
	// excluded and the comparison is on what the format actually carries.
	again, err := loaded.ToJSON()
	require.NoError(t, err)

	var a, c map[string]any
	require.NoError(t, json.Unmarshal(data, &a))
	require.NoError(t, json.Unmarshal(again, &c))
	assert.Equal(t, a, c, "a brain should survive a save/load round trip unchanged")
}

func TestRoundTripPreservesEveryComponent(t *testing.T) {
	data, err := fullyFeaturedBrain().ToJSON()
	require.NoError(t, err)
	b, err := LoadFromJson(data)
	require.NoError(t, err)

	assert.Equal(t, 99, b.Tick, "tick")
	assert.Equal(t, CurrentFormatVersion, b.FormatVersion)

	require.Contains(t, b.Modulators, Dopamine)
	assert.InDelta(t, 0.4, b.Modulators[Dopamine].Level, 1e-9, "brain modulator level")
	require.Len(t, b.ChemicalReceptors, 1)
	assert.Equal(t, TargetTypeModulator, b.ChemicalReceptors[0].TargetType, "brain receptor target type")

	src := b.Lobes["src"]
	require.NotNil(t, src)
	assert.True(t, src.Persists)
	assert.InDelta(t, 1.5, src.PlasticityConfig.TraceDecay, 1e-9, "lobe plasticity config")

	n := src.Neurons["a"]
	require.NotNil(t, n)
	assert.Equal(t, ActTanh, n.Activation)
	assert.InDelta(t, -1.5, n.Min, 1e-9)
	assert.InDelta(t, 1.5, n.Max, 1e-9)
	assert.Equal(t, 3, n.GroupID)
	assert.True(t, n.Persists)
	assert.True(t, src.Neurons["b"].Capacitor)

	require.Len(t, src.ChemicalReceptors, 1)
	assert.Equal(t, ReceptorTypeInhibitory, src.ChemicalReceptors[0].ReceptorType, "lobe receptor type")
	assert.Equal(t, "serotonin", src.ChemicalReceptors[0].ChemicalName)

	e := src.GetEmitter("e1")
	require.NotNil(t, e)
	assert.InDelta(t, 0.8, e.EmissionRate, 1e-9)
	assert.InDelta(t, 0.33, e.AccumulatedActivation, 1e-9,
		"a brain saved mid-tick should not lose an emitter's pending charge")

	assert.True(t, b.Lobes["dst"].WinnerTakesAll)

	var inter *Synapse
	for _, s := range b.Synapses {
		inter = s
	}
	require.NotNil(t, inter)
	assert.InDelta(t, 0.12, inter.Trace, 1e-9, "trace")
	assert.Equal(t, 21, inter.UsageCount, "usage count")
}

// Receptors used to serialize in PascalCase while the rest of the format was
// snake_case. Existing brain and lab snapshot files still use it, so both forms
// must load.
func TestLegacyPascalCaseReceptorsStillLoad(t *testing.T) {
	raw := []byte(`{
		"format_version": 0,
		"lobes": {"L": {"name":"L","neurons":{"n":{"id":"n"}},"synapses":{},
			"modulators":{"dopamine":{"name":"dopamine","level":0,"decay_rate":0.5}},
			"chemical_receptors":[
				{"ChemicalName":"dopamine","UptakeRate":0.2,"ReceptorType":"inhibitory",
				 "Target":"dopamine","TargetType":"modulator","Threshold":0.1,
				 "OutputNormalizationMin":0,"OutputNormalizationMax":0}],
			"plasticity_config":{"trace_decay":1}}},
		"synapses":{}, "modulators":{}, "plasticity_config":{"trace_decay":1}
	}`)

	b, err := LoadFromJson(raw)
	require.NoError(t, err)

	rs := b.Lobes["L"].ChemicalReceptors
	require.Len(t, rs, 1)
	assert.Equal(t, "dopamine", rs[0].ChemicalName, "legacy ChemicalName should be read")
	assert.InDelta(t, 0.2, rs[0].UptakeRate, 1e-9)
	assert.Equal(t, ReceptorTypeInhibitory, rs[0].ReceptorType)
	assert.Equal(t, TargetTypeModulator, rs[0].TargetType)
	assert.InDelta(t, 0.1, rs[0].Threshold, 1e-9)

	// And it is rewritten in the current form.
	out, err := b.ToJSON()
	require.NoError(t, err)
	assert.Contains(t, string(out), `"chemical_name"`)
	assert.NotContains(t, string(out), `"ChemicalName"`)
}

func TestSaveAndLoadFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "brain.json")
	original := fullyFeaturedBrain()

	require.NoError(t, original.Save(path))
	loaded, err := LoadBrain(path)
	require.NoError(t, err)

	assert.Equal(t, len(original.Lobes), len(loaded.Lobes))
	assert.Equal(t, original.Tick, loaded.Tick)

	// The file itself should carry the version.
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"format_version": 1`)
}

// A file from a future build must be refused rather than silently misread.
func TestFutureFormatVersionIsRejected(t *testing.T) {
	raw := []byte(`{"format_version": 9999, "lobes":{}, "synapses":{}, "modulators":{},
		"plasticity_config":{"trace_decay":1}}`)

	_, err := LoadFromJson(raw)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "newer than this build supports")
}

// Loading a brain must not perturb it: two loads of the same bytes are equal,
// and a load/save cycle is idempotent.
func TestLoadIsIdempotent(t *testing.T) {
	data, err := fullyFeaturedBrain().ToJSON()
	require.NoError(t, err)

	first, err := LoadFromJson(data)
	require.NoError(t, err)
	out1, err := first.ToJSON()
	require.NoError(t, err)

	second, err := LoadFromJson(out1)
	require.NoError(t, err)
	out2, err := second.ToJSON()
	require.NoError(t, err)

	assert.JSONEq(t, string(out1), string(out2))
}
