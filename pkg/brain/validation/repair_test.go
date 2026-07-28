package validation

import (
	"testing"

	"github.com/mechanical-lich/brain-lib/pkg/brain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func lobeWithConnect() (*brain.Brain, *brain.Lobe, *brain.Synapse) {
	b := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	l.AddNeuron(&brain.Neuron{ID: "a"})
	l.AddNeuron(&brain.Neuron{ID: "b"})
	syn := l.Connect("a", "b", 0.5, true)
	b.AddLobe(l)
	return b, l, syn
}

// Lobe.Connect is the ordinary way to make an intra-lobe synapse, and repair
// used to delete every one of them: Connect left FromLobe/ToLobe empty, and the
// ownership check compared "" against the lobe name on both sides, so each
// synapse was flagged Wrong_Lobe_Synapse and unconditionally deleted.
//
// This is the regression test for irreversible data loss on a real brain file.
func TestRepair_KeepsSynapsesMadeByLobeConnect(t *testing.T) {
	b, l, syn := lobeWithConnect()
	require.Len(t, l.Synapses, 1, "precondition: lobe has the connected synapse")

	complaints := ValidateBrain(b)
	_, err := FixSynapses(b, complaints)
	require.NoError(t, err)

	assert.Len(t, b.Lobes["L"].Synapses, 1,
		"repair deleted a synapse created by Lobe.Connect")
	if got, ok := b.Lobes["L"].Synapses[syn.ID]; ok {
		assert.Equal(t, 0.5, got.Weight, "surviving synapse should keep its weight")
	}
}

// Repair should also fill in the lobe fields it was previously complaining
// about, so a brain that has been through the validator is internally
// consistent rather than merely un-deleted.
func TestRepair_FillsInMissingLobeFields(t *testing.T) {
	b, _, syn := lobeWithConnect()

	_, err := FixSynapses(b, ValidateBrain(b))
	require.NoError(t, err)

	got := b.Lobes["L"].Synapses[syn.ID]
	require.NotNil(t, got, "synapse should still exist")
	assert.Equal(t, "L", got.FromLobe, "repair should attribute the synapse to its lobe")
	assert.Equal(t, "L", got.ToLobe)
}

// A brain that validates clean must stay clean through a repair round-trip.
func TestRepair_ConnectProducesNoComplaints(t *testing.T) {
	b, _, _ := lobeWithConnect()

	assert.Empty(t, ValidateBrain(b),
		"a lobe built with Connect should validate clean")
}

// Brain.AddSynapse routed to the lobe only when FromLobe == ToLobe and both
// were non-empty. A synapse with both empty — exactly what Lobe.Connect
// produced — landed at brain level, where Step skips it on FromLobe == "".
// It existed, transmitted nothing, and reported no error.
func TestAddSynapse_LobelessSynapseIsNotOrphaned(t *testing.T) {
	b := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	l.AddNeuron(&brain.Neuron{ID: "a"})
	l.AddNeuron(&brain.Neuron{ID: "b"})
	b.AddLobe(l)

	b.AddSynapse(&brain.Synapse{ID: "s1", From: "a", To: "b", Weight: 1.0})

	assert.NotContains(t, b.Synapses, "s1",
		"a synapse whose neurons both live in one lobe should not sit at brain level, where Step ignores it")
	require.Contains(t, b.Lobes["L"].Synapses, "s1",
		"it should have been routed into the lobe that owns both neurons")
	assert.Equal(t, "L", b.Lobes["L"].Synapses["s1"].FromLobe)
	assert.Equal(t, "L", b.Lobes["L"].Synapses["s1"].ToLobe)
}

// The same inference should reconstruct an inter-lobe synapse.
func TestAddSynapse_InfersLobesAcrossLobes(t *testing.T) {
	b := brain.NewBrain()
	src := brain.NewLobe("src", nil)
	src.AddNeuron(&brain.Neuron{ID: "a"})
	dst := brain.NewLobe("dst", nil)
	dst.AddNeuron(&brain.Neuron{ID: "b"})
	b.AddLobe(src)
	b.AddLobe(dst)

	b.AddSynapse(&brain.Synapse{ID: "s1", From: "a", To: "b", Weight: 1.0})

	require.Contains(t, b.Synapses, "s1", "an inter-lobe synapse belongs at brain level")
	assert.Equal(t, "src", b.Synapses["s1"].FromLobe)
	assert.Equal(t, "dst", b.Synapses["s1"].ToLobe)
}

// Where the neuron name is ambiguous there is no safe inference, so the synapse
// is left alone rather than wired to an arbitrary lobe — but the validator
// should say so rather than let it sit there silently inert.
func TestAddSynapse_AmbiguousNeuronIsReported(t *testing.T) {
	b := brain.NewBrain()
	for _, name := range []string{"L1", "L2"} {
		l := brain.NewLobe(name, nil)
		l.AddNeuron(&brain.Neuron{ID: "a"})
		l.AddNeuron(&brain.Neuron{ID: "b"})
		b.AddLobe(l)
	}

	b.AddSynapse(&brain.Synapse{ID: "s1", From: "a", To: "b", Weight: 1.0})

	complaints := ValidateBrain(b)
	found := false
	for _, c := range complaints {
		if c.ID == "s1" {
			found = true
		}
	}
	assert.True(t, found, "an unattributable synapse should be reported, not silently ignored")
}

// A synapse sitting in the wrong lobe's map should be moved to the lobe it
// names, not destroyed.
func TestRepair_MovesWrongLobeSynapseInsteadOfDeleting(t *testing.T) {
	b := brain.NewBrain()
	right := brain.NewLobe("right", nil)
	right.AddNeuron(&brain.Neuron{ID: "a"})
	right.AddNeuron(&brain.Neuron{ID: "b"})
	wrong := brain.NewLobe("wrong", nil)
	wrong.AddNeuron(&brain.Neuron{ID: "z"})
	b.AddLobe(right)
	b.AddLobe(wrong)

	// Filed under "wrong" but claiming to belong to "right".
	wrong.AddSynapse(&brain.Synapse{
		ID: "s1", From: "a", To: "b", FromLobe: "right", ToLobe: "right", Weight: 0.7,
	})

	_, err := FixSynapses(b, ValidateBrain(b))
	require.NoError(t, err)

	assert.NotContains(t, b.Lobes["wrong"].Synapses, "s1", "should have left the wrong lobe")
	require.Contains(t, b.Lobes["right"].Synapses, "s1",
		"repair destroyed a recoverable synapse instead of moving it")
	assert.Equal(t, 0.7, b.Lobes["right"].Synapses["s1"].Weight)
}

// A synapse targeting a neuroemitter is valid, but emitters do not live in
// lobe.Neurons — so it was reported as Missing_Neuron_To, whose repair deletes.
// Exactly the data-loss class this validator was fixed to stop, just one
// namespace over.
func TestRepair_KeepsNeuronToEmitterSynapses(t *testing.T) {
	b := brain.NewBrain()
	l := brain.NewLobe("L", nil)
	l.AddNeuron(&brain.Neuron{ID: "a"})
	l.AddEmitter("e1", "dopamine", 0.5, 0.0)
	syn := l.Connect("a", "e1", 0.75, true)
	b.AddLobe(l)

	assert.Empty(t, ValidateBrain(b), "a neuron->emitter synapse is valid wiring")

	_, err := FixSynapses(b, ValidateBrain(b))
	require.NoError(t, err)

	require.Contains(t, b.Lobes["L"].Synapses, syn.ID,
		"repair deleted a valid neuron->emitter synapse")
	assert.Equal(t, 0.75, b.Lobes["L"].Synapses[syn.ID].Weight)
}
