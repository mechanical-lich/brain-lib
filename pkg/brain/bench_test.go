package brain

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// realisticBrain builds a brain shaped like the one the game ships: 8 lobes,
// ~100 neurons, ~150 inter-lobe synapses plus intra-lobe wiring.
func realisticBrain() *Brain {
	b := NewBrain()
	lobeNames := []string{"sensory", "drive", "attention", "concept", "memory", "emotion", "focus", "decision"}

	for li, name := range lobeNames {
		l := NewLobe(name, nil)
		for n := 0; n < 12; n++ {
			l.AddNeuron(&Neuron{
				ID:         fmt.Sprintf("%s_n%d", name, n),
				Bias:       0.1,
				Activation: ActLinear,
			})
		}
		// Intra-lobe wiring.
		for n := 0; n < 11; n++ {
			l.Connect(fmt.Sprintf("%s_n%d", name, n), fmt.Sprintf("%s_n%d", name, n+1), 0.4, n%3 == 0)
		}
		l.Modulators[Dopamine] = &Modulator{Name: Dopamine, DecayRate: 0.5}
		if li >= 5 {
			l.WinnerTakesAll = true
		}
		b.AddLobe(l)
	}

	// Feed-forward inter-lobe wiring, with stable IDs.
	//
	// Deliberately not Brain.Connect: it mints a UUID per synapse, so two
	// structurally identical brains get different synapse IDs and cannot be
	// compared field by field.
	for i := 0; i+1 < len(lobeNames); i++ {
		from, to := lobeNames[i], lobeNames[i+1]
		for n := 0; n < 12; n++ {
			b.AddSynapse(&Synapse{
				ID:       fmt.Sprintf("%s->%s_%d", from, to, n),
				FromLobe: from, From: fmt.Sprintf("%s_n%d", from, n),
				ToLobe: to, To: fmt.Sprintf("%s_n%d", to, (n+1)%12),
				Weight: 0.3, Plastic: n%2 == 0,
				MinWeight: -2, MaxWeight: 2, InitWeight: 0.3,
			})
		}
	}

	b.Reinforce(Dopamine, 0.5)
	return b
}

func BenchmarkBrainStep(b *testing.B) {
	brain := realisticBrain()
	sensory := brain.Lobes["sensory"]

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sensory.SetNeuronInput("sensory_n0", 1.0)
		brain.Step(1.0 / 60)
	}
}

func BenchmarkBrainStepWithChemistry(b *testing.B) {
	brain := realisticBrain()
	chem := map[string]float64{"dopamine": 0.5, "serotonin": 0.2}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		brain.ApplyChemistry(chem, 1.0/60)
		brain.Step(1.0 / 60)
	}
}

func BenchmarkLobeStep(b *testing.B) {
	brain := realisticBrain()
	l := brain.Lobes["concept"]

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.Step(1.0/60, i, brain.Modulators)
	}
}

// Two identical brains driven identically must stay bit-for-bit identical.
//
// Step used to range Go maps to find synapses, so the order that float
// contributions accumulated into ExternalInput varied run to run. Within a
// cycle it also changed which neuron updated first, making the divergence
// semantic rather than just rounding.
func TestSteppingIsDeterministic(t *testing.T) {
	snapshot := func() string {
		br := realisticBrain()
		for i := 0; i < 200; i++ {
			br.Lobes["sensory"].SetNeuronInput("sensory_n0", 1.0)
			br.Step(1.0 / 60)
		}
		out, err := br.ToJSON()
		require.NoError(t, err)
		return string(out)
	}

	reference := snapshot()
	for i := 0; i < 8; i++ {
		assert.Equal(t, reference, snapshot(),
			"run %d diverged - stepping is not deterministic", i)
	}
}

// The update order itself must be stable, and must contain only real neurons.
func TestUpdateOrderIsStableAndPhantomFree(t *testing.T) {
	reference := realisticBrain()
	refOrder := append([]string(nil), reference.updateOrder...)

	for i := 0; i < 20; i++ {
		b := realisticBrain()
		assert.Equal(t, refOrder, b.updateOrder, "lobe order varies between builds")

		for name, l := range b.Lobes {
			assert.Len(t, l.updateOrder, len(l.Neurons),
				"lobe %s: updateOrder should contain exactly the real neurons", name)
			for _, id := range l.updateOrder {
				assert.Contains(t, l.Neurons, id, "lobe %s: phantom ID %q in updateOrder", name, id)
			}
		}
	}
}

// A synapse referencing a neuron that does not exist must not create a phantom
// entry in the update order.
func TestDanglingSynapseCreatesNoPhantomOrderEntry(t *testing.T) {
	l := NewLobe("L", nil)
	l.AddNeuron(&Neuron{ID: "real"})
	l.AddSynapse(&Synapse{ID: "s", From: "ghost", To: "real", FromLobe: "L", ToLobe: "L"})

	assert.Equal(t, []string{"real"}, l.updateOrder)
}

// A zero-weight synapse still counts as used when its source fires, so
// inactivity decay cannot pin it at zero forever.
func TestZeroWeightSynapseRegistersAsUsed(t *testing.T) {
	b := NewBrain()
	l := NewLobe("L", nil)
	l.AddNeuron(&Neuron{ID: "a", Bias: 1.0, Activation: ActLinear})
	l.AddNeuron(&Neuron{ID: "b", Activation: ActLinear})
	syn := l.Connect("a", "b", 0.0, true)
	b.AddLobe(l)

	b.Step(1.0 / 60)

	assert.NotZero(t, syn.LastUsed, "a synapse whose source fired should count as used even at weight 0")
	assert.NotZero(t, syn.UsageCount)
}

// RemoveEmitter deletes synapses, which invalidates the derived order.
func TestRemoveEmitterRefreshesOrder(t *testing.T) {
	l := NewLobe("L", nil)
	l.AddNeuron(&Neuron{ID: "a"})
	l.AddEmitter("e", "dopamine", 1.0, 0.0)
	l.Connect("a", "e", 0.5, false)
	require.Len(t, l.Synapses, 1)

	l.RemoveEmitter("e")

	assert.Empty(t, l.Synapses, "the emitter's synapses should be gone")
	assert.Equal(t, []string{"a"}, l.updateOrder, "the order should have been refreshed")
	assert.Empty(t, l.incoming["e"], "the incoming index should have been refreshed")
}
