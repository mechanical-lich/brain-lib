package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUpdateLobeOrder(t *testing.T) {
	b := NewBrain()

	// Create lobes A, B, C, D
	a := NewLobe("A", nil)
	bL := NewLobe("B", nil)
	c := NewLobe("C", nil)
	d := NewLobe("D", nil)

	b.AddLobe(a)
	b.AddLobe(bL)
	b.AddLobe(c)
	b.AddLobe(d)

	// Connect A->B, A->C, B->D, C->D across lobes (using neurons names arbitrary)
	b.Connect("A", "n1", "B", "n2", 1.0, false)
	b.Connect("A", "n1", "C", "n3", 1.0, false)
	b.Connect("B", "n2", "D", "n4", 1.0, false)
	b.Connect("C", "n3", "D", "n4", 1.0, false)

	order := b.updateOrder
	assert.NotEmpty(t, order)

	pos := func(name string) int {
		for i, v := range order {
			if v == name {
				return i
			}
		}
		return -1
	}

	assert.True(t, pos("A") >= 0)
	assert.True(t, pos("B") >= 0)
	assert.True(t, pos("C") >= 0)
	assert.True(t, pos("D") >= 0)

	assert.True(t, pos("A") < pos("B"), "A should come before B")
	assert.True(t, pos("A") < pos("C"), "A should come before C")
	assert.True(t, pos("B") < pos("D"), "B should come before D")
	assert.True(t, pos("C") < pos("D"), "C should come before D")
}

// TestBrainPlasticityInterLobeReward tests that inter-lobe synapses strengthen with dopamine reward
func TestBrainPlasticityInterLobeReward(t *testing.T) {
	b := NewBrain()

	// Create two lobes with neurons
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	neuronA := &Neuron{ID: "n1", Bias: 1.0, Activation: ActLinear}
	neuronB := &Neuron{ID: "n2", Bias: 0.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create plastic inter-lobe connection
	syn := b.Connect("A", "n1", "B", "n2", 0.2, true)
	initialWeight := syn.Weight

	// Run a few steps to build eligibility trace
	b.Step(0.1)
	b.Step(0.1)
	b.Step(0.1)

	// Apply dopamine reward via brain-level modulator
	b.Reinforce(Dopamine, 1.0)

	// Step again to apply plasticity
	b.Step(0.1)

	// Verify synapse strengthened
	finalWeight := b.Synapses[syn.ID].Weight
	assert.Greater(t, finalWeight, initialWeight, "Inter-lobe synapse should strengthen with dopamine reward")
}

// TestBrainPlasticityInterLobePunishment tests that inter-lobe synapses weaken with serotonin punishment
func TestBrainPlasticityInterLobePunishment(t *testing.T) {
	b := NewBrain()

	// Create two lobes with active neurons
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	neuronA := &Neuron{ID: "n1", Bias: 1.0, Activation: ActLinear}
	neuronB := &Neuron{ID: "n2", Bias: 0.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create plastic inter-lobe connection with moderate weight
	syn := b.Connect("A", "n1", "B", "n2", 0.5, true)
	initialWeight := syn.Weight

	// Run steps to build eligibility trace
	b.Step(0.1)
	b.Step(0.1)
	b.Step(0.1)

	// Apply serotonin punishment
	b.Reinforce(Serotonin, 1.0)

	// Step to apply plasticity
	b.Step(0.1)

	// Verify synapse weakened
	finalWeight := b.Synapses[syn.ID].Weight
	assert.Less(t, finalWeight, initialWeight, "Inter-lobe synapse should weaken with serotonin punishment")
}

// TestBrainPlasticityChemicalReceptors tests that chemical receptors can modulate brain-level dopamine
func TestBrainPlasticityChemicalReceptors(t *testing.T) {
	b := NewBrain()

	// Create lobes and neurons
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	// Give neurons bias so they're active
	neuronA := &Neuron{ID: "n1", Bias: 2.0, Activation: ActLinear}
	neuronB := &Neuron{ID: "n2", Bias: 1.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create plastic synapse
	syn := b.Connect("A", "n1", "B", "n2", 0.2, true)
	initialWeight := syn.Weight

	// Initialize dopamine modulator first
	b.Reinforce(Dopamine, 0.0)

	// Add brain-level receptor: "reward_chemical" -> dopamine modulator
	b.AddReceptor("reward_chemical", 1.0, Dopamine, TargetTypeModulator, ReceptorTypeExcitatory)

	// Build eligibility trace with active neurons
	b.Step(0.1)
	b.Step(0.1)
	b.Step(0.1)

	// Apply chemicals through the chemical system
	chemicals := map[string]float64{
		"reward_chemical": 2.0,
	}
	b.ApplyChemistry(chemicals, 0.1)

	// Verify dopamine was increased
	assert.Greater(t, b.Modulators[Dopamine].Level, 0.0, "Dopamine should be increased by chemical receptor")

	// Step to apply plasticity with dopamine from chemical receptor
	b.Step(0.1)
	b.Step(0.1)

	// Verify synapse strengthened due to dopamine from chemical
	finalWeight := b.Synapses[syn.ID].Weight
	assert.Greater(t, finalWeight, initialWeight, "Synapse should strengthen when chemical receptor increases dopamine")
}

// TestBrainPlasticityInhibitoryReceptor tests that inhibitory receptors can suppress learning signals
func TestBrainPlasticityInhibitoryReceptor(t *testing.T) {
	b := NewBrain()

	// Create lobes
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	neuronA := &Neuron{ID: "n1", Bias: 1.0, Activation: ActLinear}
	neuronB := &Neuron{ID: "n2", Bias: 0.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create plastic synapse
	syn := b.Connect("A", "n1", "B", "n2", 0.5, true)

	// Set initial dopamine level high
	b.Reinforce(Dopamine, 2.0)

	// Add inhibitory receptor that reduces dopamine when "stress" is present
	b.AddReceptor("stress", 1.0, Dopamine, TargetTypeModulator, ReceptorTypeInhibitory)

	// Build trace
	b.Step(0.1)

	// Apply stress chemical to inhibit dopamine
	chemicals := map[string]float64{
		"stress": 2.0,
	}
	b.ApplyChemistry(chemicals, 0.1)

	initialWeight := syn.Weight

	// Step again - should have less learning due to reduced dopamine
	b.Step(0.1)

	finalWeight := b.Synapses[syn.ID].Weight

	// Weight change should be modest since dopamine was inhibited
	weightChange := finalWeight - initialWeight
	assert.True(t, weightChange < 0.2, "Inhibitory receptor should reduce learning signal")
}

// TestBrainPlasticityHierarchicalModulators tests lobe-level modulators override brain-level
func TestBrainPlasticityHierarchicalModulators(t *testing.T) {
	b := NewBrain()

	// Create two lobes
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	neuronA := &Neuron{ID: "n1", Bias: 1.0, Activation: ActLinear}
	neuronB := &Neuron{ID: "n2", Bias: 0.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create plastic synapse
	syn := b.Connect("A", "n1", "B", "n2", 0.3, true)
	initialWeight := syn.Weight

	// Set brain-level dopamine low
	b.Reinforce(Dopamine, 0.1)

	// Set lobe B's dopamine high (should override brain-level)
	lobeB.Reinforce(Dopamine, 2.0)

	// Build trace
	b.Step(0.1)
	b.Step(0.1)

	// Step to apply plasticity
	b.Step(0.1)

	// Synapse should strengthen significantly due to lobe-level dopamine
	finalWeight := b.Synapses[syn.ID].Weight
	assert.Greater(t, finalWeight, initialWeight+0.05, "Lobe-level modulator should override brain-level")
}

// TestBrainPlasticityInactivityDecay tests that inactive synapses decay toward baseline
func TestBrainPlasticityInactivityDecay(t *testing.T) {
	b := NewBrain()

	// Create lobes with inactive neurons
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	neuronA := &Neuron{ID: "n1", Bias: 0.0, Activation: ActLinear} // Inactive
	neuronB := &Neuron{ID: "n2", Bias: 0.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create synapse with weight different from its baseline
	syn := b.Connect("A", "n1", "B", "n2", 0.5, true)
	syn.InitWeight = 0.3 // Baseline is 0.3, current is 0.5

	initialWeight := syn.Weight

	// Run many steps without activity (no reinforcement, neurons inactive)
	for i := 0; i < 100; i++ {
		b.Step(0.1)
	}

	// Weight should decay toward baseline (0.3)
	finalWeight := b.Synapses[syn.ID].Weight
	assert.Less(t, finalWeight, initialWeight, "Weight should decay toward baseline")
	assert.Greater(t, finalWeight, 0.29, "Weight should approach baseline of 0.3")
}

// TestBrainPlasticityMultipleChemicals tests multiple chemical receptors working together
func TestBrainPlasticityMultipleChemicals(t *testing.T) {
	b := NewBrain()

	// Create lobes
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	// Active neurons
	neuronA := &Neuron{ID: "n1", Bias: 2.0, Activation: ActLinear}
	neuronB := &Neuron{ID: "n2", Bias: 1.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create synapse
	syn := b.Connect("A", "n1", "B", "n2", 0.3, true)
	initialWeight := syn.Weight

	// Initialize modulators first
	b.Reinforce(Dopamine, 0.0)
	b.Reinforce(Serotonin, 0.0)

	// Add multiple receptors
	b.AddReceptor("food", 0.5, Dopamine, TargetTypeModulator, ReceptorTypeExcitatory)
	b.AddReceptor("water", 0.3, Dopamine, TargetTypeModulator, ReceptorTypeExcitatory)
	b.AddReceptor("pain", 0.8, Serotonin, TargetTypeModulator, ReceptorTypeExcitatory)

	// Build trace with active neurons
	b.Step(0.1)
	b.Step(0.1)
	b.Step(0.1)

	// Apply mixed chemicals: some reward, some punishment
	chemicals := map[string]float64{
		"food":  1.0, // Increases dopamine
		"water": 0.5, // Increases dopamine
		"pain":  0.3, // Increases serotonin
	}
	b.ApplyChemistry(chemicals, 0.1)

	// Verify modulators were affected
	dopamineLevel := b.Modulators[Dopamine].Level
	serotoninLevel := b.Modulators[Serotonin].Level
	t.Logf("Dopamine: %f, Serotonin: %f", dopamineLevel, serotoninLevel)

	// Step to apply plasticity
	b.Step(0.1)
	b.Step(0.1)

	finalWeight := b.Synapses[syn.ID].Weight

	// Should have net increase since reward chemicals outweigh punishment
	assert.Greater(t, finalWeight, initialWeight, "Mixed chemicals with more reward should strengthen synapse")
}

// TestBrainPlasticityConfigFallback tests that brain plasticity config is used when lobe doesn't have one
func TestBrainPlasticityConfigFallback(t *testing.T) {
	b := NewBrain()

	// Set brain-level plasticity config with high learning rate
	b.PlasticityConfig.RewardLearningRate = 0.5

	// Create lobes without custom config (they'll use defaults which have TraceDecay > 0)
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	// Active neurons
	neuronA := &Neuron{ID: "n1", Bias: 3.0, Activation: ActLinear}
	neuronB := &Neuron{ID: "n2", Bias: 2.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create synapse
	syn := b.Connect("A", "n1", "B", "n2", 0.1, true)
	initialWeight := syn.Weight

	// Build strong trace with very active neurons
	b.Step(0.1)
	b.Step(0.1)
	b.Step(0.1)

	// Apply strong reward
	b.Reinforce(Dopamine, 2.0)

	b.Step(0.1)
	b.Step(0.1)

	// Should use brain's high learning rate
	finalWeight := b.Synapses[syn.ID].Weight
	weightChange := finalWeight - initialWeight

	t.Logf("Initial: %f, Final: %f, Change: %f", initialWeight, finalWeight, weightChange)
	assert.Greater(t, weightChange, 0.01, "Should use brain-level plasticity config with high learning rate")
}

// TestBrainPlasticityEligibilityTrace tests that eligibility traces decay properly
func TestBrainPlasticityEligibilityTrace(t *testing.T) {
	b := NewBrain()

	// Create lobes
	lobeA := NewLobe("A", nil)
	lobeB := NewLobe("B", nil)

	neuronA := &Neuron{ID: "n1", Bias: 1.0, Activation: ActLinear}
	neuronB := &Neuron{ID: "n2", Bias: 0.0, Activation: ActLinear}

	lobeA.AddNeuron(neuronA)
	lobeB.AddNeuron(neuronB)

	b.AddLobe(lobeA)
	b.AddLobe(lobeB)

	// Create synapse
	syn := b.Connect("A", "n1", "B", "n2", 0.3, true)

	// Build trace
	b.Step(0.1)
	traceAfterOneStep := b.Synapses[syn.ID].Trace
	assert.Greater(t, traceAfterOneStep, 0.0, "Trace should be positive after activity")

	// Let trace decay without new activity (neurons inactive)
	neuronA.Bias = 0.0
	for i := 0; i < 10; i++ {
		b.Step(0.1)
	}

	traceAfterDecay := b.Synapses[syn.ID].Trace
	assert.Less(t, traceAfterDecay, traceAfterOneStep, "Trace should decay over time without activity")
}
