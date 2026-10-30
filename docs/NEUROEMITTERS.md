# Neuroemitters

## Overview
Neuroemitters are structures in lobes that release chemicals based on neural activity. Unlike receptors which consume chemicals to affect neurons, emitters produce chemicals when neurons reach certain activation thresholds.

## Key Features
- **Lobe-level only**: Neuroemitters exist only in lobes, not at the brain level
- **Synapse-driven**: Connected to neurons via synapses, accumulating activation from multiple sources
- **Threshold-based**: Only emit when accumulated activation meets or exceeds a threshold
- **Chemical production**: Add chemicals to the chemistry map that can be consumed by receptors

## Structure
```go
type Neuroemitter struct {
    ID           string  // Unique identifier
    ChemicalName string  // Name of chemical to release
    EmissionRate float64 // Rate of emission when activated
    Threshold    float64 // Activation threshold for emission
}
```

## How It Works

1. **Input Accumulation**: During each step, neurons send activation to neuroemitters through synapses
   - `activation_received = neuron_activation * synapse_weight`
   - Multiple neurons can target the same emitter, accumulating their contributions

2. **Threshold Check**: When `UpdateNeuroemitters` is called, each emitter checks if accumulated activation >= threshold

3. **Emission**: If threshold is met:
   - `emission = accumulated_activation * emission_rate * dt`
   - Chemical is added to the chemistry map

4. **Reset**: Accumulated activation is cleared after each step

## API

### Lobe Methods
```go
// Add a neuroemitter to the lobe
lobe.AddEmitter(id, chemicalName string, emissionRate, threshold float64)

// Remove a neuroemitter (also removes connected synapses)
lobe.RemoveEmitter(id string)

// Get a neuroemitter by ID
emitter := lobe.GetEmitter(id string)

// Process emitters and add chemicals to chemistry map
chemicals = lobe.UpdateNeuroemitters(dt float64, chemicals map[string]float64)
```

## Usage Example
```go
// Create a lobe
lobe := brain.NewLobe("reward", nil)

// Add neurons
n1 := &brain.Neuron{ID: "n1", ActivationVal: 0.8}
n2 := &brain.Neuron{ID: "n2", ActivationVal: 0.6}
lobe.AddNeuron(n1)
lobe.AddNeuron(n2)

// Add a dopamine emitter with threshold 0.5
lobe.AddEmitter("dopa_emit", "dopamine", 0.5, 0.5)

// Connect neurons to emitter
syn1 := &brain.Synapse{
    ID: "n1_to_emit", 
    From: "n1", 
    To: "dopa_emit", 
    Weight: 0.5,
}
syn2 := &brain.Synapse{
    ID: "n2_to_emit", 
    From: "n2", 
    To: "dopa_emit", 
    Weight: 0.5,
}
lobe.AddSynapse(syn1)
lobe.AddSynapse(syn2)

// Step the lobe (neurons activate and send to emitter)
lobe.Step(0.1, 0)

// Process emitters to produce chemicals
chemicals := make(map[string]float64)
chemicals = lobe.UpdateNeuroemitters(0.1, chemicals)

// chemicals["dopamine"] now contains the emitted dopamine
// (0.8 * 0.5 + 0.6 * 0.5) * 0.5 * 0.1 = 0.035
```

## Integration with Brain System

### Typical Update Sequence
1. `lobe.Step(dt, now)` - Updates neurons and sends activation to emitters
2. `lobe.UpdateNeuroemitters(dt, chemistry)` - Emitters produce chemicals
3. `lobe.UpdateChemicalReceptors(dt, chemistry)` - Receptors consume chemicals to affect neurons

### Receptors: output vs consumption
A receptor with `output_normalization_min`/`max` set maps the concentration
onto 0..1 and delivers it as a **level** to a neuron target (its
`ExternalInput` is consumed every step) or as a **rate** per second, scaled by
`dt`, to a modulator target (which accumulates and decays on its own).
Without normalization it delivers what it takes up. Either way consumption is
always `concentration * uptake_rate * dt`, clamped to the pool — so a
normalized receptor can read a pool strongly while draining it slowly.

### Synaptic Connections
- Emitters can receive input from any neuron in the lobe via synapses
- The `To` field of the synapse should match the emitter's ID
- Synapses targeting emitters work identically to neuron-to-neuron synapses

## Design Rationale

1. **Separation of Concerns**: Emitters produce chemicals, receptors consume them
2. **Flexible Topology**: Multiple neurons can contribute to one emitter, or multiple emitters can produce the same chemical
3. **Threshold Control**: Prevents noise from triggering chemical release
4. **Time-scaled Emission**: Using `dt` ensures consistent behavior across different time steps

## Future Enhancements
- Decay rates for accumulated activation (currently resets each step)
- Maximum emission caps
- Refractory periods after emission
- UI integration in brain-lab for visual editing
