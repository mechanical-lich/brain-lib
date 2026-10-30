# Plasticity Guide

> **Units note (2026-07-25):** every rate in `PlasticityConfig`, and every
> `decay_rate`, is **per second**. They used to be per simulation step, which
> made learning depend on frame rate. Decay is exponential (`exp(-rate*dt)`),
> so a rate is a true time constant at any step size. Files written before this
> carry `format_version` 0; see `CurrentFormatVersion` in `pkg/brain/brain.go`.


This guide explains how to use the reinforcement learning system in your Creatures-style artificial life simulation.

## Overview

The brain now supports **multi-chemical reinforcement learning** with:
- **Dopamine**: Positive reinforcement (rewards)
- **Serotonin**: Negative reinforcement (punishment)
- **Endorphin**: Wellbeing/satisfaction signals
- **Acetylcholine**: Attention/focus (for future use)

## Key Concepts

### 1. Eligibility Traces
Synapses maintain an "eligibility trace" that records recent correlations between pre- and post-synaptic activity. When a reward/punishment arrives, it affects synapses proportional to their trace value. This solves the **temporal credit assignment problem** - actions that happened a few steps ago can still be reinforced.

### 2. Learning Rates
- `RewardLearningRate` (default: 0.10): How fast to strengthen connections on reward
- `PunishLearningRate` (default: 0.15): How fast to weaken connections on punishment
- Punishment learning is slightly faster by default (animals learn faster from pain)

### 3. Inactivity Decay
Unused synapses slowly decay back toward their initial weight. This prevents:
- Runaway weight growth
- Irrelevant learned behaviors persisting forever
- Enables re-learning in new contexts

A synapse is "unused" only while its source neuron's activation is exactly 0.
A persisting or capacitor neuron with a `threshold` reaches exactly 0 once its
carried activation decays below that threshold, so the links a faded memory
feeds start to decay; with `threshold: 0` it fades asymptotically and its links
stay in use.

## Integration with Your Game

### Basic Setup

```go
// Create a brain with decision lobe
brain := brain.NewBrain()
decisionLobe := brain.NewLobe("decision", nil)

// Add neurons for different actions
decisionLobe.AddNeuron(&brain.Neuron{ID: "approach", Bias: 0.0})
decisionLobe.AddNeuron(&brain.Neuron{ID: "flee", Bias: 0.0})
decisionLobe.AddNeuron(&brain.Neuron{ID: "eat", Bias: 0.0})

// Add sensory lobe
sensoryLobe := brain.NewLobe("sensory", nil)
sensoryLobe.AddNeuron(&brain.Neuron{ID: "see_food", Bias: 0.0})
sensoryLobe.AddNeuron(&brain.Neuron{ID: "see_predator", Bias: 0.0})

// Connect with plastic synapses
brain.Connect("sensory", "see_food", "decision", "approach", 0.1, true)
brain.Connect("sensory", "see_predator", "decision", "flee", 0.1, true)
```

### Game Event Integration

#### Example 1: Critter Eats Food (Positive Reinforcement)

```go
func (critter *Critter) Eat(food *Food) {
    // Apply game effects
    critter.Hunger -= food.Nutrition
    critter.Energy += food.Energy
    
    // Positive reinforcement - release dopamine
    // The amount should scale with how good the outcome was
    rewardAmount := food.Nutrition * 0.5 // Scale to 0-1 range
    
    // Apply to decision lobe (where action selection happened)
    critter.Brain.GetLobe("decision").Reinforce(brain.Dopamine, rewardAmount)
    
    // Optionally also reward sensory pathways
    critter.Brain.GetLobe("sensory").Reinforce(brain.Dopamine, rewardAmount * 0.3)
}
```

#### Example 2: Critter Gets Hurt (Negative Reinforcement)

```go
func (critter *Critter) TakeDamage(amount float64, source DamageSource) {
    // Apply game effects
    critter.Health -= amount
    
    // Negative reinforcement - release serotonin (pain signal)
    // Scale punishment by damage severity
    punishAmount := amount * 0.1 // Scale appropriately
    
    // Punish the decision that led to this
    critter.Brain.GetLobe("decision").Reinforce(brain.Serotonin, punishAmount)
    
    // The currently active pathways will be weakened
}
```

#### Example 3: Critter Successfully Escapes Danger

```go
func (critter *Critter) EscapeDanger() {
    // Mixed signals - some punishment (was in danger) and some reward (escaped)
    
    // Small punishment for being in danger
    critter.Brain.GetLobe("decision").Reinforce(brain.Serotonin, 0.2)
    
    // Larger reward for successful escape
    critter.Brain.GetLobe("decision").Reinforce(brain.Dopamine, 0.8)
    
    // Net effect: pathway that led to escape is reinforced
}
```

#### Example 4: Complex Learning Scenario

```go
func (game *Game) UpdateCritter(critter *Critter, dt float64) {
    // Step the brain (updates neurons, eligibility traces, applies plasticity)
    critter.Brain.Step(dt)
    
    // Check various conditions and apply reinforcement
    
    // Hunger is very high - negative signal
    if critter.Hunger > 80 {
        critter.Brain.GetLobe("drive").Reinforce(brain.Serotonin, 0.1)
    }
    
    // Found a toy while bored - positive signal
    if critter.Boredom > 50 && critter.IsPlayingWithToy() {
        critter.Brain.GetLobe("decision").Reinforce(brain.Dopamine, 0.5)
        critter.Brain.GetLobe("decision").Reinforce(brain.Endorphin, 0.3)
        critter.Boredom -= 10
    }
    
    // Social interaction - reward
    if critter.IsNearFriend() {
        critter.Brain.GetLobe("social").Reinforce(brain.Dopamine, 0.3)
    }
}
```

## Learning Scenarios

### Scenario 1: Learning to Approach Food

**Timeline:**
1. Critter sees food → `see_food` neuron activates
2. Critter is hungry → `hunger` drive activates  
3. `approach` decision neuron fires (maybe by chance initially)
4. Critter moves toward food
5. **Eligibility traces build** in pathways: `see_food→approach`, `hunger→approach`
6. Critter reaches and eats food → **call `Reinforce(Dopamine, 1.0)`**
7. Traces still present → pathways strengthened
8. **Next time**: food + hunger → more likely to trigger approach

### Scenario 2: Learning to Avoid Danger

**Timeline:**
1. Critter sees predator → `see_predator` activates
2. Curiosity drive high → `investigate` neuron activates
3. Critter approaches predator (bad idea!)
4. Gets hurt → **call `Reinforce(Serotonin, 0.8)`**
5. `see_predator→investigate` pathway **weakened**
6. If critter flees and survives → **call `Reinforce(Dopamine, 0.3)`**
7. `see_predator→flee` pathway **strengthened**
8. **Next time**: predator → flee (not investigate)

### Scenario 3: Extinguishing Learned Behavior

If food becomes unavailable or the environment changes:
1. Critter approaches food location (learned behavior)
2. No food found → no reward
3. Eligibility traces decay naturally each step
4. **Inactivity decay** slowly weakens unused pathways
5. After many unrewarded attempts, behavior extinguishes
6. Critter can learn new food locations

## Tuning Parameters

### Per-Lobe Configuration

```go
// Create lobe with custom plasticity settings
lobe := brain.NewLobe("decision", nil)

// Fast learner (young critter)
lobe.PlasticityConfig.RewardLearningRate = 0.20
lobe.PlasticityConfig.PunishLearningRate = 0.25
lobe.PlasticityConfig.TraceDecay = 0.85  // Faster decay

// Slow learner (old critter)
lobe.PlasticityConfig.RewardLearningRate = 0.05
lobe.PlasticityConfig.PunishLearningRate = 0.08
lobe.PlasticityConfig.TraceDecay = 0.95  // Slower decay

// Stubborn/stable (hard to unlearn)
lobe.PlasticityConfig.InactivityDecay = 0.0001  // Very slow forgetting
```

### Chemical Decay Rates

```go
// Fast reward decay (joy is fleeting)
dopamine := &brain.Modulator{
    Name:      brain.Dopamine,
    Level:     0.0,
    DecayRate: 0.8,  // Decays 80% per step
}

// Slow punishment decay (pain lingers)
serotonin := &brain.Modulator{
    Name:      brain.Serotonin,
    Level:     0.0,
    DecayRate: 0.2,  // Decays 20% per step
}

lobe.Modulators[brain.Dopamine] = dopamine
lobe.Modulators[brain.Serotonin] = serotonin
```

### Per-Synapse Settings

```go
synapse := lobe.Connect("input", "output", 0.5, true)

// Give this synapse alone a wider reinforcement window than its lobe.
synapse.TraceDecay = 0.15 // per second; 0 = use the lobe's
// NOTE: MinDecayFrac is serialized but never read - decay has no floor.
```

`Synapse.TraceDecay` (JSON `trace_decay`) overrides the eligibility-trace decay
rate for one synapse. Left at 0 it falls back to the lobe's
`PlasticityConfig.TraceDecay` — the owning lobe's for an intra-lobe synapse,
the *target* lobe's for an inter-lobe one. `TraceDecay` is otherwise a lobe-wide
setting, so widening the window in the lobe config widens it for every plastic
synapse into that lobe, including ones you never meant to be reward-sensitive;
setting it per synapse keeps the wide window on just the pathways you want
teachable. Learning rates and inactivity decay still come from the lobe. The
field is omitted from JSON when zero, so older files load unchanged, and the
validator reports a negative value as `Invalid_Trace_Decay`.

## Recommended Starting Values

For a Creatures-style game, start with these values:

```go
// Lobe plasticity config (already defaults)
TraceDecay:         0.90   // 10% decay per step - adjust based on your timestep
RewardLearningRate: 0.10   // Moderate reward learning
PunishLearningRate: 0.15   // Slightly faster punishment learning
InactivityDecay:    0.00001 // Very slow forgetting (matches the library default)

// Modulator decay rates
Dopamine:    0.5   // Medium-fast (reward fades fairly quickly)
Serotonin:   0.3   // Slower (pain lingers)
Endorphin:   0.8   // Fast (satisfaction is brief)

// Synapse settings
// MinDecayFrac: serialized but not implemented; do not rely on it.
MinWeight:   -2.0  // Allow strong inhibition
MaxWeight:    2.0  // Allow strong excitation
```

## Debugging Tips

### 1. Add Logging

```go
// In your game update loop
if debug {
    for _, lobe := range critter.Brain.Lobes {
        if d, ok := lobe.Modulators[brain.Dopamine]; ok && d.Level > 0.01 {
            log.Printf("Lobe %s dopamine: %.3f", lobe.Name, d.Level)
        }
        if s, ok := lobe.Modulators[brain.Serotonin]; ok && s.Level > 0.01 {
            log.Printf("Lobe %s serotonin: %.3f", lobe.Name, s.Level)
        }
    }
}
```

### 2. Visualize Weights

```go
// Track synapse weights over time
func (game *Game) LogSynapseWeights(critter *Critter, synapseName string) {
    lobe := critter.Brain.GetLobe("decision")
    if syn := lobe.GetSynapse(synapseName); syn != nil {
        log.Printf("Tick %d: %s weight=%.3f trace=%.3f used=%d",
            game.TickCount, synapseName, syn.Weight, syn.Trace, syn.LastUsed)
    }
}
```

### 3. Test Learning in Controlled Scenarios

```go
func TestCritterLearnsToEat(t *testing.T) {
    critter := NewCritter()
    
    // Place critter near food repeatedly
    for i := 0; i < 100; i++ {
        // Simulate seeing food
        critter.Brain.GetLobe("sensory").SetNeuronInput("see_food", 1.0)
        critter.Brain.GetLobe("drive").SetNeuronInput("hunger", 0.8)
        
        // Step brain
        critter.Brain.Step(0.1, i)
        
        // If critter approached, give reward
        decision := critter.Brain.GetLobe("decision").GetNeuronOutput("approach")
        if decision > 0.5 {
            critter.Brain.GetLobe("decision").Reinforce(brain.Dopamine, 1.0)
            successCount++
        }
    }
    
    // After training, critter should reliably approach
    assert.Greater(t, successCount, 70, "Critter should learn to approach food")
}
```

## Advanced: Age-Dependent Learning

```go
func (critter *Critter) UpdatePlasticityForAge() {
    learningFactor := 1.0
    
    if critter.Age < 100 {
        learningFactor = 1.5  // Young: learn 50% faster
    } else if critter.Age > 500 {
        learningFactor = 0.5  // Old: learn 50% slower
    }
    
    for _, lobe := range critter.Brain.Lobes {
        lobe.PlasticityConfig.RewardLearningRate = 0.10 * learningFactor
        lobe.PlasticityConfig.PunishLearningRate = 0.15 * learningFactor
    }
}
```

## Summary

The key to successful learning in your game:

1. **Call `Brain.Step()` every game tick** - this updates traces and applies plasticity
2. **Call `Reinforce()` when meaningful events occur** - eating, getting hurt, achieving goals
3. **Scale reinforcement amounts** appropriately (0.1 to 1.0 typically)
4. **Use Dopamine for rewards, Serotonin for punishment**
5. **Tune learning rates** based on playtesting
6. **Allow sufficient time** for learning (100+ game ticks per learned behavior)

The system handles all the temporal credit assignment automatically through eligibility traces!
