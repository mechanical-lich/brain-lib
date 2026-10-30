---
title: Chemistry
parent: Brain
nav_order: 3
permalink: /brain/chemistry/
---

# Chemistry: modulators, receptors and emitters
{: .no_toc }

1. TOC
{:toc}

Three things in `pkg/brain` deal in chemicals, and they are easy to confuse:

| | Lives in | Reads | Writes |
|---|---|---|---|
| **Modulator** | A lobe or the brain | `Reinforce`, receptors with a modulator target | Plasticity (dopamine, serotonin) |
| **Chemical receptor** | A lobe or the brain | The caller's chemical pool | A neuron's `ExternalInput` or a modulator's level |
| **Neuroemitter** | A lobe only | Synapses from neurons | The caller's chemical pool |

The **chemical pool** is a `map[string]float64` that the caller owns. It is
the bloodstream: emitters add to it, receptors consume from it, and the game
can add or read anything it likes between steps (hunger, pain, a smell in the
air). The library never stores it.

## Modulators

A `Modulator` is a named level that decays exponentially at `DecayRate` per
second. `Add(delta)` changes it, clamping at zero. Plasticity reads two of
them, `dopamine` and `serotonin`, from the lobe first and the brain second.

`Lobe.Reinforce(name, amount)` and `Brain.Reinforce(name, amount)` add to a
modulator, creating it with a default decay rate when absent: 0.5/s for most,
0.3/s for serotonin (pain lingers), 0.8/s for endorphin (satisfaction is
brief). Call them when something meaningful happens in the game; the next
`Step` converts the level into weight changes through each synapse's
eligibility trace. See [plasticity](../plasticity).

## Chemical receptors

A `ChemicalReceptor` watches one chemical in the pool and drives one target.

| Field | JSON | Meaning |
|---|---|---|
| `ChemicalName` | `chemical_name` | Which pool entry to read. |
| `UptakeRate` | `uptake_rate` | Per second. Consumption is `concentration * uptake_rate * dt`, clamped to the pool. |
| `Threshold` | `threshold` | Below this concentration the receptor does nothing. |
| `Target`, `TargetType` | `target`, `target_type` | A neuron ID with `"neuron"`, or a modulator name with `"modulator"`. |
| `ReceptorType` | `receptor_type` | `"excitatory"` or `"inhibitory"`. Inhibitory negates the output. |
| `OutputNormalizationMin`, `...Max` | `output_normalization_min`, `..._max` | When either is non-zero, the output is the concentration mapped onto 0..1 over this range instead of the amount taken up. |

What a receptor delivers, from `GetOutput`:

```
uptake = min(concentration * UptakeRate * dt, concentration)
output = uptake
if normalization set {
    output = clamp((concentration - Min) / (Max - Min), 0, 1)
    if target is a modulator { output *= dt }   // a rate, not a level
}
if inhibitory { output = -output }
consumed = uptake
```

Two things follow. **Output and consumption are independent** when
normalization is set: a receptor can read a pool strongly while draining it
slowly. And **a neuron target gets a level, a modulator target gets a rate.**
A neuron's `ExternalInput` is consumed and zeroed every step, so it wants a
0..1 value that stands beside hunger, pain and boredom. A modulator
accumulates and decays on its own, so it wants `dt` scaled in, which is what
makes its level independent of frame rate.

Add receptors with `lobe.AddReceptor(...)` or `brain.AddReceptor(...)`. A
brain-level receptor can target a neuron in any lobe by ID.

`Brain.ApplyChemistry(pool, dt)` runs the brain's receptors and then each
lobe's in the stable lobe order, mutating the pool as it goes. Receptors
share the pool, so order matters: with dopamine receptors at a high uptake
rate in two lobes, the first one takes nearly the whole pool. Lobe order used
to be a map iteration, which made that a coin flip; it is now the update
order. `Lobe.UpdateChemicalReceptors(dt, pool)` and `Lobe.ApplyChemical(name,
concentration, dt)` are the per-lobe entry points.

## Neuroemitters

A `Neuroemitter` is the inverse of a receptor: neural activity produces a
chemical. Emitters exist only in lobes.

| Field | JSON | Meaning |
|---|---|---|
| `ID` | `id` | Unique within the lobe, and distinct from every neuron ID there. |
| `ChemicalName` | `chemical_name` | What it releases. |
| `EmissionRate` | `emission_rate` | Scales the emission. |
| `Threshold` | `threshold` | Accumulated input must reach this to emit. |
| `AccumulatedActivation` | `accumulated_activation` | Input received since the last emission. Persisted, so a save between receiving and emitting keeps the charge. |

Neurons connect to an emitter with ordinary intra-lobe synapses whose `to`
is the emitter's ID. During `Lobe.Step`, after neurons and WTA have resolved,
each emitter accumulates `pre.ActivationVal * weight` over its synapses.
Then, when the caller runs `lobe.UpdateNeuroemitters(dt, pool)`:

```
if accumulated >= Threshold {
    pool[ChemicalName] += accumulated * EmissionRate * dt
}
accumulated = 0
```

A plastic synapse into an emitter builds a trace from the presynaptic
activation alone (`pre * dt`), since an emitter has no activation of its own.
A Hebbian synapse into an emitter only ever forgets, for the same reason.

```go
reward := brain.NewLobe("reward", nil)
reward.AddNeuron(&brain.Neuron{ID: "ate_food"})
reward.AddEmitter("dopa_emit", "dopamine", 0.5, 0.5)   // rate, threshold
reward.Connect("ate_food", "dopa_emit", 1.0, false)
```

`AddEmitter`, `GetEmitter` and `RemoveEmitter` (which also removes its
synapses) manage them. The original design note,
[Neuroemitters](../neuroemitters), has a worked numerical example.

## Putting it together

A game tick that uses all three:

```go
pool := map[string]float64{"pain": critter.Pain, "food_smell": smellHere}

b.Step(dt)
for _, l := range b.Lobes {
    pool = l.UpdateNeuroemitters(dt, pool)  // e.g. dopamine from a reward lobe
}
b.ApplyChemistry(pool, dt)                   // receptors: pain -> serotonin modulator,
                                             //            food_smell -> a sensory neuron,
                                             //            dopamine -> the decision lobe's modulator
```

Receptors that target a modulator are the usual bridge from the pool into
learning: an emitter in one lobe releases dopamine into the pool, and a
receptor in the decision lobe turns it into that lobe's dopamine level, which
the next `Step` applies to its eligibility traces.
