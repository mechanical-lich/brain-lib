---
title: Brain
nav_order: 4
has_children: true
permalink: /brain/
---

# The creature brain (`pkg/brain`)
{: .no_toc }

A rate-coded neural network with chemistry: neurons grouped into lobes,
neuromodulators that gate learning, receptors and emitters that couple the
brain to a chemical pool, and synapses that learn from reward or from
co-activity. This is the library the critters game runs.
{: .fs-6 .fw-300 }

1. TOC
{:toc}

## The pieces

| Type | Role |
|---|---|
| `Brain` | The whole thing: a map of lobes, the inter-lobe synapses, brain-level modulators and receptors, a default `PlasticityConfig`, a tick counter and the file's `format_version`. |
| `Lobe` | A named group of neurons with its own synapses, modulators, receptors, neuroemitters and plasticity config. Optional `persists`, `decay_rate` and `winner_takes_all`. |
| `Neuron` | Bias, activation function (linear, sigmoid, tanh), optional threshold, min/max clamps, per-neuron decay rate, `persists` and `capacitor` modes. |
| `Synapse` | Directed, weighted edge between two neurons (or from a neuron to an emitter). `plastic` enables three-factor learning; `hebbian` enables two-factor learning. Carries an eligibility trace, weight bounds, `init_weight` and usage stats. |
| `Modulator` | A named chemical level (dopamine, serotonin, endorphin, acetylcholine) that decays per second. Lobe-level or brain-level. |
| `ChemicalReceptor` | Reads a chemical from an external pool and drives a neuron or a modulator with it. |
| `Neuroemitter` | Driven by synapses; releases a chemical into the pool when its accumulated input crosses a threshold. |
| `PlasticityConfig` | Trace decay, reward and punishment learning rates, inactivity decay, Hebbian rate and decay. Per lobe, with the brain's as a fallback. |

The detail pages:

- [Neurons](neurons): how an activation is computed, including persisting and capacitor modes.
- [Lobes and synapses](lobes-and-synapses): synapse ownership, update order, winner-takes-all.
- [Chemistry](chemistry): modulators, receptors, emitters and the `ApplyChemistry` loop.
- [Plasticity](plasticity): the three-factor and Hebbian learning rules and how to tune them.
- [File format](file-format): the JSON on disk and its version history.
- [Validation](validation): what the validator checks and repairs.

## A step, in order

`Brain.Step(dt)` is the only call the simulation makes per tick. It does the
following, in this order:

1. Increments `Tick`.
2. Decays every brain-level modulator.
3. For each lobe, in topological order (sources before targets):
   1. Injects inter-lobe input: for each synapse into this lobe, adds
      `from.ActivationVal * weight` to the target neuron's `ExternalInput`,
      and records the synapse as used if its source was active.
   2. Calls `Lobe.Step`, which:
      - decays the lobe's modulators;
      - computes every neuron, in topological order within the lobe, from its
        incoming intra-lobe synapses, its bias and its `ExternalInput`;
      - resolves winner-takes-all;
      - updates eligibility traces on plastic intra-lobe synapses from the
        post-WTA activations;
      - feeds neuroemitters from their synapses;
      - applies reinforcement and Hebbian learning to intra-lobe synapses;
      - applies inactivity decay.
   3. Updates eligibility traces on plastic inter-lobe synapses into this lobe.
4. Applies reinforcement and Hebbian learning to inter-lobe synapses, using the
   target lobe's modulators and config (falling back to the brain's).

Chemistry is separate. The caller owns the chemical pool, a
`map[string]float64`, and decides when to run it:

```go
b.Step(dt)
for _, lobe := range b.Lobes {
    chemicals = lobe.UpdateNeuroemitters(dt, chemicals) // emitters add
}
b.ApplyChemistry(chemicals, dt)                         // receptors consume
```

`ApplyChemistry` runs the brain-level receptors first, then each lobe's in
the same stable lobe order, because receptors consume from a shared pool and
whoever runs first gets the chemical.

## Building a brain in code

```go
b := brain.NewBrain()

senses := brain.NewLobe("senses", nil)
senses.AddNeuron(&brain.Neuron{ID: "see_food"})
senses.AddNeuron(&brain.Neuron{ID: "see_predator"})
b.AddLobe(senses)

decision := brain.NewLobe("decision", nil)
decision.AddNeuron(&brain.Neuron{ID: "approach"})
decision.AddNeuron(&brain.Neuron{ID: "flee"})
decision.WinnerTakesAll = true
b.AddLobe(decision)

// Inter-lobe, plastic: these learn from dopamine and serotonin.
b.Connect("senses", "see_food", "decision", "approach", 0.1, true)
b.Connect("senses", "see_predator", "decision", "flee", 0.1, true)

// Intra-lobe, fixed: mutual inhibition inside the decision lobe.
decision.Connect("approach", "flee", -0.5, false)
decision.Connect("flee", "approach", -0.5, false)
```

`Brain.Connect` makes an inter-lobe synapse with a UUID, bounds of
`[-2, 2]`, and `init_weight` equal to the starting weight. `Lobe.Connect`
makes an intra-lobe one with the ID `from->to` (a repeat connection of the
same pair gets a numbered suffix rather than replacing the first). To set
other fields, build the `Synapse` yourself and pass it to `AddSynapse`;
`Brain.AddSynapse` routes it to the right owner from its lobes, inferring
them from neuron membership when they are unset.

Drive it from the game loop:

```go
senses.SetNeuronInput("see_food", 1.0)   // sets ExternalInput for this step
b.Step(0.1)
if decision.GetNeuronOutput("approach") > 0.5 {
    decision.Reinforce(brain.Dopamine, 1.0)  // reward what just happened
}
```

`Reinforce` adds to a modulator, creating it with a default decay rate if the
lobe has none (dopamine 0.5/s, serotonin 0.3/s, endorphin 0.8/s). The next
`Step` turns that level, times each synapse's eligibility trace, into weight
changes. `Brain.Reinforce` does the same at brain level, which reaches every
inter-lobe synapse and every lobe without its own modulator of that name.

## Loading and saving

```go
b, err := brain.LoadBrain("critter.json")   // or LoadFromJson([]byte)
// ...
err = b.Save("critter.json")                 // or ToJSON()
```

Loading rebuilds the update order and the indexes. Saving writes the current
`format_version`. See the [file format](file-format) page for what a
version 0 file means.

## Modulators and the four chemicals

Four names are predeclared; any string works as a modulator or chemical
name, but plasticity reads exactly two of them:

| Constant | Name | Read by |
|---|---|---|
| `Dopamine` | `"dopamine"` | Reward learning rate |
| `Serotonin` | `"serotonin"` | Punishment learning rate |
| `Endorphin` | `"endorphin"` | Nothing in the library; a signal for the game |
| `Acetylcholine` | `"acetylcholine"` | Nothing yet |

Modulator levels never go negative: `Add` clamps at zero, because plasticity
multiplies the level by a signed learning rate, and a negative level would
turn punishment into reward.

## What the tests pin down

`pkg/brain` has integration tests for every behaviour described on these
pages. The ones worth reading as worked examples:

- `TestSteppingIsDeterministic` and `TestUpdateOrderIsStableAndPhantomFree` in `brain_test.go`.
- `TestBrainPlasticityInterLobeReward`, `...Punishment`, `...InactivityDecay`, `...HierarchicalModulators` in `brain_integration_test.go`.
- `TestHebbianFamiliarityMemory`, `TestWinnerTakesAllGatesHebbianLearning`, `TestHebbianWTARecruitsAFreshSlot` in `hebbian_test.go`.
- `TestChemicalReceptor_NormalizedNeuronOutputIsALevelNotARate` and its siblings in `chemical_receptor_test.go`.
- `neuron_logic_gate_test.go`: neurons as AND, OR and NOT gates.
- `timestep_test.go`: the same simulated second at different `dt`.
