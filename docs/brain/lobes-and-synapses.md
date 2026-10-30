---
title: Lobes and synapses
parent: Brain
nav_order: 2
permalink: /brain/lobes-and-synapses/
---

# Lobes and synapses
{: .no_toc }

1. TOC
{:toc}

## Lobes

A `Lobe` is a named group of neurons with everything local to them: its own
synapses, modulators, chemical receptors, neuroemitters and
`PlasticityConfig`. Lobe-level settings:

| Field | JSON | Meaning |
|---|---|---|
| `Name` | `name` | Must match the lobe's key in `Brain.Lobes`. |
| `Persists` | `persists` | Every neuron in the lobe keeps state between steps. |
| `DecayRate` | `decay_rate` | Default per-second decay for neurons that set none. |
| `WinnerTakesAll` | `winner_takes_all` | After computing, only the most active neuron keeps its activation; the rest are set to 0. Ties go to the smaller ID. |
| `PlasticityConfig` | `plasticity_config` | Learning parameters for this lobe's synapses and for inter-lobe synapses that target it. |
| `Modulators` | `modulators` | Lobe-level chemical levels. When a lobe has none of a given name, plasticity falls back to the brain's. |
| `X`, `Y`, `Width`, `Height` | | Editor hints. |

`NewLobe(name, modulators)` returns a lobe with the default plasticity config
and the given modulators (nil is fine). Add it to a brain with
`Brain.AddLobe`; that rebuilds the brain's update order.

Winner-takes-all resolves before traces, emitters and reinforcement run, so
a losing neuron reports 0 and accumulates no trace. That ordering is what
makes WTA gate Hebbian learning: only the winner's incoming synapses learn.

## Synapses

A `Synapse` is a directed, weighted edge. Its target is normally a neuron;
inside a lobe it may instead be a neuroemitter.

| Field | JSON | Meaning |
|---|---|---|
| `ID` | `id` | Unique within its owner (lobe or brain). |
| `From`, `To` | `from`, `to` | Neuron IDs (or an emitter ID for `to`). |
| `FromLobe`, `ToLobe` | `from_lobe`, `to_lobe` | Which lobes. Decide ownership (below). Not read by the simulation for intra-lobe synapses. |
| `Weight` | `weight` | Current weight. |
| `Plastic` | `plastic` | Learns from dopamine and serotonin via the eligibility trace. |
| `Hebbian` | `hebbian` | Learns from co-activity alone. Independent of `plastic`. Omitted when false. |
| `Trace` | `trace` | Eligibility trace. Persisted. |
| `MinWeight`, `MaxWeight` | `min_weight`, `max_weight` | Clip bounds. `Connect` uses `[-2, 2]`. |
| `InitWeight` | `init_weight` | Baseline that inactivity decay and Hebbian decay pull toward. |
| `TraceDecay` | `trace_decay` | Per-synapse override of the lobe's eligibility-trace decay. `0` means use the lobe's. Omitted when zero. |
| `LastUsed`, `UsageCount` | `last_used`, `usage_count` | Tick of last use and total uses. "Used" means the source was active, whatever the weight. |
| `MinDecayFrac` | `min_decay_frac` | Serialized but never read; decay has no floor. |

## Synapse ownership

A synapse lives in exactly one place, and which one is determined by its
lobes:

- **Intra-lobe** (`FromLobe == ToLobe`): stored in that `Lobe.Synapses`,
  stepped by `Lobe.Step`.
- **Inter-lobe** (`FromLobe != ToLobe`): stored in `Brain.Synapses`, stepped
  by `Brain.Step`, which injects into the target neuron's `ExternalInput`.

`Brain.AddSynapse` routes on this rule, inferring the lobes from neuron
membership when they are not set. The [validator](../validation) detects and
repairs violations: a synapse filed in the wrong place, or naming lobes that
do not exist.

For an intra-lobe synapse the `FromLobe` and `ToLobe` fields are not read by
the simulation at all; `Lobe.Step` resolves connections by neuron ID within
its own map. They matter for routing, serialization and validation.

The two kinds differ in one more way. An intra-lobe synapse's effect is
computed inside `Lobe.Step`, so a neuron sees its intra-lobe inputs from
this step's already-computed predecessors (the lobe orders neurons
topologically). An inter-lobe synapse's contribution arrives via
`ExternalInput`, injected just before the target lobe steps, from a source
lobe that has already stepped this tick.

## Update order

A brain steps its lobes in topological order: a lobe that only feeds others
steps before the lobes it feeds, so a signal crosses any acyclic chain of
lobes in a single tick. The order is a depth-first sort over inter-lobe
synapses with lobe names visited alphabetically, so it is stable; cycles
between lobes are allowed and simply mean one of them sees last tick's value.
`RefreshUpdateOrder` rebuilds it, and the mutators (`AddLobe`, `AddSynapse`,
`RemoveSynapse`, loading) call it for you.

Within a lobe, neurons are ordered the same way over intra-lobe synapses.

## Connecting

```go
// Inter-lobe: UUID id, bounds [-2, 2], init_weight = weight.
syn := b.Connect("senses", "see_food", "decision", "approach", 0.1, true)

// Intra-lobe: id "from->to" (numbered suffix if repeated).
syn2 := decision.Connect("approach", "flee", -0.5, false)

// Anything else: build it and add it. Ownership is inferred.
b.AddSynapse(&brain.Synapse{
    ID: "recog_1", From: "round", To: "slot_1",
    FromLobe: "senses", ToLobe: "recognition",
    Weight: 0.05, InitWeight: 0.05, MinWeight: 0, MaxWeight: 1,
    Hebbian: true,
})
```

`Brain.RemoveSynapse(id)` and `Lobe.RemoveSynapse(id)` remove by ID.
`Lobe.RemoveNeuron(id)` and `Lobe.RemoveEmitter(id)` also remove the
synapses attached to them.
