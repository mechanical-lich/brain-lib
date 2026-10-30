---
title: PNN
nav_order: 7
permalink: /pnn/
---

# Plastic neural network (`pkg/pnn`)
{: .no_toc }

1. TOC
{:toc}

`pkg/pnn` is a sketch of a plastic neural network: a flat graph of neurons
and synapses with named input and output mappings, modulators, and a
dopamine-gated Hebbian learning rule. It shares ideas with
[`pkg/brain`](../brain/) but is a separate, much smaller implementation with no
lobes, receptors or emitters. It has no command; it is for experiments and
other artificial-life projects.

## Structure

```go
type PNN struct {
    Neurons    map[string]*Neuron
    Synapses   map[string]*Synapse
    Inputs     map[string]string   // sensorName -> neuronID
    Outputs    map[string]string   // motorName  -> neuronID
    Modulators map[string]*Modulator
    Seed       int64
    CollectDiagnostics bool        // not serialized
}
```

A `Neuron` has a bias, an activation function (linear, sigmoid, tanh), its
current value and the IDs of its incoming synapses. A `Synapse` has a
weight, `plastic`, an eligibility `trace` and clip bounds. A `Modulator` is a
level with a per-second decay.

`NewPNN(seed)` creates an empty network with a seeded RNG (`Rand()`). After
unmarshalling JSON, call `InitRNG()`.

## Stepping

```go
outs, diag := net.Step(map[string]float64{"see_food": 1.0}, 0.1)
```

`Step` does the following, in order:

1. Adds each named input to its mapped neuron's sum.
2. Propagates through synapses using a **snapshot** of the previous
   activations, so the update is synchronous and independent of map order.
3. Computes every neuron.
4. Decays the modulators.
5. Applies plasticity to each plastic synapse:

   ```
   trace  = trace * exp(-traceDecay*dt) + pre * post * dt
   weight += trace * learningRate * dopamine * dt     (only if dopamine != 0)
   clip
   ```

   The trace is built whether or not dopamine is present, so activity before
   the first reward can still be credited when one arrives. The update is
   additive, so a synapse at exactly 0 can still learn.
6. Returns the mapped outputs and, if `CollectDiagnostics` is set, a snapshot
   of every weight and modulator level.

Learning is gated by a modulator named `"dopamine"`; with none present, or
at level 0, nothing changes.

## Serialization

`ToJSON()` returns the network as a JSON string. The field names match the
Go struct tags (`neurons`, `synapses`, `inputs`, `outputs`, `modulators`,
`seed`).
