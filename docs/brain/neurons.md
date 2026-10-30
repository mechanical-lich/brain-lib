---
title: Neurons
parent: Brain
nav_order: 1
permalink: /brain/neurons/
---

# Neurons
{: .no_toc }

1. TOC
{:toc}

A `Neuron` is a rate-coded unit: its state is one float, `ActivationVal`,
recomputed every step from the weighted sum of its inputs.

## Fields

| Field | JSON | Meaning |
|---|---|---|
| `ID` | `id` | Unique within its lobe. Also must not collide with an emitter ID in the same lobe. |
| `Bias` | `bias` | Added to the input sum every step. |
| `Activation` | `activation` | `0` linear, `1` sigmoid, `2` tanh (`ActLinear`, `ActSigmoid`, `ActTanh`). |
| `ActivationVal` | `activation_val` | The output. Persisted, so a saved brain resumes mid-thought. |
| `ExternalInput` | `external_lobe_input` | Input from inter-lobe synapses, receptors and `SetNeuronInput`. Consumed and zeroed by every `Compute`. |
| `Threshold` | `threshold` | Outputs with magnitude below this become exactly 0. Also gates carried-over state (see below). |
| `Min`, `Max` | `min_value`, `max_value` | Clamps. `0` means unset. A negative `Min` opts the neuron into signed activation. |
| `DecayRate` | `decay_rate` | Per second. `0` means "use the lobe's". |
| `Persists` | `persists` | Keep state between steps. OR'd with the lobe's `persists`. |
| `Capacitor` | `capacitor` | Integrate input over time instead of replacing it. |
| `Hidden`, `X`, `Y` | | Editor hints, ignored by the simulation. |
| `GroupID` | `group_id` | Integer group tag used by the lab to constrain rewiring. |
| `LearningRate` | `learning_rate` | Serialized but never read. Plasticity comes from the lobe's config. |

## How an activation is computed

Each step, `Lobe.Step` sums `pre.ActivationVal * weight` over the neuron's
incoming intra-lobe synapses and calls `Compute(sum, dt, persists, rate)`:

```
v   = sum + Bias + ExternalInput
out = activation(v)
ExternalInput = 0
if |out| < Threshold { out = 0 }
```

What happens to `out` depends on the neuron's mode:

**Stateless** (the default). `ActivationVal = out`. The neuron is a pure
function of this step's input.

**Persisting** (`persists` on the neuron or its lobe). Asymmetric: a stronger
signal is taken immediately, a weaker one is relaxed toward with `DecayRate`
as the time constant.

```
if |out| >= |ActivationVal| {
    ActivationVal = out                                  // rise instantly
} else {
    ActivationVal = out + (ActivationVal - out) * exp(-rate*dt)   // fall
    if |ActivationVal| < Threshold { ActivationVal = 0 }
}
```

Rising instantly matters for attention: a near object appearing beside an
established far one should win focus on the first tick, not after seconds of
smoothing. Falling relaxes, so a persisting neuron can follow its input back
down to a lower non-zero value. The threshold snap on the way down is what
lets a memory *end*: without it a pulsed memory would fade toward zero
forever without reaching it, and the plastic synapses it feeds would never
count as unused.

**Capacitor** (`capacitor: true`). Integrates, then leaks:

```
ActivationVal += out * dt
ActivationVal *= exp(-rate*dt)
```

A charge that leaks back down through `Threshold` with nothing driving it
snaps to 0, but only on the way down: a capacitor integrates below its
threshold by design (a weight-2 input at `dt` 0.1 adds 0.2 per step, under a
0.3 threshold), so sub-threshold charges are not snapped while charging.

Finally the clamps: `Max` if set, and a floor of 0 unless `Min` is negative,
in which case `Min`.

{: .note }
Rectifying at zero is deliberate. Neurons are used as logic gates throughout
the game's brains (an AND is two weight-1 inputs and a bias of -1), and a
sub-threshold sum must come out as 0 or every gate inverts. Setting `Min`
below zero is how one neuron opts into signed activation, which is what lets
an inhibitory synapse invert rather than merely cancel.

## Decay rate resolution

A lobe's `decay_rate` is a *default* for the neurons that do not set one, not
an extra decay applied on top. `EffectiveDecayRate(lobeRate)` returns the
neuron's own rate if non-zero, else the lobe's. So a neuron with
`decay_rate: 0.005` in a lobe running 0.05 decays ten times slower, which is
how a long-lived memory lives in a lobe of short-lived ones.

A resolved rate of 0 on a persisting neuron means "rise instantly, never
fall". That needs both the neuron and the lobe to be unset, and the validator
reports it as `Frozen_Persisting_Neuron`.

## Driving neurons from outside

- `lobe.SetNeuronInput(id, v)` sets `ExternalInput`, which the next `Compute`
  consumes. Call it every step you want the input present.
- `lobe.GetNeuronOutput(id)` reads `ActivationVal`.
- Inter-lobe synapses and chemical receptors with a neuron target also add to
  `ExternalInput`, so it is the one channel for everything that is not an
  intra-lobe synapse.
