---
title: Architecture
nav_order: 3
permalink: /architecture/
---

# Architecture
{: .no_toc }

How the pieces of brain-lib relate, and the conventions every package shares.
{: .fs-6 .fw-300 }

1. TOC
{:toc}

## One module, four models

The module is `github.com/mechanical-lich/brain-lib`. Its packages do not
depend on each other across the four model families; `pkg/brain`, `pkg/gpt`,
`pkg/ann` and `pkg/pnn` are independent and can be imported separately.

```
pkg/brain ──── pkg/brain/validation
                      ▲
                      │
cmd/validator ────────┘

pkg/gpt/tensor ◄── pkg/gpt/model ◄── pkg/gpt/train ◄── cmd/gpt-train
                        ▲               ▲   ▲           cmd/gpt-generate
pkg/gpt/tokenizer ──────┘               │   │
pkg/gpt/data ───────────────────────────┘   │
pkg/gpt/gpu ◄── pkg/gpt/gpumodel ───────────┘      cmd/gpt-fetch ── pkg/gpt/data

pkg/ann ◄── cmd/brain-trainer

pkg/pnn   (no command)
```

The dependency that matters most is the one that is absent: nothing in
`pkg/gpt/model` or `pkg/gpt/train` imports `pkg/gpt/gpu`. The CPU path is the
default and is unchanged by the GPU backend. `gpumodel` adapts a GPU-resident
model to the `train.Stepper` interface, and `cmd/gpt-train` picks one or the
other from its `-backend` flag.

## Package map

Which package is used by what, so "unused here" is not mistaken for "dead":

| Package | Consumers | Notes |
|---|---|---|
| `pkg/brain` | The critters game, the critter laboratory | The live library. Rate-coded neurons, neuromodulators, chemical receptors and emitters, eligibility-trace plasticity, Hebbian learning. |
| `pkg/brain/validation` | `cmd/validator`, the lab | Structural checks and repair for brain files. |
| `pkg/gpt/*` | `cmd/gpt-fetch`, `cmd/gpt-train`, `cmd/gpt-generate` | Prose today; built to be retargeted at any discrete token stream. |
| `pkg/ann` | `cmd/brain-trainer` | Classic feed-forward network with backpropagation. Not used by the game. |
| `pkg/pnn` | Experimental and other artificial-life projects | A separate implementation that shares concepts with `pkg/brain`. |
| `experiments/gpu-spike` | Nobody | Its own Go module, so the main module takes no dependency on it. The matmul benchmark that justified the GPU backend. |

## Time and units

All rates are **per second**, and all decay is exponential, so behaviour does
not depend on the simulation's frame rate. `Brain.Step(dt)` takes the elapsed
time in seconds. A rate `r` applied over `dt` multiplies by `exp(-r*dt)`,
which is exact at any step size and never diverges; the library used to apply
the linear form `x -= x*r*dt`, which goes negative once `r*dt` passes 1 and
blows up past 2.

"Second" means a **simulated** second, defined by the `dt` the caller
supplies. The library has no wall clock. The critters game steps the brain
with a constant `dt` of 0.1 at 60 ticks per wall second, so a brain there
experiences six simulated seconds per wall second: a one-second eligibility
window (`trace_decay: 1.0`) lasts about 0.17 wall seconds, and a neuron
`decay_rate` of 0.05 per second halves in about 14 simulated seconds, or 2.3
wall seconds.

The GPT side has no simulated time; its unit is the optimizer step.

## Determinism

Go randomises map iteration, and several of the library's data structures are
maps. Every place where iteration order would change a result keeps a sorted
index instead:

- A brain steps its lobes in a topological order that is rebuilt whenever
  the wiring changes, with ties broken by name. Within a lobe, neurons are
  summed from a sorted index of incoming synapses, so floating-point
  accumulation is reproducible.
- Chemical receptors consume from a shared pool in that same lobe order,
  because whoever runs first gets the chemical.
- The validator lists complaints in a stable order so two runs diff cleanly.
- On the GPU, reductions that sum into shared rows use a host-built CSR map
  and a fixed order rather than float atomics, which WGSL does not have.
- The PNN sums against a snapshot of the previous activations and commits
  afterwards, so its update is genuinely synchronous.

A fixed seed therefore reproduces a run bit for bit on the CPU. GPU float32
arithmetic agrees with the CPU to about the third decimal place.

## Serialization

| Thing | Format | Notes |
|---|---|---|
| Brain | JSON, snake_case keys, `format_version` | See the [file format](../brain/file-format) page. Version 0 files load with compatibility rules. |
| ANN | JSON, camelCase keys | `NumInputs`, `Weights[layer][neuron][input]`, `Biases`. |
| PNN | JSON | `ToJSON()`; call `InitRNG()` after unmarshalling. |
| GPT checkpoint | `encoding/gob` | Config, step, every parameter by name, AdamW moments, opaque tokenizer bytes. Written to a temp file and renamed, so never half-written. |
| Tokenizer | JSON envelope tagged with its kind | Stored inside the checkpoint. |

## Error handling

- Shape mismatches in `pkg/gpt/tensor` and `pkg/gpt/gpu` panic, as a
  programming error. GPU-side errors come back from `Submit`.
- A brain file with a `format_version` newer than the library's refuses to
  load.
- The validator never prints; it returns complaints and a repair report so a
  server or UI can decide what to show.
- `ann.Process` returns nil on a wrong-sized input; `ProcessChecked` returns
  an error instead.
