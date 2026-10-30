---
title: Home
layout: home
nav_order: 1
permalink: /
---

# brain-lib
{: .fs-9 }

Go packages and command-line tools for building, training and validating the
neural models used by the Critters project.
{: .fs-6 .fw-300 }

[Get started](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/mechanical-lich/brain-lib){: .btn .fs-5 .mb-4 .mb-md-0 }

---

brain-lib is one Go module, `github.com/mechanical-lich/brain-lib`, with no
cgo anywhere in it. It holds four unrelated kinds of neural model and the
tools that go with them.

| Package | What it is | Who uses it |
|---|---|---|
| [`pkg/brain`](brain/) | The live creature brain: rate-coded neurons grouped into lobes, neuromodulators, chemical receptors and emitters, and eligibility-trace plasticity with a two-factor Hebbian rule. Saved as JSON. | The critters game and the critter laboratory |
| [`pkg/brain/validation`](brain/validation) | Structural checks and repairs for brain files. | `cmd/validator`, the lab |
| [`pkg/gpt`](gpt/) | A from-scratch GPT: pure-Go tensors, hand-written backward passes, BPE tokenizer, AdamW, checkpoints, and a pure-Go WebGPU backend that runs the training step on a GPU. | `cmd/gpt-train`, `cmd/gpt-generate` |
| [`pkg/ann`](ann) | A classic feed-forward network trained by backpropagation. | `cmd/brain-trainer` |
| [`pkg/pnn`](pnn) | A plastic neural network sketch: a flat graph of neurons with a dopamine-gated Hebbian rule. | Experiments and other artificial-life projects |

## Where to start

- **Building a creature brain, or reading one the game saved.** Start with the
  [brain overview](brain/), then [plasticity](brain/plasticity) for how it
  learns and the [file format](brain/file-format) for what is on disk.
- **Training or sampling the language model.** The [GPT overview](gpt/) has
  the quick start; the [GPT guide](gpt/guide) walks through the math.
- **Running a tool.** Every command under `cmd/` is on the
  [command-line tools](cli/) page with its flags.
- **Contributing.** [Development](development) covers tests, benchmarks,
  cross-compiling and the GPU test switches.

## Conventions that hold everywhere

- **Time is in simulated seconds.** `Brain.Step(dt)` takes elapsed simulated
  time; every rate in the library is per second and every decay is
  exponential, so behaviour does not depend on frame rate. See
  [time and units](architecture#time-and-units).
- **Determinism.** Stepping a brain, validating one, batching training data
  and reducing on the GPU all iterate in a fixed order. The same inputs give
  the same outputs run to run.
- **Nothing silently falls back.** A GPU run with no GPU fails with a message.
  A brain file newer than the library refuses to load. Inputs of the wrong
  shape return an error or panic rather than producing garbage.
