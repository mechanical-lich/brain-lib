---
title: Getting started
nav_order: 2
permalink: /getting-started/
---

# Getting started
{: .no_toc }

1. TOC
{:toc}

## Prerequisites

- Go 1.25.1 or newer (the `go.mod` minimum; the GPU packages have been run
  with Go 1.26).
- Nothing else for the CPU paths. The GPU backend uses pure-Go WebGPU, so it
  needs no SDK. On Linux it needs the Vulkan loader (`libvulkan.so.1`) and a
  driver such as Mesa RADV; on macOS it uses Metal.

## Build and test

```bash
git clone git@github.com:mechanical-lich/brain-lib.git
cd brain-lib
go test ./...
```

The GPU tests skip with a message on a machine without a hardware adapter,
so `go test ./...` passes anywhere.

Build every command under `cmd/` for this machine into `bin/`:

```bash
make build
```

Cross-compile with `make build-linux`, `make build-mac`, `make build-windows`
or `make build-all`; binaries land in `bin/<os>-<arch>/`. `TOOLS=gpt-train`
restricts the build to one tool, and `make build-platform PLATFORM=linux/386`
targets any pair `go tool dist list` knows. Nothing uses cgo, so cross
builds need only the Go toolchain. The full target list is on the
[development](../development#make-targets) page.

## Use it as a library

```go
import "github.com/mechanical-lich/brain-lib/pkg/brain"

b := brain.NewBrain()

senses := brain.NewLobe("senses", nil)
senses.AddNeuron(&brain.Neuron{ID: "see_food"})
b.AddLobe(senses)

decision := brain.NewLobe("decision", nil)
decision.AddNeuron(&brain.Neuron{ID: "approach", Bias: 0.1})
b.AddLobe(decision)

// A plastic synapse between lobes. Weight 0.1, learns from reward.
b.Connect("senses", "see_food", "decision", "approach", 0.1, true)

for tick := 0; tick < 100; tick++ {
    senses.SetNeuronInput("see_food", 1.0)
    b.Step(0.1) // 0.1 simulated seconds
    if decision.GetNeuronOutput("approach") > 0.5 {
        decision.Reinforce(brain.Dopamine, 1.0)
    }
}

if err := b.Save("my_brain.json"); err != nil {
    // ...
}
```

The [brain overview](../brain/) explains what each of those calls does. For
the GPT, the [GPT overview](../gpt/) has an equivalent sketch.

## Run a tool

Each command runs with `go run ./cmd/<name>` or from `bin/` after
`make build`. For example, validate and repair a brain file:

```bash
go run ./cmd/validator validate local_data/test_brain.json
```

```bash
go run ./cmd/validator repair local_data/test_brain.json
```

Or fetch a corpus and train a small language model:

```bash
go run ./cmd/gpt-fetch
```

```bash
go run ./cmd/gpt-train -steps 2000 -layers 4 -dim 128
```

```bash
go run ./cmd/gpt-generate -prompt "The old house" -n 400
```

All flags are listed on the [command-line tools](../cli/) page.

## Repository layout

```
brain-lib/
├── cmd/            one main package per tool: brain-trainer, validator,
│                   gpt-fetch, gpt-train, gpt-generate
├── pkg/
│   ├── brain/      the creature brain (+ validation/)
│   ├── gpt/        the GPT: tensor/, tokenizer/, model/, train/, data/,
│   │               gpu/, gpumodel/
│   ├── ann/        feed-forward network
│   └── pnn/        plastic neural network sketch
├── docs/           this site, plus the original design notes
├── experiments/    standalone modules (gpu-spike), not part of the main module
├── local_data/     git-ignored: example brains, corpora, checkpoints
└── Makefile        build, cross-compile, test, vet
```

`local_data/` is ignored by git. The tools default to paths inside it
(`local_data/gutenberg`, `local_data/gpt/model.ckpt`), so a fresh clone has
nothing there until `gpt-fetch` or a training run creates it.
