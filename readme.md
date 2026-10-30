# brain-lib

brain-lib is a collection of Go packages and command-line tools for building, training, and validating neural/brain models used by the Critters project. It contains reusable implementations (ANNs, plasticity rules, neurotransmitter abstractions, etc.), datasets for examples, and small utilities for training and validating brains.

## Highlights

- Modular Go packages for artificial neural networks and brain simulations
- Example CLI programs for training and validating brains (in `cmd/`)
- Local example datasets in `local_data/` and documentation for neuroemitters/plasticity

## Quick Start

Prerequisites:

- Go 1.25.1+ installed (the `go.mod` minimum)

Clone the repo (if not already in your workspace) and run the tests / build examples:

```bash
cd brain-lib
go test ./...
# build every CLI tool in cmd/ for this machine -> bin/
make build
```

Cross-compile with `make build-linux` (amd64 + arm64), `make build-mac`,
`make build-windows`, or `make build-all`; binaries land in
`bin/<os>-<arch>/`. `TOOLS=gpt-train` limits which commands are built and
`make build-platform PLATFORM=linux/386` targets any other GOOS/GOARCH pair.

Run the brain trainer example (binary `cmd/brain-trainer`):

```bash
./cmd/brain-trainer
```

See each `cmd/*` folder for command-specific flags and usage.

## Repository Layout

- `cmd/` — small command-line tools and example programs (e.g., `brain-trainer`, `sandbox`, `validator`).
- `docs/` — design notes and guides (e.g., `NEUROEMITTERS.md`, `PLASTICITY_GUIDE.md`, `GPT_GUIDE.md`).
- `local_data/` — example JSON brains and training datasets used by tests and examples.
- `pkg/` — core packages:
	- `ann/` — artificial neural network implementations and training helpers
	- `brain/` — higher-level brain model abstractions, integration tests, and utilities; plasticity is three-factor (trace × dopamine/serotonin) and, per synapse, two-factor Hebbian (instar rule on activations alone, for familiarity, conjunctions and competitive recognition slots); see `docs/PLASTICITY_GUIDE.md`
	- `pnn/` — probabilistic neural network helpers
	- `gpt/` — minimal from-scratch GPT (pure-Go tensors, manual backprop); see `pkg/gpt/readme.md`

## Usage

Import packages from `brain-lib/pkg` in other Go programs. Example:

```go
import "github.com/mechanical-lich/brain-lib/pkg/brain"

// create or load a brain, run, train, etc.
```

Refer to package tests and `examples` in `pkg/*` for usage patterns.

## Testing

Run the unit tests for all packages:

```bash
cd brain-lib
go test ./... -v
```

## Package map

Which package is used by what, so "unused here" is not mistaken for "dead":

| Package | Consumers | Notes |
|---|---|---|
| `pkg/brain` | `critters` (the game), `critter-laboratory` | The live library. Rate-coded neurons, neuromodulators, chemical receptors and emitters, eligibility-trace plasticity. |
| `pkg/brain/validation` | `cmd/validator`, the lab | Structural checks and repair for brain files. |
| `pkg/ann` | `cmd/brain-trainer` | Classic feed-forward network with backpropagation. Not used by the game. |
| `pkg/pnn` | Experimental / other alife projects | Plastic neural network sketch. Shares some concepts with `pkg/brain` but is a separate implementation. |

## Synapse ownership

A synapse lives in exactly one place, and which one is determined by its lobes:

- **Intra-lobe** (`FromLobe == ToLobe`): stored in that `Lobe.Synapses`, stepped
  by `Lobe.Step`.
- **Inter-lobe** (`FromLobe != ToLobe`): stored in `Brain.Synapses`, stepped by
  `Brain.Step`, which injects into the target's `ExternalInput`.

`Brain.AddSynapse` routes on this rule, inferring the lobes from neuron
membership when they are not set. `cmd/validator` detects and repairs
violations — a synapse filed in the wrong place, or naming lobes that do not
exist.

Note that for an intra-lobe synapse the `FromLobe`/`ToLobe` fields are not read
by the simulation at all; `Lobe.Step` resolves connections by neuron ID within
its own map. They matter for routing, serialization and validation.

## Time and units

All rates are **per second**, and all decay is exponential, so behaviour does
not depend on the simulation's frame rate. `Brain.Step(dt)` takes the elapsed
time in seconds.

"Second" means a **simulated** second, as defined by the `dt` the caller
supplies — the library has no wall clock. The critters game, for example,
steps the brain with a constant `dt` of 0.1 at 60 ticks per wall second, so a
brain there experiences six simulated seconds per wall second: a 1 s
eligibility window (`trace_decay: 1.0`) lasts about 0.17 wall seconds, and a
neuron `decay_rate` of 0.05/s halves in about 14 simulated seconds, or 2.3
wall seconds.

The eligibility-trace decay rate (`trace_decay`, per second) is set per lobe in
`plasticity_config`, and a synapse may override it with its own `trace_decay`
to get a wider or narrower reinforcement window than the rest of its lobe;
0 or absent means "use the lobe's" (the target lobe's, for an inter-lobe
synapse).

Brain files carry a `format_version`; see `CurrentFormatVersion` in
`pkg/brain/brain.go` for the history and what a version 0 file means.
