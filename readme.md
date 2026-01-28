# brain-lib

brain-lib is a collection of Go packages and command-line tools for building, training, and validating neural/brain models used by the Critters project. It contains reusable implementations (ANNs, plasticity rules, neurotransmitter abstractions, etc.), datasets for examples, and small utilities for training and validating brains.

## Highlights

- Modular Go packages for artificial neural networks and brain simulations
- Example CLI programs for training and validating brains (in `cmd/`)
- Local example datasets in `local_data/` and documentation for neuroemitters/plasticity

## Quick Start

Prerequisites:

- Go 1.20+ installed

Clone the repo (if not already in your workspace) and run the tests / build examples:

```bash
cd brain-lib
go test ./...
# build example CLI tools
go build ./cmd/brain-trainer
go build ./cmd/sandbox
go build ./cmd/validator
```

Run the brain trainer example (binary `cmd/brain-trainer`):

```bash
./cmd/brain-trainer
```

See each `cmd/*` folder for command-specific flags and usage.

## Repository Layout

- `cmd/` — small command-line tools and example programs (e.g., `brain-trainer`, `sandbox`, `validator`).
- `docs/` — design notes and guides (e.g., `NEUROEMITTERS.md`, `PLASTICITY_GUIDE.md`).
- `local_data/` — example JSON brains and training datasets used by tests and examples.
- `pkg/` — core packages:
	- `ann/` — artificial neural network implementations and training helpers
	- `brain/` — higher-level brain model abstractions, integration tests, and utilities
	- `pnn/` — probabilistic neural network helpers

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
