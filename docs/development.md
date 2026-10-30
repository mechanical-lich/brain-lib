---
title: Development
nav_order: 9
permalink: /development/
---

# Development
{: .no_toc }

1. TOC
{:toc}

## Tests

```bash
go test ./...
```

```bash
go test ./... -v
```

Everything is a plain `go test`. The GPU packages skip when no hardware
adapter is present (see [GPU backend](../gpt/gpu#testing-and-environment-switches)
for the environment switches), so the full suite passes on any machine.

Useful subsets:

| Command | What it covers |
|---|---|
| `go test ./pkg/brain/...` | The creature brain and validator. |
| `go test ./pkg/gpt/model` | Gradient checks against finite differences. Run this first after changing any `Backward`. |
| `go test ./pkg/gpt/train` | Checkpoint round-trip, the overfit-a-pattern loop test. |
| `go test ./pkg/gpt/tokenizer` | Char, Symbol and BPE round-trips. |
| `go test ./pkg/gpt/gpu ./pkg/gpt/gpumodel` | Every GPU kernel and the full step against the CPU. |

## Benchmarks

```bash
go test -run XXX -bench . ./pkg/brain/
```

```bash
go test -run XXX -bench . ./pkg/gpt/tensor/ ./pkg/gpt/model/
```

```bash
go test -run XXX -bench . -benchtime 5x ./pkg/gpt/gpu/ ./pkg/gpt/gpumodel/
```

`pkg/gpt/gpumodel`'s `TestProfileStep` (run with `-v`) prints the GPU time
of each named phase of a training step.

## Make targets

| Target | Does |
|---|---|
| `make build` | Every tool under `cmd/` for this machine, into `bin/`. |
| `make build-linux` | linux/amd64 and linux/arm64, into `bin/linux-<arch>/`. |
| `make build-mac` | darwin/arm64 and darwin/amd64. |
| `make build-windows` | windows/amd64 and windows/arm64 (`.exe`). |
| `make build-all` | Every platform in `PLATFORMS`. |
| `make build-platform PLATFORM=linux/386` | Any GOOS/GOARCH pair `go tool dist list` knows. |
| `make build-<os>-<arch>` | One platform, e.g. `make build-linux-amd64`. |
| `make list-tools` | Print the tools that would be built. |
| `make test`, `make vet` | `go test ./...`, `go vet ./...`. |
| `make clean` | Remove `bin/`. |
| `make vendor`, `make update-go-deps` | Vendor, or bump every direct dependency and tidy. |

Variables: `TOOLS` (default: every directory under `cmd/`) limits which
commands are built; `BIN` changes the output root; `PLATFORMS` the list
`build-all` iterates. Cross builds use `CGO_ENABLED=0 -trimpath -ldflags
'-s -w'`.

## Dependencies

| Module | Used by | For |
|---|---|---|
| `github.com/google/uuid` | `pkg/brain` | Inter-lobe synapse IDs. |
| `github.com/gogpu/wgpu`, `github.com/gogpu/gputypes` | `pkg/gpt/gpu` | Pure-Go WebGPU. |
| `github.com/stretchr/testify` | tests | Assertions. |

Nothing uses cgo.

## Conventions

- **Rates are per second; decay is exponential.** A new time-dependent
  quantity in `pkg/brain` goes through `decayFactor(rate, dt)` and scales
  its input term by `dt`. Add a test in the style of `timestep_test.go`
  that runs the same simulated second at two or three `dt` values.
- **Iterate in a fixed order.** Anything that sums floats or produces a list
  from a map sorts first. There are tests that assert determinism; keep them
  passing.
- **Additive file-format changes.** A new brain field is `omitempty` and
  zero-means-unset, so old files load unchanged and `CurrentFormatVersion`
  does not move. Bump it only for a change in meaning, and document the
  version in the `CurrentFormatVersion` comment.
- **Comments say why.** Many comments in `pkg/brain` describe the bug a line
  fixes and the measurement that found it. Keep that habit; the measurements
  are what make the next change safe.
- **Validation is a library.** A new structural rule is a new `ComplaintType`
  with a test that builds the bad brain, and a repair only if the fix is
  mechanical. Design decisions are warnings.
- **GPU kernels follow the design rules** on the [GPU backend](../gpt/gpu#design-rules)
  page and get a test against the CPU implementation at the training shapes
  and at a ragged shape.

## This documentation

The site lives in `docs/` and is built by Jekyll with the `just-the-docs`
theme. GitHub Pages serves it from the `main` branch, `/docs` folder. To
preview locally:

```bash
cd docs && bundle install && bundle exec jekyll serve
```

Every `.md` file in `docs/` with front matter is a page; `parent:` and
`nav_order:` place it in the sidebar. The three original design notes
(`NEUROEMITTERS.md`, `PLASTICITY_GUIDE.md`, `GPT_GUIDE.md`) stay at their
paths so links from the package readmes keep working.
