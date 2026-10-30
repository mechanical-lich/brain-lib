---
title: GPT
nav_order: 5
has_children: true
permalink: /gpt/
---

# The GPT (`pkg/gpt`)
{: .no_toc }

A minimal decoder-only transformer written from scratch in pure Go: a
row-major matrix type, hand-written backward passes, no autograd. It is small
enough to read end to end, and built so the same model can be pointed at any
discrete token stream, not just prose.
{: .fs-6 .fw-300 }

1. TOC
{:toc}

## Packages

| Package | Purpose |
|---|---|
| `tensor/` | Row-major float64 `Mat` and the few ops a transformer needs: `MatMul`, `MatMulT` (a @ b^T), `MatMulTA` (a^T @ b), broadcast add, column sums, `Parallel`. The matmuls are cache-tiled and run on all cores; `MatMulTInto` and `MatMulTAInto` are the same tiles on the calling goroutine, for attention's one-product-per-head workers. |
| `tokenizer/` | The `Tokenizer` interface (`Encode`, `Decode`, `VocabSize`, `Kind`) and three implementations: `Char` (one id per rune), `Symbol` (one id per whitespace-separated symbol, for sensor/action vocabularies) and `BPE` (GPT-2 style byte-level byte-pair encoding learned from the corpus). `Marshal`/`Unmarshal` wrap any of them in a JSON envelope for the checkpoint. |
| `model/` | `Config`, `Param`, `Linear`, `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` (tied embedding and output head, GPT-2 initialization, inverted dropout) and `Generate`. Every layer has `Forward` and `Backward`. |
| `train/` | `CrossEntropy`, `AdamW` with `ClipGradNorm` and a `Cosine` schedule, gob checkpoints (`Snapshot`, `Save`, `Load`), the `Batcher` and `Stepper` interfaces, and `Run` / `RunStepper`, the training loop. |
| `data/` | `Dataset` (random windows from a token stream; satisfies `train.Batcher`), `Split`, `ReadText`, `Clean` (Gutenberg corpus normalization) and the Project Gutenberg fetcher with a starter horror list. |
| `gpu/` | float32 device-resident tensors and the WGSL kernels a training step needs, through pure-Go WebGPU (Metal on macOS, Vulkan on Linux, no cgo). Every kernel is tested against the float64 CPU code. See [GPU backend](gpu). |
| `gpumodel/` | The whole training step (forward, loss, backward, gradient norm, AdamW) on those kernels, as one GPU submission, and a `train.Stepper` so the trainer can drive it. Checkpoints stay in the CPU format. |

Commands: [`gpt-fetch`](../cli/#gpt-fetch), [`gpt-train`](../cli/#gpt-train),
[`gpt-generate`](../cli/#gpt-generate).

## Quick start (horror prose)

```bash
go run ./cmd/gpt-fetch                                   # -> local_data/gutenberg/*.txt
```

```bash
go run ./cmd/gpt-train -steps 2000 -layers 4 -dim 128    # -> local_data/gpt/model.ckpt
```

```bash
go run ./cmd/gpt-train -backend gpu -tokenizer bpe -bpe-vocab 2000 -layers 6 -dim 384 -heads 6 -ctx 256 -steps 4000
```

```bash
go run ./cmd/gpt-generate -prompt "The old house" -n 400
```

`gpt-train -resume` continues from the checkpoint: model, optimizer moments
and tokenizer are all restored, and the run can switch between `-backend cpu`
and `-backend gpu` at any resume because the checkpoint is the same file
either way. New models train with dropout 0.1 (`-dropout`, stored in the
checkpoint); it only acts during training steps.

`gpt-fetch` saves books already cleaned by `data.Clean` (unwrapped
paragraphs, no italics markers or editorial blocks, ASCII quotes). Use
`gpt-train -clean` for corpora from elsewhere. Cleaning changes the char
vocabulary, so it must be the same for the first run and any resumed run.

## Shapes and conventions

Activations are `[B*T, D]` matrices; row `b*T + t` is token `t` of sequence
`b`. `GPT.Forward(tokens [][]int)` returns logits `[B*T, VocabSize]`.
`train.CrossEntropy` returns the mean loss and `dLogits`; pass that to
`GPT.Backward`, which accumulates into each `Param.Grad`. Call
`GPT.ZeroGrads()` before each forward/backward pair. `GPT.Params()` lists
every parameter in a fixed order that the optimizer, the checkpoint and the
GPU model all rely on.

A GPT starts in inference mode. `train.Run` turns training mode (dropout) on
only for each step's forward and backward.

Gradients are verified against finite differences in `model/gpt_test.go`
and `model/layers_test.go`. Read the [GPT guide](guide) for the math behind
every pass.

## Training on something other than text

The model only sees `[]int`. To train on, say, robot sensor and action
tokens:

1. Build a vocabulary with `tokenizer.NewSymbol([]string{"SENSOR_L_NEAR", "ACTION_TURN_L", ...})`,
   or skip tokenizers and produce ids yourself.
2. Implement `train.Batcher` (`Batch(rng, batchSize, contextLen) (x, y [][]int)`),
   or feed a flat id stream to `data.NewDataset`.
3. `model.New(model.Config{VocabSize: n, ContextLen: ..., NumLayers: ..., NumHeads: ..., EmbedDim: ..., Dropout: 0.1}, rng)`
   and `train.Run(...)`, as `cmd/gpt-train` does.
4. At decision time, call `GPT.Forward` on the recent history and read the
   logits of the last row restricted to the action ids.

The guide's [robot decisions](guide#using-the-model-for-robot-decisions)
section has the code.

## Backends

The training loop body sits behind `train.Stepper`:

```go
type Stepper interface {
    TrainStep(x, y [][]int, lr float64, step int) (loss, gradNorm float64, err error)
    Eval(x, y [][]int) (float64, error)
    Sync(step int) error
}
```

`train.NewCPUStepper` is the float64 implementation; `gpumodel.NewStepper`
is the float32 GPU one. `train.RunStepper` owns the schedule, validation,
checkpoint cadence and the `onStep` callback for either. `Sync` runs before
every validation and checkpoint so the CPU model and optimizer, which
checkpoints are always written from, reflect the backend's state.

| | CPU | GPU |
|---|---|---|
| Arithmetic | float64 | float32 |
| Per step, BPE shape (6 layers, width 384, context 256, batch 16), Apple M5 | 2525 ms | 255 ms |
| Same shape, AMD Radeon 780M (Vulkan) | 7.9 s | 232 ms |
| Checkpoint | float64 gob | the same file |
| Without a GPU | the default | fails with a message |

Losses agree between the two to about the third decimal place; dropout
masks come from different generators.

## Measured performance (CPU)

Per-step cost scales roughly with `BatchSize × ContextLen × NumLayers ×
EmbedDim²` plus the attention term `BatchSize × NumLayers × ContextLen² ×
EmbedDim`. The `gpt-train` defaults (4 layers, width 128, context 128, batch
16) run at about half a second per step on an Apple Silicon laptop. The
guide has a [tuning table](guide#tuning) for the usual symptoms.
