# gpt

A minimal GPT (decoder-only transformer) written from scratch in pure Go:
hand-rolled matrix ops and hand-written backward passes, no autograd library.
It is small enough to read end to end and is meant as a base that can be
retargeted from prose to any discrete token stream.

## Layout

| package      | purpose |
|--------------|---------|
| `tensor/`    | row-major float64 `Mat` and the few ops a transformer needs (`MatMul`, `MatMulT`, `MatMulTA`, broadcast add, ...). The matmuls are cache-tiled and parallel; `MatMulTInto` and `MatMulTAInto` are the same tiles run on the calling goroutine, for one-product-per-worker callers such as attention. |
| `tokenizer/` | `Tokenizer` interface with `Char` (one token per rune), `Symbol` (one token per whitespace-separated symbol) and `BPE` (GPT-2 style byte-level byte-pair encoding learned from the corpus; `gpt-train -tokenizer bpe -bpe-vocab 2000`). JSON `Marshal`/`Unmarshal`. Losses are per token, so a BPE run's loss is not comparable with a character-level run's. |
| `model/`     | `Config`, `Param`, `Linear`, `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPT` (tied embedding / output head), and `Generate`. Every layer has `Forward` and `Backward`. |
| `train/`     | `CrossEntropy`, `AdamW` + `ClipGradNorm` + `Cosine` LR schedule, gob checkpoints (`Snapshot`/`Save`/`Load`), and `Run`, a training loop over any `Batcher`. |
| `data/`      | `Dataset` (random windows from a token stream, satisfies `train.Batcher`), `Split`, `ReadText`, `Clean` (corpus normalization), and the Project Gutenberg fetcher. |
| `gpumodel/` | The training step (forward, loss, backward, AdamW) on the `gpu/` kernels, verified against `model` + `train`. Phase two of the GPU backend. |
| `gpu/`       | float32 GPU tensors and the WGSL kernels a training step needs (matmuls, softmax, LayerNorm, GELU, embeddings, cross-entropy, dropout, AdamW) through pure-Go WebGPU, each tested against this package's CPU code. Additive and not yet wired into training; see `gpu/readme.md`. |

Commands live in `cmd/gpt-fetch`, `cmd/gpt-train`, `cmd/gpt-generate`.

## Quick start (horror prose)

```bash
go run ./cmd/gpt-fetch                                  # -> local_data/gutenberg/*.txt
go run ./cmd/gpt-train -steps 2000 -layers 4 -dim 128   # -> local_data/gpt/model.ckpt
go run ./cmd/gpt-generate -prompt "The old house" -n 400
```

`gpt-train -resume` continues from the checkpoint (model, optimizer moments
and tokenizer are all restored). Training is CPU-only float64, so keep models
small: a 4-layer, 128-wide model at context 128 is a reasonable laptop run.
New models train with dropout 0.1 (`-dropout`, stored in the checkpoint and
ignored on `-resume`); it only acts during training steps.

`gpt-fetch` saves books already cleaned by `data.Clean` (unwrapped
paragraphs, no italics markers or editorial blocks, ASCII quotes); use
`gpt-train -clean` for corpora from elsewhere. Cleaning changes the char
vocabulary, so it must be the same for the first run and any `-resume` run.

## Using it for something other than text

The model only sees `[]int`. To train on, say, robot sensor/action tokens:

1. Build a vocabulary with `tokenizer.NewSymbol([]string{"SENSOR_L_NEAR", "ACTION_TURN_L", ...})`,
   or skip tokenizers entirely and produce ids yourself.
2. Implement `train.Batcher` (`Batch(rng, batchSize, contextLen) (x, y [][]int)`),
   or feed a flat id stream to `data.NewDataset`.
3. `model.New(model.Config{VocabSize: n, ContextLen: ..., NumLayers: ..., NumHeads: ..., EmbedDim: ..., Dropout: 0.1}, rng)`
   and `train.Run(...)` as `cmd/gpt-train` does.
4. At decision time, call `GPT.Forward` on the recent sensor/action history
   and read the logits of the last row restricted to action ids.

## Shapes and conventions

Activations are `[B*T, D]` matrices; row `b*T + t` is token `t` of sequence
`b`. `GPT.Forward(tokens [][]int)` returns logits `[B*T, VocabSize]`.
`train.CrossEntropy` returns the loss and `dLogits`; pass that to
`GPT.Backward`, which accumulates into each `Param.Grad`. Call
`GPT.ZeroGrads()` before each forward/backward pair.

Gradients are verified against finite differences in `model/gpt_test.go`.

For a walkthrough of the math behind each forward/backward pass, the training
loop and the robot-token path, see [docs/GPT_GUIDE.md](../../docs/GPT_GUIDE.md).
