---
title: Command-line tools
nav_order: 8
permalink: /cli/
---

# Command-line tools
{: .no_toc }

Every directory under `cmd/` is one tool. Run any of them with
`go run ./cmd/<name> ...`, or build them all into `bin/` with `make build`.
{: .fs-6 .fw-300 }

1. TOC
{:toc}

| Tool | Package it drives | Purpose |
|---|---|---|
| [`validator`](#validator) | `pkg/brain/validation` | Check a brain file; optionally repair it. |
| [`brain-trainer`](#brain-trainer) | `pkg/ann` | Train or query a feed-forward network. |
| [`gpt-fetch`](#gpt-fetch) | `pkg/gpt/data` | Download a Project Gutenberg corpus. |
| [`gpt-train`](#gpt-train) | `pkg/gpt/*` | Train a GPT on a corpus, on the CPU or a GPU. |
| [`gpt-generate`](#gpt-generate) | `pkg/gpt/*` | Sample text from a checkpoint. |

## validator

```bash
go run ./cmd/validator validate <brain.json>
```

```bash
go run ./cmd/validator repair <brain.json>
```

`validate` prints every complaint with its type and the ID involved, or
"Brain validation succeeded". `repair` runs the fixes, prints the repair
report, warns if any synapse was deleted rather than moved, and writes the
result to `<brain.json>.fixed`, leaving the original untouched. The
complaint types and repairs are on the [validation](../brain/validation)
page.

The validator takes positional arguments, not flags.

## brain-trainer

Trains a `pkg/ann` network from a JSON examples file, or loads one and
answers queries interactively.

```bash
go run ./cmd/brain-trainer -mode train -data local_data/training_data.json -ann ann.json -epochs 1000 -lr 0.1
```

```bash
go run ./cmd/brain-trainer -mode test -ann ann.json
```

| Flag | Default | Meaning |
|---|---|---|
| `-mode` | `train` | `train` or `test`. |
| `-ann` | `ann.json` | Network file. In `train` mode an existing file is loaded and trained further; otherwise a new network is created. |
| `-data` | | Training examples (required for `train`). A JSON array of `{"Inputs": [...], "Outputs": [...]}`. |
| `-epochs` | `1000` | Training epochs. |
| `-lr` | `0.1` | Learning rate. |
| `-hidden` | `1` | Number of hidden layers (new networks only). |
| `-hiddenSize` | `0` | Neurons per hidden layer; `0` means inputs + outputs. |

The network shape is inferred from the first example. In `test` mode, type
comma-separated inputs at the prompt; `exit` or an empty line quits.

## gpt-fetch

Downloads public-domain texts from Project Gutenberg, strips the license
header and footer, runs `data.Clean` with default options, and writes
`<id>.txt` files. Existing files are left alone.

```bash
go run ./cmd/gpt-fetch
```

```bash
go run ./cmd/gpt-fetch -ids 345,84 -out ./corpus
```

```bash
go run ./cmd/gpt-fetch -list
```

| Flag | Default | Meaning |
|---|---|---|
| `-out` | `local_data/gutenberg` | Output directory. |
| `-ids` | | Comma-separated Gutenberg ids. Default is the built-in horror list (`data.HorrorBooks`: Lovecraft, Stoker, Shelley, Le Fanu, Chambers, James, Poe). |
| `-list` | `false` | Print the default list and exit. |

Exits non-zero if any download failed.

## gpt-train

Trains a GPT on a text corpus and writes a checkpoint. Logs go to stderr as
`slog` text: a `step` line every ten steps with loss, learning rate, gradient
norm and ms per step, an `eval` line with validation loss at each
validation, and a text sample after each validation.

```bash
go run ./cmd/gpt-train -data local_data/gutenberg -steps 2000
```

```bash
go run ./cmd/gpt-train -resume -steps 4000
```

```bash
go run ./cmd/gpt-train -backend gpu -tokenizer bpe -bpe-vocab 2000 -layers 6 -heads 6 -dim 384 -ctx 256 -steps 4000
```

**Data and tokenizer**

| Flag | Default | Meaning |
|---|---|---|
| `-data` | `local_data/gutenberg` | A `.txt` file or a directory of `.txt` files (read in name order, joined with blank lines). |
| `-out` | `local_data/gpt/model.ckpt` | Checkpoint path. |
| `-resume` | `false` | Resume from `-out` if it exists. Model shape, `-tokenizer`, `-bpe-vocab` and `-dropout` are then ignored; the checkpoint's are used. |
| `-tokenizer` | `char` | `char`, `bpe` or `symbol` for a new model. `symbol` is the mode for a sensor/action vocabulary. |
| `-bpe-vocab` | `2000` | Vocabulary size to learn with `-tokenizer bpe` (256 byte ids plus merges). |
| `-clean` | `false` | Run `data.Clean` on the corpus before tokenizing. Use the same setting when resuming; cleaning changes the char vocabulary. |

**Model** (new models only)

| Flag | Default | Meaning |
|---|---|---|
| `-layers` | `4` | Transformer blocks. |
| `-heads` | `4` | Attention heads. |
| `-dim` | `128` | Embedding dimension. Must be divisible by `-heads`. |
| `-ctx` | `128` | Context length. |
| `-dropout` | `0.1` | Dropout probability in `[0, 1)`. Stored in the checkpoint. |

**Optimization**

| Flag | Default | Meaning |
|---|---|---|
| `-steps` | `1000` | Total optimizer steps, including those already done on resume. |
| `-batch` | `16` | Batch size. |
| `-lr` | `1e-3` | Peak learning rate. |
| `-min-lr` | `1e-4` | Final learning rate of the cosine schedule. |
| `-warmup` | `100` | Linear warmup steps. |
| `-wd` | `0.1` | Weight decay (on weight matrices and embeddings only). |
| `-clip` | `1.0` | Gradient clip norm; `0` disables. |
| `-seed` | `1` | RNG seed. |

**Evaluation and checkpoints**

| Flag | Default | Meaning |
|---|---|---|
| `-val-frac` | `0.1` | Fraction of the corpus held out for validation. |
| `-eval-every` | `100` | Validate every N steps; `0` disables. |
| `-eval-batches` | `8` | Batches averaged per validation. |
| `-ckpt-every` | `100` | Checkpoint every N steps (and at the end); `0` disables. |
| `-sample` | `200` | Tokens to sample after each validation; `0` disables. |

**Backend**

| Flag | Default | Meaning |
|---|---|---|
| `-backend` | `cpu` | `cpu` (float64) or `gpu` (float32 via pure-Go WebGPU). Fails with a message when no hardware GPU adapter exists. |

Losses are per token, so a BPE run's loss is not comparable with a
character-level run's. The `corpus` log line reports `bytes_per_token`;
multiply a BPE loss by `1 / bytes_per_token` for nats per character. A
fresh model starts near `ln(VocabSize)`.

## gpt-generate

Samples text from a checkpoint written by `gpt-train`. The checkpoint must
carry a tokenizer (every `gpt-train` checkpoint does).

```bash
go run ./cmd/gpt-generate -prompt "The old house" -n 400 -temp 0.8
```

```bash
go run ./cmd/gpt-generate -backend gpu -prompt "The old house" -n 400
```

| Flag | Default | Meaning |
|---|---|---|
| `-ckpt` | `local_data/gpt/model.ckpt` | Checkpoint path. |
| `-prompt` | `"\n"` | Prompt text. |
| `-n` | `300` | Tokens to generate. |
| `-temp` | `0.8` | Sampling temperature; `0` is greedy. |
| `-topk` | `40` | Top-k cutoff; `0` disables. |
| `-seed` | `0` | RNG seed; `0` means time-based. |
| `-backend` | `cpu` | `cpu` or `gpu`. The GPU runs the forward passes; sampling is identical. |

There is no KV cache: each new token re-runs a forward pass over the trailing
context window. For models this size that is fine on the CPU; on the GPU a
200-token sample at the BPE shape takes about a second.
