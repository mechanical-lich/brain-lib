# GPT Guide

This guide explains how the from-scratch GPT in `pkg/gpt` works: the data
flow, the math behind every forward and backward pass, how training and
checkpointing fit together, and how to point the same model at a non-text
vocabulary such as robot sensor/action tokens.

It is written for someone who knows Go and basic linear algebra but has not
necessarily implemented a transformer before. Reading it alongside the source
is the intended experience; each section names the file it describes.

## Overview

A GPT is a next-token predictor. Given a sequence of integer token ids
`t_0, t_1, ..., t_{T-1}` it outputs, for every position `t`, a score (logit)
for each possible next token. Training nudges the scores so the real
`t_{t+1}` gets a high score. Generation feeds the model its own predictions
one token at a time.

The implementation has five packages:

| package      | role |
|--------------|------|
| `tensor`     | a row-major `float64` matrix and the handful of ops the model needs |
| `tokenizer`  | text (or symbols) ↔ integer ids |
| `model`      | the network: embeddings, attention, MLP, LayerNorm, and the backward pass for each |
| `train`      | cross-entropy loss, AdamW, LR schedule, gradient clipping, checkpoints, training loop |
| `data`       | token streams, random-window batching, corpus loading, Gutenberg fetch |

There is no autograd. Every layer has a `Forward` that caches what its
`Backward` needs, and a `Backward` that takes the gradient of the loss with
respect to its output and returns the gradient with respect to its input,
accumulating parameter gradients along the way. The chain rule is applied by
hand, layer by layer, in reverse order.

## Key Concepts

### 1. Shapes

Everything is a 2D `tensor.Mat`. Batch and sequence dimensions are flattened
together: an activation for a batch of `B` sequences of length `T` with width
`D` is a `[B*T, D]` matrix, where row `b*T + t` is token `t` of sequence `b`.
Attention, the only layer that cares which rows belong to which sequence,
takes `B` and `T` as explicit arguments and indexes by hand.

```
tokens [B][T] ints
   │  embedding lookup (tok_emb[id] + pos_emb[t])
   ▼
x  [B*T, D]
   │  × NumLayers blocks:  x = x + Attn(LN1(x));  x = x + MLP(LN2(x))
   ▼
h  [B*T, D]   (after final LayerNorm)
   │  tied output head: logits = h @ tok_emb^T
   ▼
logits [B*T, V]
```

### 2. Parameters and gradients

`model.Param` pairs a `Value` matrix with a `Grad` matrix of the same shape
and a `Decay` flag. Backward passes **add** into `Grad`, so you call
`GPT.ZeroGrads()` before each forward/backward pair. `GPT.Params()` returns
every parameter in a fixed, deterministic order; the optimizer and the
checkpoint format both rely on that order.

`Decay` is true for weight matrices and embeddings, false for biases and
LayerNorm gains. AdamW applies weight decay only where `Decay` is set, which
is the usual convention.

### 3. Caching for backward

Each layer stores exactly what its backward pass needs during forward: the
`Linear` keeps its input, `LayerNorm` keeps the normalized input and the
per-row `1/std`, the `MLP` keeps its pre-GELU activations, and attention keeps
the full `qkv` projection plus the softmax weights for every (sequence, head,
query, key) tuple. A second `Forward` overwrites these caches, so a
`Backward` must follow the `Forward` it belongs to.

## The Layers

The notation below uses `@` for matrix multiply, `^T` for transpose, and
`dX` for ∂loss/∂X. Each backward formula is what the code computes.

### Linear (`model/linear.go`)

Forward: `y = x @ W + b`, with `x [n, in]`, `W [in, out]`, `b [1, out]`.

Backward, given `dy [n, out]`:

```
dW += x^T @ dy          tensor.MatMulTA(x, dy)
db += column sums of dy
dx  = dy @ W^T          tensor.MatMulT(dy, W)
```

The three matmul variants in `tensor` (`MatMul`, `MatMulT`, `MatMulTA`)
exist precisely so these three products can be written without the caller
transposing anything. All three compute the output in 4x4 register tiles
(16 independent accumulators) and run in parallel; `MatMulTA` additionally
blocks over `k` so each slab of its operands is reused from cache.

### LayerNorm (`model/layernorm.go`)

Per row: `xhat = (x - mean) / sqrt(var + eps)`, then `y = xhat * g + b`,
where `g` and `b` are learned per-feature vectors.

Backward, per row, given `dy`:

```
dg += dy * xhat                (summed over rows)
db += dy
dxhat = dy * g
dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
```

The last line is the standard compact form of differentiating through the
mean and variance; the two `mean(...)` terms are what make LayerNorm's
gradient non-local within a row.

### GELU (`model/mlp.go`)

The MLP uses the tanh approximation of GELU, the same one GPT-2 uses:

```
gelu(x) = 0.5 x (1 + tanh(c (x + 0.044715 x³)))     c = sqrt(2/π)
```

`geluGrad` is its closed-form derivative. `TestGeluGrad` checks it against
finite differences.

### MLP (`model/mlp.go`)

`y = Proj(gelu(FC(x)))` where `FC` widens `D → 4D` and `Proj` narrows back.
Backward is just the chain: `Proj.Backward`, multiply elementwise by
`geluGrad(pre)`, then `FC.Backward`.

### Causal self-attention (`model/attention.go`)

This is the layer worth reading slowly. For each sequence `b` and head `h`:

1. A single `Linear` maps `x [B*T, D]` to `qkv [B*T, 3D]`. Columns
   `[0, D)` are queries, `[D, 2D)` keys, `[2D, 3D)` values. Within each of
   those, head `h` owns columns `[h*hs, (h+1)*hs)` where `hs = D / NumHeads`.
2. For query position `t`, scores against every key position `s ≤ t`:
   `score[t, s] = (q_t · k_s) / sqrt(hs)`. Positions `s > t` are never
   computed, which is the causal mask: a token cannot see the future.
3. Softmax over `s` gives attention weights `att[t, s]`, cached for backward.
4. The head's output is the weighted sum of values: `out_t = Σ_s att[t, s] v_s`.
5. All heads' outputs are concatenated back into `[B*T, D]` and passed
   through the output projection `Proj`.

Backward reverses this. Given `dOut [B*T, D]` from `Proj.Backward`:

```
dAtt[t, s] = dOut_t · v_s
dV_s      += att[t, s] * dOut_t
dScore[t, s] = att[t, s] * (dAtt[t, s] - Σ_s' att[t, s'] dAtt[t, s'])   (softmax backward)
dQ_t += dScore[t, s] * k_s / sqrt(hs)
dK_s += dScore[t, s] * q_t / sqrt(hs)
```

`dQ`, `dK`, `dV` are written into the matching column ranges of a
`dqkv [B*T, 3D]` matrix, which `QKV.Backward` turns into `dx`. Because the
mask was applied by simply not computing `s > t`, those `att` entries are
zero and contribute nothing, so the mask needs no special handling in
backward.

The `(b, h)` pairs are independent, so both passes run them in parallel with
`tensor.Parallel`.

### Block (`model/block.go`)

Pre-norm residual block:

```
h   = x + Drop(Attn(LN1(x), B, T))
out = h + Drop(MLP(LN2(h)))
```

Backward applies the residual rule (gradient flows straight through the
`+` and also through the branch):

```
dh = dy + LN2.Backward(MLP.Backward(M2 * dy))
dx = dh + LN1.Backward(Attn.Backward(M1 * dh))
```

`Drop` is inverted dropout (`model/dropout.go`), GPT-2 style: in training
mode each element of the branch output (after its `Proj`, before the
residual add) is zeroed with probability `Config.Dropout` and the survivors
are scaled by `1/(1-p)`, so the expected value is unchanged and inference
needs no rescaling. Forward stores the mask `M` (entries `0` or `1/(1-p)`)
and Backward multiplies the incoming gradient by the same mask. The same
dropout is applied once more to the embedding sum before the first block.

Dropout only runs when the GPT is in training mode (`SetTraining(true)`).
A new model starts in inference mode, so `Forward` in `Generate`,
`Evaluate` and the tests is deterministic; `train.Run` turns training mode
on just for each step's forward/backward. Masks come from the rng passed to
`model.New`; `SetRNG` replaces it (`cmd/gpt-train` uses it to seed a model
restored from a checkpoint). `TestGradientCheckDropout` reseeds the rng
before every `Forward` so the masks are fixed and finite differences still
apply.

### GPT (`model/gpt.go`)

Forward builds `x` by adding the token embedding row for each id to the
positional embedding row for its position, runs the blocks, applies the final
LayerNorm, and computes logits with the **tied** output head
`logits = h @ tok_emb^T`. Tying means the same matrix maps ids to vectors on
the way in and vectors to scores on the way out; it halves the embedding
parameter count and is what GPT-2 does.

Backward, given `dLogits [B*T, V]`:

```
tok_emb.Grad += dLogits^T @ h       (output-head share of the tied gradient)
dh = dLogits @ tok_emb
... LNF.Backward, blocks in reverse ...
tok_emb.Grad[id]  += dx[row]        (embedding-lookup share, scatter-add)
pos_emb.Grad[t]   += dx[row]
```

Initialization follows GPT-2: all weights `N(0, 0.02)`, except the two
output projections in each block (attention `Proj` and MLP `Proj`) which use
`0.02 / sqrt(2 * NumLayers)` so the residual stream's variance does not grow
with depth.

## Training

### Loss (`train/loss.go`)

`CrossEntropy(logits, targets)` returns the mean negative log-likelihood and
`dLogits` in one pass. Per row the gradient is simply
`softmax(logits) - onehot(target)`, divided by the number of rows because the
loss is a mean. The log-sum-exp is computed with the row max subtracted so
large logits do not overflow.

### Optimizer (`train/adamw.go`)

`AdamW` is Adam with decoupled weight decay. For each parameter element:

```
m = β1 m + (1-β1) g
v = β2 v + (1-β2) g²
w -= lr * (m / (1-β1^t)) / (sqrt(v / (1-β2^t)) + eps)
w -= lr * weightDecay * w        (only if Param.Decay)
```

Defaults are `β1 = 0.9`, `β2 = 0.95`, `eps = 1e-8`, `weightDecay = 0.1`.

Two helpers go with it:

- `ClipGradNorm(params, maxNorm)` rescales all gradients so their global L2
  norm is at most `maxNorm`, and returns the pre-clip norm. Watching that
  number is the easiest way to spot a diverging run.
- `Cosine(warmup, total, maxLR, minLR)` returns a `Schedule`: linear warmup
  to `maxLR`, cosine decay to `minLR` at `total`, flat afterwards.

### The loop (`train/trainer.go`)

`train.Run` is the whole training loop. One step is:

```go
opt.Config.LR = schedule(step)
x, y := batcher.Batch(rng, batchSize, contextLen)   // y[i][t] is the target for x[i][t]
g.ZeroGrads()
g.SetTraining(true)                                  // dropout on
loss, dLogits := train.CrossEntropy(g.Forward(x), flatten(y))
g.Backward(dLogits)
g.SetTraining(false)                                 // eval and onStep see no dropout
norm := train.ClipGradNorm(g.Params(), gradClip)
opt.Step()
```

Around that it runs validation every `EvalEvery` steps, saves a checkpoint
every `CheckpointEvery` steps and at the end, and calls your `onStep`
callback with a `Metrics` struct. `Run` takes a `Batcher` interface rather
than a `data.Dataset`, which is the seam for non-text training (see below).

### Checkpoints (`train/checkpoint.go`)

A `Checkpoint` holds the `model.Config`, the step count, every parameter by
name, the optimizer's moment estimates and step, and an opaque `Tokenizer`
byte slice. `Config` includes `Dropout`; checkpoints written before that field
existed load with `Dropout` 0. It is written with `encoding/gob` to a temp file and renamed into
place, so a checkpoint on disk is never half-written and you can sample from
it while training continues.

Restoring is two calls: `ck.Model()` rebuilds the network from the saved
config and copies the weights in; `ck.OptimizerFor(g, cfg)` rebuilds AdamW
with the saved moments so a resumed run continues exactly where it left off
(`TestCheckpointRoundTrip` verifies a resumed run is bit-for-bit identical).

## Generation (`model/generate.go`)

`GPT.Generate` is deliberately naive: each new token re-runs a full forward
pass over the last `ContextLen` tokens and samples from the final row of
logits. There is no key/value cache, so generation is O(n²) in context length.
For models this size it does not matter.

Sampling options: `Temperature <= 0` is greedy argmax; otherwise logits are
divided by the temperature, optionally truncated to the `TopK` largest, and
sampled from the softmax.

## Cleaning the corpus (`data/clean.go`)

A character-level model spends capacity on every distinct rune and every
recurring pattern, so typesetting noise in the source text is learned as if
it were prose. Project Gutenberg files hard-wrap lines at about 70 columns,
mark italics with `_underscores_`, carry `[Illustration: ...]`,
`[Footnote N: ...]`, `[Transcriber's Note: ...]` and `[Source: ...]` blocks, mix curly and
straight quotes, and occasionally contain mojibake such as `Ã¤` for `ä`.

`data.Clean(text, data.DefaultCleanOptions())` fixes this, in order:

1. normalize CRLF to LF;
2. drop bracketed editorial blocks (matched across lines) and `[1]`-style
   footnote references;
3. strip italic underscores, keeping the words;
4. turn curly quotes and the acute accent used as an apostrophe into ASCII
   quotes, en/em dashes into `--` (`NormalizePunctuation`);
5. repair two-rune mojibake and delete leftover mojibake runes, keeping real
   accented letters and `æ`/`œ`;
6. unwrap hard line breaks so each paragraph is one line; a blank line is a
   paragraph break and an indented line (verse, epigraph) stays on its own
   line with a four-space indent;
7. collapse runs of spaces and of more than two newlines.

`DropHeader` (off by default) also removes the title/author lines before the
first paragraph longer than 200 characters.

`gpt-fetch` applies the defaults to every downloaded book, and
`gpt-train -clean` applies them to any corpus before tokenizing. Because the
`Char` tokenizer's vocabulary is the set of runes in the training text,
cleaning changes the vocabulary: decide on it before the first run and use
the same setting for every resumed run and for any text added later.
`gpt-train` logs the vocabulary size and warns when the corpus contains runes
the checkpoint's tokenizer has never seen.

## Tokenizers (`tokenizer/`)

The model never sees text. `Tokenizer` is a four-method interface
(`Encode`, `Decode`, `VocabSize`, `Kind`), and `Marshal`/`Unmarshal` wrap any
implementation in a JSON envelope tagged with its kind so a checkpoint can
restore the right one.

- `Char`: one id per distinct rune in the corpus it was built from, sorted so
  ids are stable. Id 0 is reserved for unknown runes.
- `Symbol`: one id per whitespace-separated symbol from a fixed list. Id 0 is
  unknown. This is the tokenizer for discrete action/sensor vocabularies.
- `BPE`: byte-level byte-pair encoding as in GPT-2 (`gpt-train -tokenizer bpe
  -bpe-vocab 2000`). Ids 0-255 are the byte values, so any text encodes and
  there is no unknown id (0 is the NUL byte). `NewBPE(corpus, vocabSize)`
  learns `vocabSize - 256` merges, each adding one id for the concatenation of
  two existing tokens. The checkpoint stores the ordered merge list, and
  `-resume` restores it rather than relearning.

How `BPE` learns and encodes:

1. **Pre-tokenize.** Text is split into words with GPT-2's rule: an optional
   leading space plus a run of letters, a run of digits, or a run of other
   non-space characters; contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`,
   `'d`) split off; a newline before text is its own word. So `" the"`,
   `" horror"`, `","` and `"\n"` are words, and a merge never crosses a word
   boundary.
2. **Learn (Sennrich).** Count each distinct word once with its frequency,
   then repeat: find the most frequent adjacent pair across the dictionary
   (weighted by word count; ties go to the smaller ids), record it as the next
   merge, and rewrite only the words that contain it, updating pair counts
   incrementally. On the 2.9 MB Lovecraft corpus 1744 merges take about 0.3 s.
3. **Encode.** For each word (cached), start from its bytes and repeatedly
   apply the lowest-ranked merge present until none applies. `Decode`
   concatenates the tokens' bytes; a generated sample can stop partway
   through a multibyte rune.

`Vocab()` returns every id's token as a string for inspection. Early merges
are pieces like `" t"`, `"he"`, `" the"`; by id 2000 they are whole words and
stems such as `" slow"` and `" remem"`. With 2000 ids, Lovecraft averages
about 3.2 bytes per token, so a 256-token context covers roughly 830
characters instead of 256.

**Losses are per token.** Cross-entropy is reported in nats per token, so a
BPE run's loss is not comparable with a character-level run's. To compare,
convert both to nats per character: multiply a BPE loss by tokens/characters
from the `corpus` log line (`1 / bytes_per_token` for mostly-ASCII text). A
fresh BPE model starts near `ln(VocabSize)`, about 7.6 for 2000 ids.

Another tokenizer means implementing the interface and adding a case to the
`switch` in `Marshal` and `Unmarshal`.

## Using the Model for Robot Decisions

Because `train.Run` only needs a `Batcher` and the model only needs ids,
retargeting from prose to behaviour is mostly a data problem. A sketch:

```go
// 1. Vocabulary: every sensor reading and every action is one symbol.
tok := tokenizer.NewSymbol([]string{
    "SEE_FOOD_L", "SEE_FOOD_R", "SEE_NONE", "DANGER_NEAR", "DANGER_NONE",
    "ACT_LEFT", "ACT_RIGHT", "ACT_FORWARD", "ACT_STOP",
})

// 2. Episodes become alternating sensor/action token streams, e.g.
//    "SEE_FOOD_L DANGER_NONE ACT_LEFT SEE_FOOD_R DANGER_NONE ACT_RIGHT ..."
stream := tok.Encode(strings.Join(episodes, " "))
trainSet, valSet := data.Split(stream, 0.1)

// 3. A small model; context covers a few dozen sense/act cycles.
cfg := model.Config{VocabSize: tok.VocabSize(), ContextLen: 64, NumLayers: 2, NumHeads: 2, EmbedDim: 32}
g := model.New(cfg, rng)
opt := train.NewAdamW(g.Params(), train.DefaultAdamW())
train.Run(g, opt, trainSet, valSet, train.Options{Steps: 2000, BatchSize: 32, GradClip: 1}, rng, nil)

// 4. At decision time: feed the recent history, read only the action logits.
logits := g.Forward([][]int{history})
last := logits.Row(logits.Rows - 1)
best := actionIDs[0]
for _, id := range actionIDs {
    if last[id] > last[best] { best = id }
}
```

Two refinements worth knowing about: the loss currently trains the model to
predict sensor tokens as well as actions, which is harmless but wasted
capacity, so a custom `Batcher` could mark sensor positions to be ignored
(an `ignoreIndex` in `CrossEntropy` is a small change); and if you write your
own `Batcher` that samples whole episodes you can skip `data.Dataset`
entirely. Nothing in `train` or `model` would need to change.

## Tuning

Per-step cost scales roughly with `BatchSize × ContextLen × (NumLayers ×
EmbedDim²)` plus the attention term `BatchSize × NumLayers × ContextLen² ×
EmbedDim`. The defaults in `cmd/gpt-train` (4 layers, width 128, context 128,
batch 16) run at about half a second per step on an Apple Silicon laptop.

| symptom | what to try |
|---------|-------------|
| loss stuck near `ln(VocabSize)` | learning rate too low, or warmup too long; try `-lr 3e-3` |
| loss spikes or becomes NaN | learning rate too high; also check `gnorm` in the log and lower `-clip` |
| train loss falls, val loss rises | model is memorizing; add data, shrink the model, or raise `-wd` |
| samples are plausible letters but not words | just needs more steps; character models take a few thousand |
| far too slow | shrink `-ctx` first (attention is quadratic in it), then `-dim` |

## Debugging Tips

### 1. Trust the gradient checks

`model/gpt_test.go` perturbs random parameters of a tiny model by `±1e-5`
and compares the finite-difference slope with the analytic gradient, and
`model/layers_test.go` does the same for the input gradient of every layer
in isolation. If you change any `Backward`, run `go test ./pkg/gpt/model`
first; a wrong gradient almost always shows up as a mismatch of several
orders of magnitude, not a subtle one.

### 2. Check the causal mask

`TestCausalMask` changes tokens at positions 3 and 4 and asserts logits at
positions 0 to 2 are unchanged to `1e-12`. Any change to attention indexing
should keep that test passing.

### 3. Overfit something tiny

`TestRunOverfitsPattern` trains a 2-layer model on a six-token repeating
pattern and expects loss below 0.1 in 250 steps, then expects greedy
generation to reproduce the pattern. If a real run is not learning, write a
version of this test with your own vocabulary before suspecting the data.

### 4. Watch the right numbers

The training log prints `loss`, `lr`, and `gnorm` every ten steps. Gradient
norm settling into a steady range while loss decreases is healthy. A norm
that keeps climbing precedes a divergence by a few hundred steps.

## Extension Points

- **Dropout**: implemented (`Config.Dropout`, `cmd/gpt-train -dropout`,
  default 0.1 for new models). Attention-weight dropout (on the softmax
  probabilities, as GPT-2 also does) is not; it would need the mask kept per
  `B*H*T*T` entry in `CausalSelfAttention` and applied to `dAtt` in Backward.
- **KV cache**: `Generate` could keep each block's keys and values and only
  compute the new position; `CausalSelfAttention` would need an incremental
  `Forward`.
- **BPE tokenizer**: implemented (`tokenizer.BPE`, `cmd/gpt-train -tokenizer bpe`).
- **Ignore index in the loss**: let `CrossEntropy` skip targets of `-1` so
  only chosen positions (for example actions) contribute.
- **float32 or SIMD**: `tensor` is the only place that would change.

## Summary

The model is a stack of pre-norm transformer blocks over tied token and
position embeddings, written so that every forward pass caches what its
hand-written backward pass needs. Loss, optimizer, schedule, clipping,
checkpoints and the loop live in `train`, driven by a `Batcher` interface
that makes the text pipeline in `data` one option among many. The tests pin
the gradients to finite differences, so the backward passes can be trusted
when the architecture is extended.
