# gpumodel

The GPT training step from `pkg/gpt/model` running on a GPU, on the float32
tensor layer and kernels in `pkg/gpt/gpu` (pure-Go WebGPU, no cgo). This is
phase two of the GPU backend. The CPU path in `model` and `train` is
unchanged and still the default; nothing in `cmd/` uses this package yet.

## What it provides

| | |
|---|---|
| `New(dev, cfg)` | Allocates weights, gradients and AdamW moments on the device, in the same order and under the same names as `model.GPT.Params()`. |
| `LoadFrom(g)`, `LoadOptimizer(opt)` | Upload a CPU model's weights and a `train.AdamW`'s moments. Random initialization stays on the CPU: build a `model.GPT`, then load it. |
| `StoreTo(g, opt, step)` | Download weights and moments back into the CPU types, so `train.Snapshot` writes the same float64 gob checkpoint as a CPU run. |
| `TrainStep(x, y, adamw, maxNorm, step)` | Forward with dropout, mean cross-entropy, backward, global gradient norm with optional clipping, and the AdamW update of every parameter, recorded into **one** `CommandBatch` and submitted once. Reads back two floats: the loss and the squared gradient norm. |
| `Eval(x, y)` | Mean cross-entropy without dropout and without touching the parameters; one submission, one float back. |
| `Forward(tokens)` | The logits `[B*T, V]` as a `tensor.Mat`, the same contract as `model.GPT.Forward`; downloads the whole matrix, so it is for evaluation, sampling and tests. |
| `Backward(dLogits)`, `Grads()` | The backward pass on its own, for checking against the CPU. `TrainStep` does not use them. |
| `SetTraining`, `SetSeed`, `Close` | Dropout on/off, the dropout seed, and releasing every buffer. |

## Layout

Everything mirrors the CPU model so the two can be compared element by
element:

- Activations are `[B*T, D]` with row `b*T + t`. `qkv` is `[B*T, 3D]` with
  the q, k and v heads side by side, and attention weights are `[B*H*T, T]`
  with row `(b*H + h)*T + t`. The per-head products are single batched
  `GEMM` dispatches over that layout; nothing is gathered or transposed.
- Every forward activation a backward pass needs is kept on the device:
  per layer the LayerNorm `xhat` and `1/std`, `qkv`, the softmaxed scores,
  the concatenated head outputs, the residual `h`, the MLP pre-activation
  and `gelu` output, the block output, and the two dropout masks. They are
  allocated once per batch shape and reused across steps. The BPE run's
  shape (6 layers, width 384, context 256, batch 16, vocabulary 2000) holds
  about 900 MB of activations.
- The backward pass rotates three `[B*T, D]` buffers for `dy`, `dh` and
  `dx`, plus scratch for the dropout-masked gradient, the LayerNorm input
  gradient, the attention output gradient, `dqkv`, the MLP gradient and
  `dScores`.
- The tied output head is `logits = hF @ tokEmb^T`; its backward adds
  `dLogits^T @ hF` into the token embedding gradient, and the embedding
  gradient from the lookups is added by `ScatterAdd` at the end, as on the
  CPU.
- Dropout masks are a pure function of (seed, site, step): the embedding
  sum and each block's attention and MLP outputs are separate sites, so a
  step draws independent masks at every site and a fresh set each step
  without RNG state on the device.

## What one TrainStep submits

```
Zero every gradient
Gather tok_emb[ids] ; Gather pos_emb[t] (+=) ; Dropout
for each block:
  LayerNorm ; MatMul qkv ; AddRow ; GEMM scores=q k^T ; Softmax(causal) ; GEMM out=att v
  MatMul proj ; AddRow ; Dropout ; Add residual
  LayerNorm ; MatMul fc ; AddRow ; GELU ; MatMul proj ; AddRow ; Dropout ; Add residual
LayerNorm ; MatMulT logits = hF tok_emb^T
CrossEntropy (per-row losses, dLogits in place) ; Sum -> loss
MatMulTA tok_emb.grad += dLogits^T hF ; MatMul dhF = dLogits tok_emb ; LayerNormBackward
for each block, reversed:
  [Mul mask] ; MatMulTA/ColSum/MatMulT (mlp proj) ; GELUBackward ; MatMulTA/ColSum/MatMulT (fc) ; LayerNormBackward ; Add
  [Mul mask] ; MatMulTA/ColSum/MatMulT (attn proj)
  GEMM dScores=dOut v^T ; GEMM dV=att^T dOut ; SoftmaxBackward ; GEMM dQ=dS k ; GEMM dK=dS^T q
  MatMulTA/ColSum/MatMulT (qkv) ; LayerNormBackward ; Add
[Mul mask] ; ScatterAdd tok_emb.grad ; ScatterAdd pos_emb.grad
Sum of squares over every gradient -> normSq
AdamW for every parameter (clipping folded in from normSq)
Read loss, normSq
```

At the BPE shape that is about 330 dispatches in one submission.

## Verification

`gpumodel_test.go` builds a CPU `model.GPT` with a fixed seed (weights
scaled up so the nonlinearities are exercised), loads it onto the GPU and
compares, at two shapes including one with four heads, a ragged vocabulary
and a context that is not a multiple of the row block:

- forward logits, to `2e-4` relative;
- every parameter gradient from the same `dLogits`, to `2e-3` relative;
- five `TrainStep` calls against `train.CrossEntropy` + `ClipGradNorm` +
  `train.AdamW` with identical hyperparameters, with and without clipping:
  loss to `0.2%`, gradient norm to `0.5%`, and the weights afterwards.
  Adam normalizes each element's update to about `lr`, so an element whose
  float32 gradient has the opposite sign from the float64 one moves `2*lr`
  the other way; the test bounds the worst element by `3*lr*steps` and the
  mean by `lr/5`;
- a 250-step run on a repeating pattern with dropout 0.1 on, which must
  reach loss 0.15, and an `Eval` under that afterwards;
- `StoreTo` into a fresh CPU model and optimizer, whose forward then
  matches, and `LoadFrom` + `LoadOptimizer` back onto a second GPU model,
  whose next step matches the first one to `1e-5`.

Without a GPU adapter every test skips with a message, so `go test ./...`
passes on any machine. `BRAIN_GPU_DISABLE=1` forces that path.

## Measured

Apple M5, Metal, Go 1.26, BPE run shape (6 layers, width 384, 6 heads,
vocabulary 2000, context 256, batch 16, dropout 0.1), five iterations:

| | ms per training step |
|---|---|
| CPU, `model` + `train` (float64, all cores) | 2525 |
| GPU, `gpumodel.TrainStep`, phase two | 308 |
| GPU, after the fusion pass (below) | **255** |

9.9x on this machine.

### Where a step goes

`TestProfileStep` (run with `-v`) submits the step as named phases and
prints their GPU time; `Model.Profile` is the API behind it. On the M5 at
this shape, before any fusion, the matmul phases were about three quarters
of the step and ran at the kernel benchmarks' rate, the attention products
about a tenth, and the row and elementwise kernels the rest. That ruled
out a hidden matmul problem and pointed at three things, done in order:

| change | step on the M5 |
|---|---|
| phase two baseline | 308 ms |
| fused epilogues (bias, GELU, dropout+residual, GELU-grad), `MulInto`, residual in LayerNorm backward | 304 |
| lane row kernels for LayerNorm and softmax (latency-bound, not bandwidth-bound) | 290 |
| 128-row matmul tiles for M >= 1024 | 259 |
| split-K for the weight gradients | 255 |

The elementwise fusions bought little here because the M5 has 200 GB/s.
They matter on the AMD Radeon 780M server (Vulkan, 67 GB/s), where a real
`gpt-train -backend gpu` run at this shape reported 535 ms per step before
this pass and 232 ms after it, 2.3x, against 7.9 s per step on that
machine's CPU. The phase-one kernels ran about 15% faster on the AMD
Radeon 780M server than here, and that server's CPU step is 7.9 s, so the
GPU step there should land near 30x. Not yet measured there; the cross
compiled test binary is the way to do it:

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go test -c -o bin/linux-amd64/gpumodel.test ./pkg/gpt/gpumodel/
# on the server:
./gpumodel.test -test.v
./gpumodel.test -test.run XXX -test.bench TrainStep -test.benchtime 5x
```

## In the trainer

`Stepper` (stepper.go) adapts a `Model` to `train.Stepper`, so
`train.RunStepper` drives the GPU with the same schedule, validation,
checkpoint cadence and logging as the CPU. `Sync` downloads weights and
moments into the CPU model and optimizer before each validation and
checkpoint, so `train.Snapshot` writes the same float64 gob file as a CPU
run and `-resume` works in either direction. `cmd/gpt-train -backend gpu`
is the switch; it fails with a message when there is no hardware adapter.
Samples during training come from the synced CPU model through
`model.Generate`, which is fine at the eval cadence.
