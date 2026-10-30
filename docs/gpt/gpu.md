---
title: GPU backend
parent: GPT
nav_order: 2
permalink: /gpt/gpu/
---

# GPU backend (`pkg/gpt/gpu`, `pkg/gpt/gpumodel`)
{: .no_toc }

1. TOC
{:toc}

The GPU backend runs the GPT training step in float32 on the machine's GPU
through [gogpu/wgpu](https://github.com/gogpu/wgpu), a pure-Go WebGPU
implementation. No cgo, no SDK: it cross-compiles with `CGO_ENABLED=0` like
the rest of the module, uses Metal on macOS and Vulkan on Linux, and is
checked kernel by kernel against the float64 CPU code.

It is two packages:

- **`gpu`** is the tensor layer: a `Device`, a `Tensor` in a storage buffer,
  an `Index` for embedding lookups, and a `CommandBatch` that records any
  number of kernel dispatches and submits them once. Every kernel a training
  step needs is here as a method on `CommandBatch`.
- **`gpumodel`** is the model: the same parameters as `model.GPT` in the same
  order, every forward activation kept on the device for the backward pass,
  and `TrainStep`, which records forward, loss, backward, gradient norm and
  AdamW into one batch and reads back two floats. `Stepper` adapts it to
  `train.Stepper`.

Nothing in `model` or `train` imports either package. The CPU path is the
default and unchanged.

## Using it

Through the trainer, `-backend gpu` is the switch:

```bash
go run ./cmd/gpt-train -backend gpu -steps 2000
```

```bash
go run ./cmd/gpt-generate -backend gpu -prompt "The old house" -n 400
```

Both fail with a message when there is no hardware adapter. Checkpoints are
identical to CPU ones, so a run can move between backends at any `-resume`.

From Go:

```go
dev, err := gpu.Open()
if errors.Is(err, gpu.ErrNoAdapter) {
    // fall back to the CPU
}
defer dev.Close()

gm, err := gpumodel.New(dev, g.Config)   // g is a model.GPT, randomly initialized on the CPU
defer gm.Close()
gm.LoadFrom(g)
gm.SetSeed(1)

st, err := gpumodel.NewStepper(gm, g, opt, clip)   // opt is a train.AdamW over g.Params()
err = train.RunStepper(st, g, opt, trainSet, valSet, opts, rng, onStep)
```

Random initialization stays on the CPU: build a `model.GPT`, then `LoadFrom`
it. `StoreTo` downloads weights and moments back so `train.Snapshot` writes
the usual float64 checkpoint. `Stepper.Sync` does that before each
validation and checkpoint.

## What one training step submits

```
Zero every gradient
Gather tok_emb[ids] ; Gather pos_emb[t] (+=) ; Dropout
for each block:
  LayerNorm ; GEMM qkv (+bias) ; GEMM scores = q k^T ; Softmax(causal) ; GEMM out = att v
  GEMM proj (+bias, dropout, +residual)
  LayerNorm ; GEMM fc (+bias, GELU) ; GEMM proj (+bias, dropout, +residual)
LayerNorm ; MatMulT logits = hF tok_emb^T
CrossEntropy (per-row losses, dLogits in place) ; Sum -> loss
backward through every block in reverse, with the GELU gradient and the
  residual adds fused into the matmuls and LayerNorm backward
ScatterAdd into tok_emb.grad and pos_emb.grad
Sum of squares over every gradient -> normSq
AdamW for every parameter (clipping folded in from normSq)
Read loss, normSq
```

About 330 dispatches in one submission at the BPE shape.

## Design rules

These came out of the `experiments/gpu-spike` measurements, and breaking any
of them costs more than the kernels themselves.

1. **One submission per training step.** A queue submit-and-wait costs about
   3 ms on Vulkan, more than most kernels. Anything that needs a GPU-computed
   value takes it from a buffer: AdamW reads the clipping norm from a
   GPU-side sum of squares, and bias corrections come from the host step
   counter. `Tensor.Download` does its own submission and does not belong
   inside a step.
2. **Read back only scalars.** Readback measured about 1 GB/s. `b.Read` on
   one-element tensors (loss, squared gradient norm); `Download` only for
   checkpoints and evaluation.
3. **No dynamically indexed local arrays in WGSL.** They spill to memory and
   cost 3x. Per-thread state lives in named scalars and vectors.
4. **Stable recording order.** The bind-group cache and the uniform arena
   assume each step records the same dispatches in the same order, so every
   step after the first allocates nothing.
5. **No float atomics.** `ScatterAdd` sums rows that share an id through a
   host-built CSR map in a fixed order, so it is deterministic.
6. **Never bind the same tensor twice in one dispatch** when one binding is
   writable; the in-place kernels bind it once.

## Kernels

Every kernel is WGSL in a Go string. The table names the CPU counterpart each
is tested against.

| Method | Tested against |
|---|---|
| `MatMul`, `MatMulT`, `MatMulTA` (optionally accumulating) | `tensor.MatMul`, `MatMulT`, `MatMulTA` |
| `GEMM`: strided, batched, any transpose variant, with a fused epilogue (bias, GELU, dropout + residual, residual, GELU gradient) | the plain product followed by the CPU `AddRow`, `gelu`, dropout mask, residual add, `geluGrad` |
| split-K for deep products with few output tiles (automatic) | the same `tensor` products |
| `Softmax(x, scale, causal)`, `SoftmaxBackward` | `model`'s causal softmax |
| `LayerNorm`, `LayerNormBackward`, `LayerNormBackwardAdd` | `model.LayerNorm.Forward` / `Backward` |
| `GELU`, `GELUBackward` | `model`'s `gelu` / `geluGrad` |
| `Fill`, `Zero`, `Copy`, `Add`, `AddScaled`, `Scale`, `Mul`, `MulInto`, `AddRow`, `ColSum`, `Transpose` | `tensor.Mat` methods |
| `Gather`, `ScatterAdd` | the embedding loops in `GPT.Forward` / `Backward` |
| `CrossEntropy` | `train.CrossEntropy` |
| `Sum` (two-stage, plain or squares) | float64 sums |
| `Dropout(y, mask, p, seed, step)`: counter-based hash, no RNG state on the device | a bit-exact Go mirror of the hash |
| `AdamW` with optional clipping from a GPU-side norm | `train.ClipGradNorm` + `train.AdamW.Step` |

Comparisons are `max|gpu - cpu| / max|cpu|`, at `1e-4` for anything that
reduces over long rows and `1e-5` to `1e-7` for element-wise kernels, at the
training shapes and at ragged ones.

## Measured

Apple M5, Metal, Go 1.26, after the fusion pass. Each matmul benchmark
records 20 dispatches into one batch and reports per-dispatch time with data
resident on the GPU.

| Benchmark | Shape (m x k x n) | GFLOP/s |
|---|---|---|
| NN, MLP fc forward | 4096 x 384 x 1536 | 1490 |
| NN, MLP proj forward | 4096 x 1536 x 384 | 1390 |
| NT, logits forward | 4096 x 384 x 2000 | 1461 |
| TN, fc weight gradient (split-K) | 384 x 4096 x 1536 | 1235 |
| TN, embedding gradient | 2000 x 4096 x 384 | 1414 |

| Row kernel | Shape | GB/s |
|---|---|---|
| LayerNorm forward | 4096 x 384 | 134 |
| LayerNorm backward | 4096 x 384 | 111 |
| causal softmax forward | 24576 x 256 | 106 |
| causal softmax backward | 24576 x 256 | 175 |

The CPU `tensor.MatMul` runs at 148 GFLOP/s on the same machine.

Whole training step at the BPE shape (6 layers, width 384, 6 heads,
vocabulary 2000, context 256, batch 16, dropout 0.1):

| | ms per step |
|---|---|
| CPU, `model` + `train`, Apple M5 | 2525 |
| GPU, `gpumodel.TrainStep`, Apple M5 | 255 |
| CPU, AMD Ryzen 7 8745H | 7900 |
| GPU, AMD Radeon 780M (Vulkan, Mesa RADV) | 232 |

`Model.Profile` (and `TestProfileStep -v`) splits a step into named phases
and times each, which is how the fusion pass was prioritized. The package
readmes under `pkg/gpt/gpu/` and `pkg/gpt/gpumodel/` have the before/after
tables and the reasoning for each change.

## Testing and environment switches

```bash
go test ./pkg/gpt/gpu/ ./pkg/gpt/gpumodel/    # uses this machine's GPU; prints the adapter
```

```bash
go test -run XXX -bench . -benchtime 5x ./pkg/gpt/gpu/
```

On a machine without a hardware GPU, `Open` returns `ErrNoAdapter` and every
test skips with a message, so `go test ./...` passes. gogpu always registers
a software adapter as a fallback; this package treats it as no GPU.

| Variable | Effect |
|---|---|
| `BRAIN_GPU_DISABLE=1` | Force the no-GPU path. Check the skips, or keep a GPU machine on the CPU. |
| `BRAIN_GPU_ALLOW_CPU=1` | Accept the software adapter, for debugging kernels on a machine without a GPU. |

Linux needs the Vulkan loader (`libvulkan.so.1`) and a driver such as Mesa
RADV. To test on a server without a Go toolchain, cross-compile the test
binaries:

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go test -c -o bin/linux-amd64/gpu.test ./pkg/gpt/gpu/
```

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go test -c -o bin/linux-amd64/gpumodel.test ./pkg/gpt/gpumodel/
```

Then on the server: `./gpu.test -test.v` and
`./gpumodel.test -test.run XXX -test.bench TrainStep -test.benchtime 5x`.

## The spike

`experiments/gpu-spike` is a standalone Go module that asked the original
question: does a float32 matmul through pure-Go WebGPU beat the CPU kernel?
It times the CPU kernel against three WGSL kernels at the MLP shape and
probes bandwidth. Results: 6.5x the CPU on an Apple M5, 35x on an AMD Radeon
780M. The naive-to-unrolled spread showed that the shader compiler, not the
library, is what kernels must be written for. Run it with
`cd experiments/gpu-spike && go run .`.
