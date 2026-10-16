# gpu

A float32, device-resident tensor layer for training the GPT in `pkg/gpt` on
a GPU, through [gogpu/wgpu](https://github.com/gogpu/wgpu) (pure-Go WebGPU,
no cgo). It runs on Metal on macOS and on Vulkan on Linux, and cross-compiles
with `CGO_ENABLED=0` like the rest of the module.

This is phase one of the GPU backend: the tensor type and the complete kernel
set a training step needs, each one checked against the float64 CPU code.
Nothing in `pkg/gpt` uses it yet. The CPU path is still the default and has
not changed. Phase two moves the model's forward and backward passes onto
these kernels. Checkpoints stay in the float64 gob format: upload on load,
download on save.

## Design

| type | role |
|---|---|
| `Device` | Owns the wgpu instance, adapter, device and queue. Compiles each kernel's pipeline on first use. Holds a bind-group cache, a uniform arena for per-dispatch parameters and the scratch buffer for `Sum`. `Open()` returns `ErrNoAdapter` when there is no hardware GPU. |
| `Tensor` | A `Rows x Cols` float32 storage buffer. `Upload(*tensor.Mat)` and `Download(*tensor.Mat)` convert from and to float64. `UploadF32` and `DownloadF32` skip the conversion. |
| `Index` | Row ids into a table (token ids, positions, targets). It also stores the inverse map in CSR form, built on the host, so `ScatterAdd` is race-free and deterministic. WGSL has no float atomics. |
| `CommandBatch` | Records any number of dispatches into one command encoder and one compute pass, then `Submit()` sends them in one queue submission and waits once. `Read(t)` schedules a copy back to the host at the end of the batch. |

A step looks like this:

```go
d, err := gpu.Open()          // errors.Is(err, gpu.ErrNoAdapter) -> use the CPU
b := d.Batch()
b.Gather(x, tokEmb, ids, false)
b.Gather(x, posEmb, positions, true)
// ... forward, loss, backward, AdamW ...
loss := b.Read(lossScalar)
err = b.Submit()              // one submission, one wait
fmt.Println(loss.Values()[0])
```

How the batch works:

- **Parameters.** Each dispatch gets a 256-byte slot in a uniform arena. The
  slots are written with one `WriteBuffer` per 4096-slot chunk at submit time.
  A training step records the same dispatches in the same order every time,
  so slot offsets repeat from step to step. Bind groups are keyed on
  (pipeline, buffers, slot), which makes every step after the first a pure
  cache hit.
- **Ordering.** Dispatches run in the order they were recorded, and each one
  sees the writes of the ones before it. gogpu's Metal and Vulkan HALs insert
  a compute-to-compute memory barrier after every dispatch.
- **Uploads.** `Upload` goes through the queue's staging belt and lands
  before the next submission. That is WebGPU semantics: an upload made while
  a batch is still recording is seen by every dispatch in that batch.
- **Errors.** Shape mismatches panic, as they do in `pkg/gpt/tensor`. GPU-side
  errors come back from `Submit`.

## Kernels

Every kernel is WGSL in a Go string with a `main` compute entry point.

| method | CPU counterpart it is tested against |
|---|---|
| `MatMul`, `MatMulT`, `MatMulTA` (`accumulate` gives `C +=`) | `tensor.MatMul`, `MatMulT`, `MatMulTA` |
| `GEMM` epilogue: `Bias` by column, then `Mode` (`EpilogueGELU` writes `gelu(result)` to `Out2`; `EpilogueDropoutResidual` writes the mask to `Out2` and `Aux + mask*result` to C; `EpilogueResidual`; `EpilogueGELUGrad` multiplies by `gelu'(Aux)`) | the plain product followed by the CPU `AddRow`, `gelu`, dropout mask, residual add and `geluGrad` |
| split-K: a plain product with few output tiles and deep K runs as K-slices into a scratch and a reduce (automatic) | the same `tensor` products |
| `MulInto` (z = x*y in one pass) | element-wise multiply |
| `LayerNormBackwardAdd` (dx = LN'(dy) + res) | `model.LayerNorm.Backward` plus `Add` |
| `GEMM`: strided, batched, any of the three variants, operands at offsets inside larger tensors | per-head `tensor` products on the `[B*T, 3D]` qkv layout |
| `Transpose` | `tensor.TransposeInto` |
| `Softmax(x, scale, causal)`, `SoftmaxBackward` (row-wise, in place, causal over `[B*H*T, T]`) | `model.causalSoftmax` / `causalSoftmaxBackward` (mirrored in the test; they are unexported) |
| `LayerNorm` (writes `y`, `xhat`, `rstd`), `LayerNormBackward` (`dx`, plus `dg +=` and `db +=`) | `model.LayerNorm.Forward` / `Backward` |
| `GELU`, `GELUBackward` | `model` `gelu` / `geluGrad` (mirrored) |
| `Fill`/`Zero`, `Copy`, `Add`, `AddScaled`, `Scale`, `Mul`, `AddRow` (bias broadcast), `ColSum` (bias gradient) | `tensor.Mat` methods |
| `Gather` (rows by index, optionally accumulating), `ScatterAdd` (its gradient) | the embedding loops in `GPT.Forward` / `Backward` |
| `CrossEntropy` (per-row losses and `dLogits` in one pass, optionally in place over the logits) | `train.CrossEntropy` |
| `Sum` (two-stage reduction to one float: plain or sum of squares, scaled, optionally accumulating) | float64 sums |
| `Dropout(y, mask, p, seed, step)`: mask from a counter-based hash, so there is no RNG state on the device; backward is `Mul(dy, mask)` | a Go mirror of the hash (bit-exact), the drop rate, and a different mask per step |
| `AdamW` (hyperparameters and bias corrections as uniforms; optional clipping from a GPU-side sum of squares) | `train.ClipGradNorm` + `train.AdamW.Step`, three steps |

The tests use the training shapes (batch 16, context 256, width 384, 6 heads,
vocabulary 2000) plus ragged ones (1x1x1, 37x53x29, 65x17x130, context 67,
widths 53 and 1000). Each comparison is `max|gpu - cpu| / max|cpu|`, at
`1e-4` for anything that reduces over long rows and at `1e-5` to `1e-7` for
element-wise kernels.

### Matmul choices, measured

- **Starting point.** The spike's unrolled 64x64-tile, vec4-FMA kernel with
  named accumulators.
- **vec4 A tile.** The spike staged A's tile as scalars, four scalar
  workgroup loads per inner step. Staging it k-major as vec4s, like B, makes
  each inner step two vector loads and four FMAs. NN went from 938 to 1296
  GFLOP/s and TN from 767 to 1120 on the M5.
- **Transposed operands get dedicated loads.** The variants differ only in
  how they fill the tiles. For a transposed operand, each thread gathers four
  stored rows at one k, and consecutive threads take consecutive k. This beat
  `Transpose` followed by the NN kernel at every training shape (table
  below), and it needs no scratch buffer.
- **Strided and batched for attention.** All three variants take batch
  strides. Attention can then run every (batch, head) product in one
  dispatch, with no gather.

## Running the tests

```bash
go test ./pkg/gpt/gpu/                        # uses this machine's GPU; prints the adapter
go test -v -run LayerNorm ./pkg/gpt/gpu/      # one kernel
go test -run XXX -bench . -benchtime 5x ./pkg/gpt/gpu/
```

On a machine with no GPU, `Open` returns `ErrNoAdapter` and every test
skips with a message, so `go test ./...` passes. gogpu always registers a
software adapter as a fallback, and this package treats it as no GPU. Two
environment variables change that:

- `BRAIN_GPU_DISABLE=1` forces the no-GPU path. Use it to check the skips or
  to keep a GPU machine on the CPU.
- `BRAIN_GPU_ALLOW_CPU=1` accepts the software adapter, for debugging kernels
  on a machine without a GPU.

Linux needs the Vulkan loader (`libvulkan.so.1`) and a driver, such as Mesa
RADV for AMD. For the server:

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go test -c -o bin/linux-amd64/gpu.test ./pkg/gpt/gpu/
# then on the server: ./gpu.test -test.v  and  ./gpu.test -test.run XXX -test.bench . -test.benchtime 5x
```

## Measured

Apple M5, Metal, Go 1.26. Each benchmark records 20 dispatches into one
batch and reports per-dispatch time with the data resident on the GPU.
Three runs agreed to within 1%.

| benchmark | shape (m x k x n) | GFLOP/s | dedicated kernel | transpose + NN |
|---|---|---|---|---|
| NN, MLP fc forward | 4096 x 384 x 1536 | **1296** | 3.73 ms | |
| NN, MLP proj forward | 4096 x 1536 x 384 | 1247 | 3.88 ms | |
| NT, logits forward | 4096 x 384 x 2000 | 1271 | 4.95 ms | 5.00 ms (1259) |
| NT, fc input gradient | 4096 x 1536 x 384 | 1247 | 3.87 ms | 3.92 ms (1232) |
| NN, logits input gradient | 4096 x 2000 x 384 | 1251 | 5.03 ms | |
| TN, fc weight gradient | 384 x 4096 x 1536 | 1118 | 4.32 ms | 4.42 ms (1093) |
| TN, embedding gradient | 2000 x 4096 x 384 | 1146 | 5.49 ms | 6.08 ms (1035) |

| benchmark | shape | GB/s (logical traffic) | per dispatch |
|---|---|---|---|
| LayerNorm forward | 4096 x 384 | 56 | 339 µs |
| LayerNorm backward (dx + dg/db) | 4096 x 384 | 68 | 462 µs |
| causal Softmax forward | 24576 x 256 | 29 | 1.76 ms |
| causal Softmax backward | 24576 x 256 | 71 | 1.06 ms |

For comparison, the trainer's CPU `tensor.MatMul` runs at 148 GFLOP/s on this
machine, so the GPU matmuls are 7.5 to 8.8 times faster. The AMD Radeon 780M
numbers in `experiments/gpu-spike` (1481 GFLOP/s for the spike kernel) have
not been re-measured with these kernels yet.

The row kernels run well below the M5's 200 GB/s. They give each row one
256-thread workgroup with two or three passes over it. That is simple and
exact, and still a small share of a step: the matmuls are about 330 GFLOP,
roughly 0.27 s at these rates. The softmax forward is the weakest kernel and
the first to tune if profiling in phase two says so. Options are keeping a
row's values in named registers instead of re-reading global memory, or
using smaller workgroups for 256-wide rows. `ColSum` and the LayerNorm
gain/bias reduction use only `ceil(cols/32)` workgroups. A split-rows
two-stage version would fill the GPU better at width 384.

### After the fusion pass (phase 3b)

The step profile in `pkg/gpt/gpumodel` showed the row kernels were bound not
by bandwidth but by the latency of their workgroup reductions (one row per
256-thread workgroup, sixteen barrier rounds for a 256-wide row), and that
the elementwise passes were small on the M5's 200 GB/s but not on the
Radeon's 67. Changes, all verified against the CPU at the previous
tolerances:

- **Fused matmul epilogue.** Bias, GELU (writing both pre-activation and
  activation), dropout plus residual (writing the mask), residual, and
  GELU-gradient run as the result is stored. A forward sub-layer is now one
  dispatch; the GELU backward disappears into the gradient matmul.
- **128-row tiles** for products with M >= 1024: eight accumulators per
  thread, half the workgroup-memory traffic per FLOP. Selected by M; the
  64-row tile stays for shorter products so they keep enough workgroups.
- **Split-K** for plain products with fewer than 512 output tiles and K >=
  1024 (the weight gradients): K-slices as batch entries into a scratch,
  then a fixed-order reduce.
- **Lane row kernels** (`rows_lane.go`) for rows up to 512 wide: four rows
  per workgroup, 64 lanes per row, eight registers per lane, six barrier
  rounds shared by four rows. Rows up to 1024 use a one-row register kernel
  (`rows_reg.go`); wider rows keep the multi-pass kernels.
- `MulInto` and `LayerNormBackwardAdd` fold the dropout backward's copy and
  the residual gradient add into one pass each.

Apple M5, same benchmarks, before and after:

| benchmark | before | after |
|---|---|---|
| NN fc forward 4096x384x1536 | 1296 GFLOP/s | **1490** |
| NN proj forward 4096x1536x384 | 1247 | 1390 |
| NT logits forward 4096x384x2000 | 1271 | 1461 |
| NT fc dx 4096x1536x384 | 1247 | 1388 |
| NN logits dx 4096x2000x384 | 1251 | 1462 |
| TN fc dW 384x4096x1536 (split-K, 64-row tile) | 1118 | 1235 |
| TN emb dW 2000x4096x384 | 1146 | 1414 |
| LayerNorm forward 4096x384 | 56 GB/s | **134** |
| LayerNorm backward (dx + dg/db) | 68 | 111 |
| causal softmax forward 24576x256 | 29 | **106** |
| causal softmax backward | 71 | 175 |

## Constraints phase two must respect

These three rules come from the spike's measurements. Breaking any of them
costs more than the kernels themselves.

1. **One submission per training step.** Each queue submit-and-wait costs
   about 3 ms on Vulkan, more than most kernels. Record the whole step
   (forward, loss, backward, gradient norm, AdamW) into one `CommandBatch` and
   call `Submit` once. Anything that needs a value computed on the GPU has to
   take it from a buffer, not from the host. That is why `AdamW` reads the
   clipping norm from a GPU-side `Sum(..., squares=true)` instead of a host
   scalar, and why bias corrections come from the host step counter.
   `Tensor.Download` does its own submission, so it does not belong inside a
   step.
2. **Read back only scalars.** Readback measured about 1 GB/s. Use
   `b.Read` on one-element tensors (the mean loss, the squared gradient
   norm), never on activations or gradients. Use `Download` only for
   checkpoints and evaluation.
3. **No dynamically indexed local arrays in WGSL.** Per-thread arrays indexed
   at run time spill to memory and cost 3x in the spike. Keep per-thread
   state in named scalars and vectors (`acc0`..`acc3`, `v.x`..`v.w`), and
   index only workgroup or storage memory dynamically. Every kernel here
   follows this rule; new kernels must too.

Some smaller rules follow from the design:

- **Keep the recording order stable.** The bind-group cache and the uniform
  arena assume each step records the same dispatches in the same order, so
  every step after the first allocates nothing.
- **No float atomics.** Anything that sums into shared rows needs a
  deterministic layout. `ScatterAdd` uses an `Index`'s CSR. Column sums use
  workgroup reductions.
- **Never bind the same tensor twice in one dispatch** when one of the
  bindings is writable. WebGPU forbids it, so the in-place kernels
  (`Softmax`, `CrossEntropy` with `dLogits == logits`, `GELUBackward`, `Mul`)
  bind it once.
- **Size limits.** Buffers are capped by the adapter's
  `maxStorageBufferBindingSize`. `Open` raises it to the adapter maximum
  (the default is 128 MiB). Offsets and lengths are u32 element counts.
