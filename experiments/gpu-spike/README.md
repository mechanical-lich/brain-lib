# gpu-spike

A standalone experiment, in its own Go module so the main module takes on no
dependency: does a float32 matmul through [gogpu/wgpu](https://github.com/gogpu/wgpu)
(pure Go WebGPU, no cgo) beat the CPU kernel in `pkg/gpt/tensor`?

It multiplies matrices at the shape of a GPT training step's MLP layer
(4096 x 384 by 384 x 1536, the BPE Lovecraft run), times the CPU kernel, then
times three WGSL kernels with the data resident on the GPU and verifies each
against the CPU result. A bandwidth probe at the end shows whether the GPU
path itself is healthy, independent of kernel quality.

```bash
cd experiments/gpu-spike
go run .                               # this machine's GPU (Metal on macOS, Vulkan on Linux)
go run . -m 4096 -k 384 -n 1536 -reps 20

# cross-compile for the Linux server
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o ../../bin/linux-amd64/gpu-spike .
```

## Reading the output

| line | meaning |
|---|---|
| `CPU tensor.MatMul` | the trainer's own kernel, float64, all cores |
| `naive 16x16 tile` | one output per thread through workgroup memory; the textbook kernel |
| `blocked 64x64 tile` | 4x4 outputs per thread, accumulators in a dynamically indexed array |
| `unrolled 64x64 tile, vec4 FMA` | same blocking with named accumulators and vector loads; the one that matters |
| `bandwidth probe` | a copy kernel: GB/s achieved and per-dispatch overhead |
| `summary` | best kernel vs CPU, and what a ~330 GFLOP training step would cost at that rate |

Each kernel line reports `dispatches in one submit`, the rate a training step
would see with everything resident on the GPU, followed by a PASS/FAIL check
against the CPU at float32 tolerance.

## Results so far

Apple M5 (Metal): CPU 148 GFLOP/s; naive 126; blocked 243; unrolled 965
GFLOP/s, 6.5x the CPU. Bandwidth probe 200 GB/s at 60 µs per dispatch.
The naive-to-unrolled spread shows the shader compiler, not the library, is
the thing to write kernels for: dynamically indexed locals spill to memory.

AMD Radeon 780M (Vulkan via Mesa RADV, Minisforum UM870 Slim, Ryzen 7 8745H):
CPU 42 GFLOP/s; naive 521; blocked 1159; unrolled 1481 GFLOP/s, 35x the
CPU. Separate submits cost about 3 ms each more than batched dispatches, so a
training step must be one submission. Readback ran at about 1 GB/s, so only
scalars should come back per step. Bandwidth probe 67 GB/s.
