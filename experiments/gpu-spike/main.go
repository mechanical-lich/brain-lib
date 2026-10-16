// Command gpu-spike measures whether a float32 matmul on the GPU through
// gogpu/wgpu (pure Go, no cgo) beats the CPU kernel in pkg/gpt/tensor.
//
// It multiplies A [M,K] by B [K,N] at the shape of the MLP's first linear
// layer in a GPT training step (M = batch*context, K = width, N = 4*width),
// times the CPU kernel, then times the GPU: data upload, one dispatch, a
// run of dispatches with the data resident, and the readback. It checks the
// GPU result against the CPU result and prints GFLOP/s for everything.
//
//	go run .                       # default shape 4096 x 384 x 1536
//	go run . -m 4096 -k 384 -n 1536 -reps 20
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
	"unsafe"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// matmulWGSL is a classic shared-memory tiled matmul: each 16x16 workgroup
// computes a 16x16 tile of C, streaming 16-wide slabs of A and B through
// workgroup memory. It is a straightforward kernel, not a tuned one; a
// register-blocked version would typically run 2 to 4 times faster.
const matmulWGSL = `
struct Params { m: u32, k: u32, n: u32, pad: u32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

const TILE: u32 = 16u;
var<workgroup> as: array<f32, 256>;
var<workgroup> bs: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    var acc: f32 = 0.0;
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * TILE;
        let ak = k0 + lid.x;
        if (row < p.m && ak < p.k) {
            as[lid.y * TILE + lid.x] = a[row * p.k + ak];
        } else {
            as[lid.y * TILE + lid.x] = 0.0;
        }
        let bk = k0 + lid.y;
        if (bk < p.k && col < p.n) {
            bs[lid.y * TILE + lid.x] = b[bk * p.n + col];
        } else {
            bs[lid.y * TILE + lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TILE; kk = kk + 1u) {
            acc = acc + as[lid.y * TILE + kk] * bs[kk * TILE + lid.x];
        }
        workgroupBarrier();
    }
    if (row < p.m && col < p.n) {
        c[row * p.n + col] = acc;
    }
}

// Register-blocked variant: a 16x16 workgroup computes a 64x64 tile of C,
// each thread owning a 4x4 block held in registers, so every value loaded
// from workgroup memory feeds four multiply-adds instead of one.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
var<workgroup> at: array<f32, 1024>; // BM x BK, [m][k]
var<workgroup> bt: array<f32, 1024>; // BK x BN, [k][n]

@compute @workgroup_size(16, 16)
fn main_blocked(@builtin(workgroup_id) wid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>,
                @builtin(local_invocation_index) li: u32) {
    let row0 = wid.y * BM + lid.y * 4u;
    let col0 = wid.x * BN + lid.x * 4u;
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) { acc[i] = vec4<f32>(0.0); }
    let tiles = (p.k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * BK;
        for (var e = li; e < BM * BK; e = e + 256u) {
            let r = e / BK;
            let kk = e % BK;
            let gr = wid.y * BM + r;
            let gk = k0 + kk;
            var v: f32 = 0.0;
            if (gr < p.m && gk < p.k) { v = a[gr * p.k + gk]; }
            at[e] = v;
        }
        for (var e = li; e < BK * BN; e = e + 256u) {
            let kk = e / BN;
            let cc = e % BN;
            let gk = k0 + kk;
            let gc = wid.x * BN + cc;
            var v: f32 = 0.0;
            if (gk < p.k && gc < p.n) { v = b[gk * p.n + gc]; }
            bt[e] = v;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let bb = kk * BN + lid.x * 4u;
            let bv = vec4<f32>(bt[bb], bt[bb + 1u], bt[bb + 2u], bt[bb + 3u]);
            for (var i = 0u; i < 4u; i = i + 1u) {
                let av = at[(lid.y * 4u + i) * BK + kk];
                acc[i] = acc[i] + av * bv;
            }
        }
        workgroupBarrier();
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let gr = row0 + i;
        if (gr < p.m) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let gc = col0 + j;
                if (gc < p.n) { c[gr * p.n + gc] = acc[i][j]; }
            }
        }
    }
}

// Same 64x64 / 4x4 blocking with no dynamically indexed locals: the four
// accumulator rows are named vec4s and B's tile is stored as vec4 columns,
// so each inner step is 4 scalar loads, 1 vector load and 4 vector FMAs.
var<workgroup> au: array<f32, 1024>;       // BM x BK, [m][k]
var<workgroup> bu: array<vec4<f32>, 256>;  // BK x BN/4, [k][n/4]

@compute @workgroup_size(16, 16)
fn main_unrolled(@builtin(workgroup_id) wid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>,
                 @builtin(local_invocation_index) li: u32) {
    let row0 = wid.y * BM + lid.y * 4u;
    let col0 = wid.x * BN + lid.x * 4u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let tiles = (p.k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * BK;
        for (var e = li; e < BM * BK; e = e + 256u) {
            let r = e / BK;
            let kk = e % BK;
            let gr = wid.y * BM + r;
            let gk = k0 + kk;
            var v: f32 = 0.0;
            if (gr < p.m && gk < p.k) { v = a[gr * p.k + gk]; }
            au[e] = v;
        }
        // 16 x 64 B tile as 16 x 16 vec4s: one vec4 per thread
        {
            let kk = li / 16u;
            let c4 = li % 16u;
            let gk = k0 + kk;
            let gc = wid.x * BN + c4 * 4u;
            var v = vec4<f32>(0.0);
            if (gk < p.k) {
                let base = gk * p.n + gc;
                if (gc + 3u < p.n) {
                    v = vec4<f32>(b[base], b[base + 1u], b[base + 2u], b[base + 3u]);
                } else {
                    if (gc < p.n) { v.x = b[base]; }
                    if (gc + 1u < p.n) { v.y = b[base + 1u]; }
                    if (gc + 2u < p.n) { v.z = b[base + 2u]; }
                }
            }
            bu[li] = v;
        }
        workgroupBarrier();
        let abase = lid.y * 4u * BK;
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let bv = bu[kk * 16u + lid.x];
            let a0 = au[abase + kk];
            let a1 = au[abase + BK + kk];
            let a2 = au[abase + 2u * BK + kk];
            let a3 = au[abase + 3u * BK + kk];
            acc0 = fma(vec4<f32>(a0), bv, acc0);
            acc1 = fma(vec4<f32>(a1), bv, acc1);
            acc2 = fma(vec4<f32>(a2), bv, acc2);
            acc3 = fma(vec4<f32>(a3), bv, acc3);
        }
        workgroupBarrier();
    }
    if (col0 + 3u < p.n) {
        if (row0 < p.m)      { let o = row0 * p.n + col0;        c[o] = acc0.x; c[o + 1u] = acc0.y; c[o + 2u] = acc0.z; c[o + 3u] = acc0.w; }
        if (row0 + 1u < p.m) { let o = (row0 + 1u) * p.n + col0; c[o] = acc1.x; c[o + 1u] = acc1.y; c[o + 2u] = acc1.z; c[o + 3u] = acc1.w; }
        if (row0 + 2u < p.m) { let o = (row0 + 2u) * p.n + col0; c[o] = acc2.x; c[o + 1u] = acc2.y; c[o + 2u] = acc2.z; c[o + 3u] = acc2.w; }
        if (row0 + 3u < p.m) { let o = (row0 + 3u) * p.n + col0; c[o] = acc3.x; c[o + 1u] = acc3.y; c[o + 2u] = acc3.z; c[o + 3u] = acc3.w; }
    } else {
        // ragged right edge: scalar stores with bounds checks
        var rows = array<vec4<f32>, 4>(acc0, acc1, acc2, acc3);
        for (var i = 0u; i < 4u; i = i + 1u) {
            let gr = row0 + i;
            if (gr < p.m) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let gc = col0 + j;
                    if (gc < p.n) { c[gr * p.n + gc] = rows[i][j]; }
                }
            }
        }
    }
}

// Bandwidth probe: c[i] = 2 * a[i] over the first p.m * p.k elements. No
// arithmetic to speak of, so its time is memory traffic plus dispatch cost.
@compute @workgroup_size(256)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.m * p.k) {
        c[i] = 2.0 * a[i];
    }
}
`

func main() {
	m := flag.Int("m", 4096, "rows of A (batch * context)")
	k := flag.Int("k", 384, "cols of A / rows of B (embedding width)")
	n := flag.Int("n", 1536, "cols of B (4 * width)")
	reps := flag.Int("reps", 20, "resident dispatches to time")
	cpuReps := flag.Int("cpu-reps", 3, "CPU kernel repetitions to time")
	flag.Parse()
	if err := run(*m, *k, *n, *reps, *cpuReps); err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		os.Exit(1)
	}
}

func run(m, k, n, reps, cpuReps int) error {
	flops := 2 * float64(m) * float64(k) * float64(n)
	gflops := func(d time.Duration) float64 { return flops / d.Seconds() / 1e9 }
	fmt.Printf("shape: [%d x %d] @ [%d x %d], %.2f GFLOP per multiply\n\n", m, k, k, n, flops/1e9)

	rng := rand.New(rand.NewSource(1))
	a := tensor.Randn(m, k, 1, rng)
	b := tensor.Randn(k, n, 1, rng)

	// --- CPU reference: the tiled float64 kernel the trainer uses ---
	var cpu *tensor.Mat
	start := time.Now()
	for i := 0; i < cpuReps; i++ {
		cpu = tensor.MatMul(a, b)
	}
	cpuPer := time.Since(start) / time.Duration(cpuReps)
	fmt.Printf("CPU tensor.MatMul (float64): %v per multiply, %.1f GFLOP/s\n\n", cpuPer.Round(time.Microsecond), gflops(cpuPer))

	// --- GPU ---
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return fmt.Errorf("CreateInstance: %w", err)
	}
	defer instance.Release()
	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: gputypes.PowerPreferenceHighPerformance})
	if err != nil {
		return fmt.Errorf("RequestAdapter: %w", err)
	}
	defer adapter.Release()
	info := adapter.Info()
	fmt.Printf("GPU adapter: %s (backend %v, type %v)\n", info.Name, info.Backend, info.DeviceType)
	lim := adapter.Limits()
	fmt.Printf("limits: maxStorageBufferBindingSize=%d MB, maxComputeWorkgroupsPerDimension=%d\n",
		lim.MaxStorageBufferBindingSize/(1<<20), lim.MaxComputeWorkgroupsPerDimension)
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		return fmt.Errorf("RequestDevice: %w", err)
	}
	defer device.Release()
	queue := device.Queue()

	aBytes := f64ToF32Bytes(a.Data)
	bBytes := f64ToF32Bytes(b.Data)
	cSize := uint64(m*n) * 4

	bufA, err := device.CreateBuffer(&wgpu.BufferDescriptor{Label: "a", Size: uint64(len(aBytes)), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst})
	if err != nil {
		return err
	}
	defer bufA.Release()
	bufB, err := device.CreateBuffer(&wgpu.BufferDescriptor{Label: "b", Size: uint64(len(bBytes)), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst})
	if err != nil {
		return err
	}
	defer bufB.Release()
	bufC, err := device.CreateBuffer(&wgpu.BufferDescriptor{Label: "c", Size: cSize, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		return err
	}
	defer bufC.Release()
	staging, err := device.CreateBuffer(&wgpu.BufferDescriptor{Label: "staging", Size: cSize, Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead})
	if err != nil {
		return err
	}
	defer staging.Release()
	params := make([]byte, 16)
	putU32(params[0:], uint32(m))
	putU32(params[4:], uint32(k))
	putU32(params[8:], uint32(n))
	bufP, err := device.CreateBuffer(&wgpu.BufferDescriptor{Label: "params", Size: 16, Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst})
	if err != nil {
		return err
	}
	defer bufP.Release()

	// upload
	start = time.Now()
	if err := queue.WriteBuffer(bufA, 0, aBytes); err != nil {
		return err
	}
	if err := queue.WriteBuffer(bufB, 0, bBytes); err != nil {
		return err
	}
	if err := queue.WriteBuffer(bufP, 0, params); err != nil {
		return err
	}
	if err := device.WaitIdle(); err != nil {
		return fmt.Errorf("WaitIdle after upload: %w", err)
	}
	upload := time.Since(start)
	fmt.Printf("upload A+B (%.1f MB): %v\n", float64(len(aBytes)+len(bBytes))/(1<<20), upload.Round(time.Microsecond))

	// pipeline
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "matmul", WGSL: matmulWGSL})
	if err != nil {
		return fmt.Errorf("shader: %w", err)
	}
	defer shader.Release()
	bgl, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "matmul-bgl",
		Entries: []gputypes.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		return fmt.Errorf("bind group layout: %w", err)
	}
	defer bgl.Release()
	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label: "matmul-bg", Layout: bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: bufA}, {Binding: 1, Buffer: bufB}, {Binding: 2, Buffer: bufC}, {Binding: 3, Buffer: bufP},
		},
	})
	if err != nil {
		return fmt.Errorf("bind group: %w", err)
	}
	defer bg.Release()
	pl, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "matmul-pl", BindGroupLayouts: []*wgpu.BindGroupLayout{bgl}})
	if err != nil {
		return err
	}
	defer pl.Release()
	// readback + verify, run after each kernel so every one is checked
	verify := func() error {
		start := time.Now()
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			return err
		}
		enc.CopyBufferToBuffer(bufC, 0, staging, 0, cSize)
		cb, err := enc.Finish()
		if err != nil {
			return err
		}
		if _, err := queue.Submit(cb); err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := staging.Map(ctx, wgpu.MapModeRead, 0, cSize); err != nil {
			return fmt.Errorf("map staging: %w", err)
		}
		rngData, err := staging.MappedRange(0, cSize)
		if err != nil {
			return err
		}
		gpuC := f32BytesToF64(rngData.Bytes())
		if err := staging.Unmap(); err != nil {
			return err
		}
		readback := time.Since(start)
		var maxAbs, maxRef float64
		for i, v := range cpu.Data {
			if d := math.Abs(v - gpuC[i]); d > maxAbs {
				maxAbs = d
			}
			if av := math.Abs(v); av > maxRef {
				maxRef = av
			}
		}
		rel := maxAbs / maxRef
		fmt.Printf("  readback %.1f MB in %v; max |cpu-gpu| = %.3g, relative %.2e: ", float64(cSize)/(1<<20), readback.Round(time.Microsecond), maxAbs, rel)
		if rel > 1e-4 {
			fmt.Println("FAIL")
			return fmt.Errorf("GPU result disagrees with CPU beyond float32 tolerance")
		}
		fmt.Println("PASS")
		return nil
	}

	type kernel struct {
		name, entry string
		tile        int
	}
	kernels := []kernel{{"naive 16x16 tile", "main", 16}, {"blocked 64x64 tile, 4x4 per thread", "main_blocked", 64}, {"unrolled 64x64 tile, vec4 FMA", "main_unrolled", 64}}
	var best time.Duration
	for _, kn := range kernels {
		start = time.Now()
		pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: kn.entry, Layout: pl, Module: shader, EntryPoint: kn.entry})
		if err != nil {
			return fmt.Errorf("compute pipeline %s: %w", kn.entry, err)
		}
		fmt.Printf("\n[%s] pipeline compile: %v\n", kn.name, time.Since(start).Round(time.Microsecond))
		gx, gy := uint32((n+kn.tile-1)/kn.tile), uint32((m+kn.tile-1)/kn.tile)
		dispatch := func(count int) error {
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				return err
			}
			pass, err := enc.BeginComputePass(nil)
			if err != nil {
				return err
			}
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			for i := 0; i < count; i++ {
				pass.Dispatch(gx, gy, 1)
			}
			if err := pass.End(); err != nil {
				return err
			}
			cb, err := enc.Finish()
			if err != nil {
				return err
			}
			if _, err := queue.Submit(cb); err != nil {
				return err
			}
			return device.WaitIdle()
		}
		if err := dispatch(1); err != nil {
			return fmt.Errorf("warm-up dispatch: %w", err)
		}
		start = time.Now()
		if err := dispatch(1); err != nil {
			return err
		}
		single := time.Since(start)
		fmt.Printf("  single dispatch (submit + wait):    %v, %.1f GFLOP/s\n", single.Round(time.Microsecond), gflops(single))
		start = time.Now()
		for i := 0; i < reps; i++ {
			if err := dispatch(1); err != nil {
				return err
			}
		}
		perSubmit := time.Since(start) / time.Duration(reps)
		fmt.Printf("  %d separate submits, resident:      %v each, %.1f GFLOP/s\n", reps, perSubmit.Round(time.Microsecond), gflops(perSubmit))
		start = time.Now()
		if err := dispatch(reps); err != nil {
			return err
		}
		perBatched := time.Since(start) / time.Duration(reps)
		fmt.Printf("  %d dispatches in one submit:        %v each, %.1f GFLOP/s\n", reps, perBatched.Round(time.Microsecond), gflops(perBatched))
		if err := verify(); err != nil {
			return fmt.Errorf("%s: %w", kn.name, err)
		}
		if best == 0 || perBatched < best {
			best = perBatched
		}
		pipeline.Release()
	}
	perBatched := best
	fmt.Println()

	// bandwidth probe
	{
		pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Label: "probe", Layout: pl, Module: shader, EntryPoint: "probe"})
		if err != nil {
			return fmt.Errorf("probe pipeline: %w", err)
		}
		elems := m * k
		gx := uint32((elems + 255) / 256)
		run := func(count int) error {
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				return err
			}
			pass, err := enc.BeginComputePass(nil)
			if err != nil {
				return err
			}
			pass.SetPipeline(pipeline)
			pass.SetBindGroup(0, bg, nil)
			for i := 0; i < count; i++ {
				pass.Dispatch(gx, 1, 1)
			}
			if err := pass.End(); err != nil {
				return err
			}
			cb, err := enc.Finish()
			if err != nil {
				return err
			}
			if _, err := queue.Submit(cb); err != nil {
				return err
			}
			return device.WaitIdle()
		}
		if err := run(1); err != nil {
			return err
		}
		start = time.Now()
		if err := run(reps); err != nil {
			return err
		}
		per := time.Since(start) / time.Duration(reps)
		bytes := float64(elems) * 4 * 2
		fmt.Printf("[bandwidth probe, %d elements x2 bytes=%.1f MB] %v per dispatch, %.1f GB/s\n", elems, bytes/(1<<20), per.Round(time.Microsecond), bytes/per.Seconds()/1e9)
		pipeline.Release()
	}

	fmt.Printf("\nsummary: best GPU kernel is %.1fx the CPU kernel; one step of the BPE run (~330 GFLOP) would take ~%v on the GPU at this rate vs ~%v on this CPU\n",
		cpuPer.Seconds()/perBatched.Seconds(),
		(time.Duration(330e9 / gflops(perBatched) / 1e9 * float64(time.Second))).Round(time.Millisecond),
		(time.Duration(330e9 / gflops(cpuPer) / 1e9 * float64(time.Second))).Round(time.Millisecond))
	return nil
}

func f64ToF32Bytes(x []float64) []byte {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(v)
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), len(out)*4)
}

func f32BytesToF64(b []byte) []float64 {
	f := unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
	out := make([]float64, len(f))
	for i, v := range f {
		out[i] = float64(v)
	}
	return out
}

func putU32(b []byte, v uint32) {
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}
