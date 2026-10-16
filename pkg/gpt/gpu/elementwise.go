package gpu

import "fmt"

// Element-wise kernels run one thread per element in 256-thread
// workgroups over a (possibly 2D) grid; i is the flat element index.
const elemHead = `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let i = flat(wid, nwg) * 256u + li;
    if (i >= p.n) { return; }
`

var kFill = &kernel{name: "fill", binds: "w", src: `
struct P { n: u32, v: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
` + elemHead + `    y[i] = p.v;
}
`}

var kCopy = &kernel{name: "copy", binds: "rw", src: `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + elemHead + `    y[i] = x[i];
}
`}

var kAddScaled = &kernel{name: "add_scaled", binds: "rw", src: `
struct P { n: u32, s: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + elemHead + `    y[i] = fma(p.s, x[i], y[i]);
}
`}

var kScale = &kernel{name: "scale", binds: "w", src: `
struct P { n: u32, s: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
` + elemHead + `    y[i] = y[i] * p.s;
}
`}

var kMul = &kernel{name: "mul", binds: "rw", src: `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + elemHead + `    y[i] = y[i] * x[i];
}
`}

var kAddRow = &kernel{name: "add_row", binds: "rw", src: `
struct P { n: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + elemHead + `    y[i] = y[i] + bias[i % p.cols];
}
`}

// GELU is the tanh approximation used by GPT-2, as in pkg/gpt/model. The
// tanh argument is clamped: tanh(15) is 1 in float32, and some backends'
// tanh overflows to NaN through exp for large arguments.
const geluWGSL = `
const GELU_C: f32 = 0.7978845608028654;
const GELU_K: f32 = 0.044715;
fn tanh_arg(x: f32) -> f32 { return clamp(GELU_C * (x + GELU_K * x * x * x), -15.0, 15.0); }
`

var kGELU = &kernel{name: "gelu", binds: "rw", src: geluWGSL + `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + elemHead + `    let v = x[i];
    y[i] = 0.5 * v * (1.0 + tanh(tanh_arg(v)));
}
`}

var kGELUBackward = &kernel{name: "gelu_backward", binds: "rw", src: geluWGSL + `
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> pre: array<f32>;
@group(0) @binding(1) var<storage, read_write> d: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + elemHead + `    let v = pre[i];
    let t = tanh(tanh_arg(v));
    let du = GELU_C * (1.0 + 3.0 * GELU_K * v * v);
    d[i] = d[i] * (0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * du);
}
`}

// The dropout mask comes from a counter-based hash of (seed, step, index),
// so it needs no RNG state on the device, is reproducible for a given seed
// and step, and changes every step. pcg is the PCG-RXS-M-XS 32-bit hash;
// dropoutHash in Go mirrors it for the tests.
var kDropout = &kernel{name: "dropout", binds: "ww", src: `
struct P { n: u32, p: f32, scale: f32, seed: u32, step: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> mask: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
` + elemHead + `    let h = pcg(i ^ pcg(p.seed ^ pcg(p.step)));
    let u = f32(h >> 8u) * (1.0 / 16777216.0);
    var m = p.scale;
    if (u < p.p) { m = 0.0; }
    mask[i] = m;
    y[i] = y[i] * m;
}
`}

func (b *CommandBatch) same(op string, x, y *Tensor) {
	if x.Rows != y.Rows || x.Cols != y.Cols {
		panic(fmt.Sprintf("gpu: %s shape mismatch %s vs %s", op, x.Shape(), y.Shape()))
	}
}

// Fill records y[i] = v.
func (b *CommandBatch) Fill(y *Tensor, v float64) {
	b.dispatch(kFill, elems(y.Len()), []uint32{uint32(y.Len()), fbits(v), 0, 0}, y.buf)
}

// Zero records y = 0, as for clearing gradients at the start of a step.
func (b *CommandBatch) Zero(y *Tensor) { b.Fill(y, 0) }

// Copy records dst = src.
func (b *CommandBatch) Copy(dst, src *Tensor) {
	b.same("Copy", src, dst)
	b.dispatch(kCopy, elems(src.Len()), []uint32{uint32(src.Len()), 0, 0, 0}, src.buf, dst.buf)
}

// Add records y += x.
func (b *CommandBatch) Add(y, x *Tensor) { b.AddScaled(y, x, 1) }

// AddScaled records y += s*x.
func (b *CommandBatch) AddScaled(y, x *Tensor, s float64) {
	b.same("AddScaled", x, y)
	b.dispatch(kAddScaled, elems(y.Len()), []uint32{uint32(y.Len()), fbits(s), 0, 0}, x.buf, y.buf)
}

// Scale records y *= s.
func (b *CommandBatch) Scale(y *Tensor, s float64) {
	b.dispatch(kScale, elems(y.Len()), []uint32{uint32(y.Len()), fbits(s), 0, 0}, y.buf)
}

// Mul records y *= x element-wise; with x a dropout mask it is the dropout
// backward pass.
func (b *CommandBatch) Mul(y, x *Tensor) {
	b.same("Mul", x, y)
	b.dispatch(kMul, elems(y.Len()), []uint32{uint32(y.Len()), 0, 0, 0}, x.buf, y.buf)
}

// AddRow records y[i,j] += bias[j] (bias broadcast) for bias [1, y.Cols].
func (b *CommandBatch) AddRow(y, bias *Tensor) {
	if bias.Len() != y.Cols {
		panic(fmt.Sprintf("gpu: AddRow %s onto %s", bias.Shape(), y.Shape()))
	}
	b.dispatch(kAddRow, elems(y.Len()), []uint32{uint32(y.Len()), uint32(y.Cols), 0, 0}, bias.buf, y.buf)
}

// GELU records y = gelu(x).
func (b *CommandBatch) GELU(y, x *Tensor) {
	b.same("GELU", x, y)
	b.dispatch(kGELU, elems(x.Len()), []uint32{uint32(x.Len()), 0, 0, 0}, x.buf, y.buf)
}

// GELUBackward records d *= gelu'(pre), turning the gradient with respect
// to the GELU output into the gradient with respect to its input.
func (b *CommandBatch) GELUBackward(d, pre *Tensor) {
	b.same("GELUBackward", pre, d)
	b.dispatch(kGELUBackward, elems(d.Len()), []uint32{uint32(d.Len()), 0, 0, 0}, pre.buf, d.buf)
}

// Dropout records inverted dropout on y in place: each element is zeroed
// with probability p and the rest are scaled by 1/(1-p). The mask (0 or
// 1/(1-p) per element) is written to mask for the backward pass, which is
// Mul(dy, mask). The mask is a pure function of (seed, step, index): pass
// the optimizer step (and a distinct seed per dropout site) so each step
// draws a fresh mask without any RNG state on the device.
func (b *CommandBatch) Dropout(y, mask *Tensor, p float64, seed, step uint32) {
	b.same("Dropout", y, mask)
	if !(p >= 0 && p < 1) {
		panic(fmt.Sprintf("gpu: dropout probability %g", p))
	}
	b.dispatch(kDropout, elems(y.Len()), []uint32{uint32(y.Len()), fbits(p), fbits(1 / (1 - p)), seed, step, 0, 0, 0}, y.buf, mask.buf)
}

// dropoutHash is the Go mirror of the dropout kernel's hash, for tests.
func dropoutHash(i, seed, step uint32) uint32 {
	pcg := func(v uint32) uint32 {
		s := v*747796405 + 2891336453
		w := ((s >> ((s >> 28) + 4)) ^ s) * 277803737
		return (w >> 22) ^ w
	}
	return pcg(i ^ pcg(seed^pcg(step)))
}
