package gpu

import (
	"fmt"
	"strings"
)

// wgslReduce is a 256-thread workgroup tree reduction. Callers must reach
// it in uniform control flow: every early return before it depends only on
// the workgroup id.
const wgslReduce = `
var<workgroup> red: array<f32, 256>;

fn reduce_sum(li: u32, v: f32) -> f32 {
    red[li] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { red[li] = red[li] + red[li + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn reduce_max(li: u32, v: f32) -> f32 {
    red[li] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { red[li] = max(red[li], red[li + s]); }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

const NEG_BIG: f32 = -3.0e38;
`

// sumMaxPartials caps the first stage of Sum; its scratch buffer is owned
// by the Device.
const sumMaxPartials = 1024

var kSumPartial = &kernel{name: "sum_partial", binds: "rw", src: wgslReduce + `
struct P { n: u32, squares: u32, groups: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    var s: f32 = 0.0;
    for (var i = wid.x * 256u + li; i < p.n; i = i + p.groups * 256u) {
        let v = x[i];
        if (p.squares != 0u) { s = fma(v, v, s); } else { s = s + v; }
    }
    let t = reduce_sum(li, s);
    if (li == 0u) { partial[wid.x] = t; }
}
`}

var kSumFinal = &kernel{name: "sum_final", binds: "rw", src: wgslReduce + `
struct P { count: u32, index: u32, acc: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    var s: f32 = 0.0;
    for (var i = li; i < p.count; i = i + 256u) { s = s + partial[i]; }
    let t = reduce_sum(li, s) * p.scale;
    if (li == 0u) {
        if (p.acc != 0u) { out[p.index] = out[p.index] + t; } else { out[p.index] = t; }
    }
}
`}

// Sum records out[index] (+)= scale * sum(x), or scale * sum(x*x) when
// squares is set (for the gradient norm). It runs in two deterministic
// stages through a scratch buffer on the device: up to 1024 workgroups
// write partial sums, then one workgroup adds those up.
func (b *CommandBatch) Sum(out *Tensor, index int, x *Tensor, squares, accumulate bool, scale float64) {
	if index < 0 || index >= out.Len() {
		panic(fmt.Sprintf("gpu: Sum into element %d of %s", index, out.Shape()))
	}
	groups := min(ceilDiv(x.Len(), 1024), sumMaxPartials)
	b.dispatch(kSumPartial, [3]uint32{uint32(groups), 1, 1}, []uint32{uint32(x.Len()), flag(squares), uint32(groups), 0}, x.buf, b.d.partials)
	b.dispatch(kSumFinal, [3]uint32{1, 1, 1}, []uint32{uint32(groups), uint32(index), flag(accumulate), fbits(scale)}, b.d.partials, out.buf)
}

// Column reductions use 32 columns x 8 row lanes per workgroup: each lane
// walks every eighth row, so a warp reads 32 consecutive floats of a row,
// and the 8 lanes are folded in workgroup memory at the end.
const colHead = `
var<workgroup> colred: array<f32, 256>;
var<workgroup> colred2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let lx = li % 32u;
    let ly = li / 32u;
    let col = wid.x * 32u + lx;
    var s: f32 = 0.0;
    var s2: f32 = 0.0;
    if (col < p.cols) {
        for (var r = ly; r < p.rows; r = r + 8u) {
            let o = r * p.cols + col;
COL_BODY
        }
    }
    colred[li] = s;
    colred2[li] = s2;
    workgroupBarrier();
    if (ly == 0u && col < p.cols) {
        var t: f32 = 0.0;
        var t2: f32 = 0.0;
        for (var k = 0u; k < 8u; k = k + 1u) {
            t = t + colred[k * 32u + lx];
            t2 = t2 + colred2[k * 32u + lx];
        }
COL_STORE
    }
}
`

func colKernel(name, binds, decls, body, store string) *kernel {
	src := decls + strings.Replace(strings.Replace(colHead, "COL_BODY", body, 1), "COL_STORE", store, 1)
	return &kernel{name: name, binds: binds, src: src}
}

var kColSum = colKernel("colsum", "rw", `
struct P { rows: u32, cols: u32, acc: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
`, `            s = s + x[o];`, `        if (p.acc != 0u) { out[col] = out[col] + t; } else { out[col] = t; }`)

// ColSum records out[j] (+)= sum_i x[i,j] for out [1, x.Cols]: the bias
// gradient of a Linear layer.
func (b *CommandBatch) ColSum(out, x *Tensor, accumulate bool) {
	if out.Len() != x.Cols {
		panic(fmt.Sprintf("gpu: ColSum of %s into %s", x.Shape(), out.Shape()))
	}
	b.dispatch(kColSum, [3]uint32{uint32(ceilDiv(x.Cols, 32)), 1, 1}, []uint32{uint32(x.Rows), uint32(x.Cols), flag(accumulate), 0}, x.buf, out.buf)
}
