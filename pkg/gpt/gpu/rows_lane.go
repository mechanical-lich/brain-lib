package gpu

import (
	"fmt"
	"strings"
)

// Lane kernels: rows of up to laneCols columns are handled four rows per
// 256-thread workgroup, 64 lanes per row, each lane holding up to eight
// elements in named registers. The row kernels in rows_reg.go were measured
// to be bound not by memory bandwidth but by the latency of their
// workgroup reductions (one row per workgroup, sixteen barrier rounds for a
// 256-wide row), so these amortize each barrier over four rows and halve
// the rounds. Every early return is replaced by a valid flag so all four
// rows reach the barriers.
const (
	laneCols  = 512
	laneRows  = 4
	laneWidth = 64
)

// wgslLaneReduce reduces within each 64-lane segment of the workgroup.
const wgslLaneReduce = `
var<workgroup> lred: array<f32, 256>;

fn lane_sum(li: u32, lane: u32, v: f32) -> f32 {
    lred[li] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lane < s) { lred[li] = lred[li] + lred[li + s]; }
        workgroupBarrier();
    }
    let r = lred[li - lane];
    workgroupBarrier();
    return r;
}

fn lane_max(li: u32, lane: u32, v: f32) -> f32 {
    lred[li] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lane < s) { lred[li] = max(lred[li], lred[li + s]); }
        workgroupBarrier();
    }
    let r = lred[li - lane];
    workgroupBarrier();
    return r;
}

const NEG_BIG: f32 = -3.0e38;
`

// laneHead maps the thread to (row, lane). valid is false for rows past
// the end; such threads still take part in every reduction with zeros.
const laneHead = `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let lane = li % 64u;
    let row = flat(wid, nwg) * 4u + li / 64u;
    let valid = row < p.rows;
    let base = row * p.cols;
`

// unroll8 expands a template over k = 0..7, replacing {k} with the index
// and {j} with the column expression lane + 64*k.
func unroll8(tmpl string) string {
	var sb strings.Builder
	for k := 0; k < 8; k++ {
		line := strings.ReplaceAll(tmpl, "{k}", fmt.Sprint(k))
		line = strings.ReplaceAll(line, "{j}", fmt.Sprintf("(lane + %du)", 64*k))
		sb.WriteString(line)
		sb.WriteString("\n")
	}
	return sb.String()
}

var kLayerNormLane = &kernel{name: "layernorm_lane", binds: "rrrwww", src: wgslLaneReduce + `
struct P { rows: u32, cols: u32, eps: f32, pad0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> xhat: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstd: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
` + laneHead + `    let n = p.cols;
    let inv_n = 1.0 / f32(n);
` + unroll8(`    var v{k}: f32 = 0.0; if (valid && {j} < n) { v{k} = x[base + {j}]; }`) + `
    let mean = lane_sum(li, lane, v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) * inv_n;
    var q: f32 = 0.0;
` + unroll8(`    if ({j} < n) { let dv = v{k} - mean; q = fma(dv, dv, q); }`) + `
    let r = 1.0 / sqrt(lane_sum(li, lane, q) * inv_n + p.eps);
    if (!valid) { return; }
    if (lane == 0u) { rstd[row] = r; }
` + unroll8(`    if ({j} < n) { let xh = (v{k} - mean) * r; xhat[base + {j}] = xh; y[base + {j}] = fma(xh, g[{j}], bias[{j}]); }`) + `
}
`}

var kLayerNormBackwardLane = &kernel{name: "layernorm_backward_lane", binds: "rrrrrw", src: wgslLaneReduce + `
struct P { rows: u32, cols: u32, hasres: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> xhat: array<f32>;
@group(0) @binding(2) var<storage, read> rstd: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
` + laneHead + `    let n = p.cols;
    let inv_n = 1.0 / f32(n);
` + unroll8(`    var d{k}: f32 = 0.0; var h{k}: f32 = 0.0; if (valid && {j} < n) { d{k} = dy[base + {j}] * g[{j}]; h{k} = xhat[base + {j}]; }`) + `
    let mean_d = lane_sum(li, lane, d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7) * inv_n;
    let mean_dx = lane_sum(li, lane, d0 * h0 + d1 * h1 + d2 * h2 + d3 * h3 + d4 * h4 + d5 * h5 + d6 * h6 + d7 * h7) * inv_n;
    if (!valid) { return; }
    let r = rstd[row];
` + unroll8(`    if ({j} < n) { var v = r * (d{k} - mean_d - h{k} * mean_dx); if (p.hasres != 0u) { v = v + res[base + {j}]; } dx[base + {j}] = v; }`) + `
}
`}

var kSoftmaxLane = &kernel{name: "softmax_lane", binds: "w", src: wgslLaneReduce + `
struct P { rows: u32, cols: u32, scale: f32, causal: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
` + laneHead + `    var n = p.cols;
    if (p.causal != 0u) { n = row % p.cols + 1u; }
` + unroll8(`    var v{k} = NEG_BIG; if (valid && {j} < n) { v{k} = x[base + {j}] * p.scale; }`) + `
    let m = lane_max(li, lane, max(max(max(v0, v1), max(v2, v3)), max(max(v4, v5), max(v6, v7))));
` + unroll8(`    var e{k}: f32 = 0.0; if ({j} < n) { e{k} = exp(v{k} - m); }`) + `
    let inv = 1.0 / lane_sum(li, lane, e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7);
    if (!valid) { return; }
` + unroll8(`    if ({j} < p.cols) { x[base + {j}] = e{k} * inv; }`) + `
}
`}

var kSoftmaxBackwardLane = &kernel{name: "softmax_backward_lane", binds: "rw", src: wgslLaneReduce + `
struct P { rows: u32, cols: u32, scale: f32, causal: u32 }
@group(0) @binding(0) var<storage, read> att: array<f32>;
@group(0) @binding(1) var<storage, read_write> d: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + laneHead + `    var n = p.cols;
    if (p.causal != 0u) { n = row % p.cols + 1u; }
` + unroll8(`    var a{k}: f32 = 0.0; var d{k}: f32 = 0.0; if (valid && {j} < n) { a{k} = att[base + {j}]; d{k} = d[base + {j}]; }`) + `
    let dot = lane_sum(li, lane, a0 * d0 + a1 * d1 + a2 * d2 + a3 * d3 + a4 * d4 + a5 * d5 + a6 * d6 + a7 * d7);
    if (!valid) { return; }
` + unroll8(`    if ({j} < p.cols) { d[base + {j}] = a{k} * (d{k} - dot) * p.scale; }`) + `
}
`}

// laneGrid is the workgroup grid for rows rows at four rows per workgroup.
func laneGrid(rows int) [3]uint32 { return grid(ceilDiv(rows, laneRows)) }
