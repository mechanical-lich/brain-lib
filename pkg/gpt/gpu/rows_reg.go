package gpu

// Single-pass row kernels for rows of up to regCols columns: each of the
// 256 threads holds up to four elements of the row in named registers, so
// the row is read from global memory once and written once. The multi-pass
// kernels in rows.go re-read the row for every reduction and remain the
// fallback for wider rows (the vocabulary-wide cross-entropy, for one).
const regCols = 1024

// regLoad declares the four column indices and loads x into v0..v3 (0 when
// out of range). n is the number of live columns.
const regLoadX = `
    let j0 = li;
    let j1 = li + 256u;
    let j2 = li + 512u;
    let j3 = li + 768u;
    var v0: f32 = 0.0;
    var v1: f32 = 0.0;
    var v2: f32 = 0.0;
    var v3: f32 = 0.0;
    if (j0 < n) { v0 = x[base + j0]; }
    if (j1 < n) { v1 = x[base + j1]; }
    if (j2 < n) { v2 = x[base + j2]; }
    if (j3 < n) { v3 = x[base + j3]; }
`

var kLayerNormReg = &kernel{name: "layernorm_reg", binds: "rrrwww", src: wgslReduce + `
struct P { rows: u32, cols: u32, eps: f32, pad0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> xhat: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstd: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
` + rowHead + `    let n = p.cols;
    let inv_n = 1.0 / f32(n);
` + regLoadX + `
    let mean = reduce_sum(li, v0 + v1 + v2 + v3) * inv_n;
    var q: f32 = 0.0;
    if (j0 < n) { let dv = v0 - mean; q = fma(dv, dv, q); }
    if (j1 < n) { let dv = v1 - mean; q = fma(dv, dv, q); }
    if (j2 < n) { let dv = v2 - mean; q = fma(dv, dv, q); }
    if (j3 < n) { let dv = v3 - mean; q = fma(dv, dv, q); }
    let r = 1.0 / sqrt(reduce_sum(li, q) * inv_n + p.eps);
    if (li == 0u) { rstd[row] = r; }
    if (j0 < n) { let xh = (v0 - mean) * r; xhat[base + j0] = xh; y[base + j0] = fma(xh, g[j0], bias[j0]); }
    if (j1 < n) { let xh = (v1 - mean) * r; xhat[base + j1] = xh; y[base + j1] = fma(xh, g[j1], bias[j1]); }
    if (j2 < n) { let xh = (v2 - mean) * r; xhat[base + j2] = xh; y[base + j2] = fma(xh, g[j2], bias[j2]); }
    if (j3 < n) { let xh = (v3 - mean) * r; xhat[base + j3] = xh; y[base + j3] = fma(xh, g[j3], bias[j3]); }
}
`}

var kLayerNormBackwardReg = &kernel{name: "layernorm_backward_reg", binds: "rrrrrw", src: wgslReduce + `
struct P { rows: u32, cols: u32, hasres: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> xhat: array<f32>;
@group(0) @binding(2) var<storage, read> rstd: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
` + rowHead + `    let n = p.cols;
    let inv_n = 1.0 / f32(n);
    let j0 = li;
    let j1 = li + 256u;
    let j2 = li + 512u;
    let j3 = li + 768u;
    var d0: f32 = 0.0; var h0: f32 = 0.0;
    var d1: f32 = 0.0; var h1: f32 = 0.0;
    var d2: f32 = 0.0; var h2: f32 = 0.0;
    var d3: f32 = 0.0; var h3: f32 = 0.0;
    if (j0 < n) { d0 = dy[base + j0] * g[j0]; h0 = xhat[base + j0]; }
    if (j1 < n) { d1 = dy[base + j1] * g[j1]; h1 = xhat[base + j1]; }
    if (j2 < n) { d2 = dy[base + j2] * g[j2]; h2 = xhat[base + j2]; }
    if (j3 < n) { d3 = dy[base + j3] * g[j3]; h3 = xhat[base + j3]; }
    let mean_d = reduce_sum(li, d0 + d1 + d2 + d3) * inv_n;
    let mean_dx = reduce_sum(li, d0 * h0 + d1 * h1 + d2 * h2 + d3 * h3) * inv_n;
    let r = rstd[row];
    if (j0 < n) { var v = r * (d0 - mean_d - h0 * mean_dx); if (p.hasres != 0u) { v = v + res[base + j0]; } dx[base + j0] = v; }
    if (j1 < n) { var v = r * (d1 - mean_d - h1 * mean_dx); if (p.hasres != 0u) { v = v + res[base + j1]; } dx[base + j1] = v; }
    if (j2 < n) { var v = r * (d2 - mean_d - h2 * mean_dx); if (p.hasres != 0u) { v = v + res[base + j2]; } dx[base + j2] = v; }
    if (j3 < n) { var v = r * (d3 - mean_d - h3 * mean_dx); if (p.hasres != 0u) { v = v + res[base + j3]; } dx[base + j3] = v; }
}
`}

var kSoftmaxReg = &kernel{name: "softmax_reg", binds: "w", src: wgslReduce + `
struct P { rows: u32, cols: u32, scale: f32, causal: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
` + rowHead + `    var n = p.cols;
    if (p.causal != 0u) { n = row % p.cols + 1u; }
` + regLoadX + `
    var m0 = NEG_BIG; var m1 = NEG_BIG; var m2 = NEG_BIG; var m3 = NEG_BIG;
    if (j0 < n) { v0 = v0 * p.scale; m0 = v0; }
    if (j1 < n) { v1 = v1 * p.scale; m1 = v1; }
    if (j2 < n) { v2 = v2 * p.scale; m2 = v2; }
    if (j3 < n) { v3 = v3 * p.scale; m3 = v3; }
    let m = reduce_max(li, max(max(m0, m1), max(m2, m3)));
    var e0: f32 = 0.0; var e1: f32 = 0.0; var e2: f32 = 0.0; var e3: f32 = 0.0;
    if (j0 < n) { e0 = exp(v0 - m); }
    if (j1 < n) { e1 = exp(v1 - m); }
    if (j2 < n) { e2 = exp(v2 - m); }
    if (j3 < n) { e3 = exp(v3 - m); }
    let inv = 1.0 / reduce_sum(li, e0 + e1 + e2 + e3);
    if (j0 < p.cols) { x[base + j0] = e0 * inv; }
    if (j1 < p.cols) { x[base + j1] = e1 * inv; }
    if (j2 < p.cols) { x[base + j2] = e2 * inv; }
    if (j3 < p.cols) { x[base + j3] = e3 * inv; }
}
`}

var kSoftmaxBackwardReg = &kernel{name: "softmax_backward_reg", binds: "rw", src: wgslReduce + `
struct P { rows: u32, cols: u32, scale: f32, causal: u32 }
@group(0) @binding(0) var<storage, read> att: array<f32>;
@group(0) @binding(1) var<storage, read_write> d: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + rowHead + `    var n = p.cols;
    if (p.causal != 0u) { n = row % p.cols + 1u; }
    let j0 = li;
    let j1 = li + 256u;
    let j2 = li + 512u;
    let j3 = li + 768u;
    var a0: f32 = 0.0; var d0: f32 = 0.0;
    var a1: f32 = 0.0; var d1: f32 = 0.0;
    var a2: f32 = 0.0; var d2: f32 = 0.0;
    var a3: f32 = 0.0; var d3: f32 = 0.0;
    if (j0 < n) { a0 = att[base + j0]; d0 = d[base + j0]; }
    if (j1 < n) { a1 = att[base + j1]; d1 = d[base + j1]; }
    if (j2 < n) { a2 = att[base + j2]; d2 = d[base + j2]; }
    if (j3 < n) { a3 = att[base + j3]; d3 = d[base + j3]; }
    let dot = reduce_sum(li, a0 * d0 + a1 * d1 + a2 * d2 + a3 * d3);
    if (j0 < p.cols) { d[base + j0] = a0 * (d0 - dot) * p.scale; }
    if (j1 < p.cols) { d[base + j1] = a1 * (d1 - dot) * p.scale; }
    if (j2 < p.cols) { d[base + j2] = a2 * (d2 - dot) * p.scale; }
    if (j3 < p.cols) { d[base + j3] = a3 * (d3 - dot) * p.scale; }
}
`}
