package gpu

import (
	"fmt"
	"strings"
)

// Row-wise kernels (softmax, LayerNorm, cross-entropy) give each row one
// 256-thread workgroup: threads stride across the row's columns and meet
// in workgroup reductions for the max, sums and dot products.
const rowHead = `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let row = flat(wid, nwg);
    if (row >= p.rows) { return; }
    let base = row * p.cols;
`

// Softmax: with causal set, row r of a [B*H*T, T] score matrix is query
// position t = r % T and only keys s <= t take part; the rest are written
// as 0, as model.causalSoftmax does. Scores are multiplied by scale first.
var kSoftmax = &kernel{name: "softmax", binds: "w", src: wgslReduce + `
struct P { rows: u32, cols: u32, scale: f32, causal: u32 }
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
` + rowHead + `    var n = p.cols;
    if (p.causal != 0u) { n = row % p.cols + 1u; }
    var m = NEG_BIG;
    for (var j = li; j < n; j = j + 256u) { m = max(m, x[base + j] * p.scale); }
    m = reduce_max(li, m);
    var s: f32 = 0.0;
    for (var j = li; j < n; j = j + 256u) {
        let e = exp(x[base + j] * p.scale - m);
        x[base + j] = e;
        s = s + e;
    }
    let inv = 1.0 / reduce_sum(li, s);
    for (var j = li; j < p.cols; j = j + 256u) {
        if (j < n) { x[base + j] = x[base + j] * inv; } else { x[base + j] = 0.0; }
    }
}
`}

// SoftmaxBackward: d holds dL/d(att) and becomes dL/d(scores before
// scaling): att * (d - sum(att*d)) * scale, zero above the diagonal.
var kSoftmaxBackward = &kernel{name: "softmax_backward", binds: "rw", src: wgslReduce + `
struct P { rows: u32, cols: u32, scale: f32, causal: u32 }
@group(0) @binding(0) var<storage, read> att: array<f32>;
@group(0) @binding(1) var<storage, read_write> d: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
` + rowHead + `    var n = p.cols;
    if (p.causal != 0u) { n = row % p.cols + 1u; }
    var s: f32 = 0.0;
    for (var j = li; j < n; j = j + 256u) { s = fma(att[base + j], d[base + j], s); }
    let dot = reduce_sum(li, s);
    for (var j = li; j < p.cols; j = j + 256u) {
        if (j < n) {
            d[base + j] = att[base + j] * (d[base + j] - dot) * p.scale;
        } else {
            d[base + j] = 0.0;
        }
    }
}
`}

// Softmax records x = softmax(scale * x) row-wise in place. With causal,
// x is [B*H*T, T] (one row per query) and row r attends to keys
// [0, r % T]; the masked tail is set to 0.
func (b *CommandBatch) Softmax(x *Tensor, scale float64, causal bool) {
	k, g := kSoftmax, grid(x.Rows)
	switch {
	case x.Cols <= laneCols:
		k, g = kSoftmaxLane, laneGrid(x.Rows)
	case x.Cols <= regCols:
		k = kSoftmaxReg
	}
	b.dispatch(k, g, []uint32{uint32(x.Rows), uint32(x.Cols), fbits(scale), flag(causal)}, x.buf)
}

// SoftmaxBackward records, in place, d = att * (d - rowsum(att*d)) * scale
// given the Softmax output att: the gradient with respect to the scores
// before scaling, from the gradient with respect to att.
func (b *CommandBatch) SoftmaxBackward(d, att *Tensor, scale float64, causal bool) {
	b.same("SoftmaxBackward", att, d)
	k, g := kSoftmaxBackward, grid(d.Rows)
	switch {
	case d.Cols <= laneCols:
		k, g = kSoftmaxBackwardLane, laneGrid(d.Rows)
	case d.Cols <= regCols:
		k = kSoftmaxBackwardReg
	}
	b.dispatch(k, g, []uint32{uint32(d.Rows), uint32(d.Cols), fbits(scale), flag(causal)}, att.buf, d.buf)
}

var kLayerNorm = &kernel{name: "layernorm", binds: "rrrwww", src: wgslReduce + `
struct P { rows: u32, cols: u32, eps: f32, pad0: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> xhat: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstd: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
` + rowHead + `    let inv_n = 1.0 / f32(p.cols);
    var s: f32 = 0.0;
    for (var j = li; j < p.cols; j = j + 256u) { s = s + x[base + j]; }
    let mean = reduce_sum(li, s) * inv_n;
    var q: f32 = 0.0;
    for (var j = li; j < p.cols; j = j + 256u) {
        let dv = x[base + j] - mean;
        q = fma(dv, dv, q);
    }
    let r = 1.0 / sqrt(reduce_sum(li, q) * inv_n + p.eps);
    if (li == 0u) { rstd[row] = r; }
    for (var j = li; j < p.cols; j = j + 256u) {
        let xh = (x[base + j] - mean) * r;
        xhat[base + j] = xh;
        y[base + j] = fma(xh, g[j], bias[j]);
    }
}
`}

var kLayerNormBackward = &kernel{name: "layernorm_backward", binds: "rrrrrw", src: wgslReduce + `
struct P { rows: u32, cols: u32, hasres: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> xhat: array<f32>;
@group(0) @binding(2) var<storage, read> rstd: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
` + rowHead + `    let inv_n = 1.0 / f32(p.cols);
    var s: f32 = 0.0;
    var sx: f32 = 0.0;
    for (var j = li; j < p.cols; j = j + 256u) {
        let dxh = dy[base + j] * g[j];
        s = s + dxh;
        sx = fma(dxh, xhat[base + j], sx);
    }
    let mean_d = reduce_sum(li, s) * inv_n;
    let mean_dx = reduce_sum(li, sx) * inv_n;
    let r = rstd[row];
    for (var j = li; j < p.cols; j = j + 256u) {
        let dxh = dy[base + j] * g[j];
        var v = r * (dxh - mean_d - xhat[base + j] * mean_dx);
        if (p.hasres != 0u) { v = v + res[base + j]; }
        dx[base + j] = v;
    }
}
`}

var kLayerNormGrads = colKernel("layernorm_grads", "rrww", `
struct P { rows: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> xhat: array<f32>;
@group(0) @binding(2) var<storage, read_write> dg: array<f32>;
@group(0) @binding(3) var<storage, read_write> db: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
`, `            let v = dy[o];
            s = fma(v, xhat[o], s);
            s2 = s2 + v;`, `        dg[col] = dg[col] + t;
        db[col] = db[col] + t2;`)

// LayerNorm records the forward pass of model.LayerNorm on x [n, d] with
// gain g and bias bias [1, d]: y = xhat*g + bias, keeping xhat [n, d] and
// rstd [n, 1] (1/std per row) for the backward pass.
func (b *CommandBatch) LayerNorm(y, xhat, rstd, x, g, bias *Tensor, eps float64) {
	b.same("LayerNorm", x, y)
	b.same("LayerNorm", x, xhat)
	if g.Len() != x.Cols || bias.Len() != x.Cols || rstd.Len() != x.Rows {
		panic(fmt.Sprintf("gpu: LayerNorm of %s with gain %s, bias %s, rstd %s", x.Shape(), g.Shape(), bias.Shape(), rstd.Shape()))
	}
	k, gr := kLayerNorm, grid(x.Rows)
	switch {
	case x.Cols <= laneCols:
		k, gr = kLayerNormLane, laneGrid(x.Rows)
	case x.Cols <= regCols:
		k = kLayerNormReg
	}
	b.dispatch(k, gr, []uint32{uint32(x.Rows), uint32(x.Cols), fbits(eps), 0}, x.buf, g.buf, bias.buf, y.buf, xhat.buf, rstd.buf)
}

// LayerNormBackward records the backward pass of model.LayerNorm: dx from
// dy and the forward's xhat and rstd, and dg += sum(dy*xhat), db += sum(dy)
// over rows (accumulated, like the CPU parameter gradients).
func (b *CommandBatch) LayerNormBackward(dx, dg, db, dy, xhat, rstd, g *Tensor) {
	b.LayerNormBackwardAdd(dx, dg, db, dy, xhat, rstd, g, nil)
}

// LayerNormBackwardAdd is LayerNormBackward with the residual connection's
// gradient folded in: dx = LN'(dy) + res (res may be nil). In a pre-norm
// block the gradient into x is the LayerNorm's input gradient plus the
// gradient that skipped the sub-layer, so this saves a pass over dx.
func (b *CommandBatch) LayerNormBackwardAdd(dx, dg, db, dy, xhat, rstd, g, res *Tensor) {
	b.same("LayerNormBackward", dy, dx)
	b.same("LayerNormBackward", dy, xhat)
	if g.Len() != dy.Cols || dg.Len() != dy.Cols || db.Len() != dy.Cols || rstd.Len() != dy.Rows {
		panic(fmt.Sprintf("gpu: LayerNormBackward of %s with gain %s, dg %s, db %s, rstd %s", dy.Shape(), g.Shape(), dg.Shape(), db.Shape(), rstd.Shape()))
	}
	resBuf, hasres := b.d.dummyR, uint32(0)
	if res != nil {
		b.same("LayerNormBackwardAdd", dy, res)
		resBuf, hasres = res.buf, 1
	}
	k, gr := kLayerNormBackward, grid(dy.Rows)
	switch {
	case dy.Cols <= laneCols:
		k, gr = kLayerNormBackwardLane, laneGrid(dy.Rows)
	case dy.Cols <= regCols:
		k = kLayerNormBackwardReg
	}
	b.dispatch(k, gr, []uint32{uint32(dy.Rows), uint32(dy.Cols), hasres, 0}, dy.buf, xhat.buf, rstd.buf, g.buf, resBuf, dx.buf)
	b.dispatch(kLayerNormGrads, [3]uint32{uint32(ceilDiv(dy.Cols, 32)), 1, 1}, []uint32{uint32(dy.Rows), uint32(dy.Cols), 0, 0}, dy.buf, xhat.buf, dg.buf, db.buf)
}

// Cross-entropy, one workgroup per row: max, log-sum-exp, the row's loss
// and its dLogits row (softmax - onehot, times inv = 1/N for the mean). The
// in-place variant overwrites the logits with dLogits, which saves a
// [N, V] buffer; the target logit is read before anything is written.
const crossEntropyBody = `
struct P { rows: u32, cols: u32, inv: f32, pad0: u32 }
` + rowHead + `    let t = targets[row];
    let xt = x[base + t];
    var m = NEG_BIG;
    for (var j = li; j < p.cols; j = j + 256u) { m = max(m, x[base + j]); }
    m = reduce_max(li, m);
    var s: f32 = 0.0;
    for (var j = li; j < p.cols; j = j + 256u) { s = s + exp(x[base + j] - m); }
    let log_sum = log(reduce_sum(li, s)) + m;
    if (li == 0u) { losses[row] = log_sum - xt; }
    for (var j = li; j < p.cols; j = j + 256u) {
        var v = exp(x[base + j] - log_sum) * p.inv;
        if (j == t) { v = v - p.inv; }
        OUT[base + j] = v;
    }
}
`

var kCrossEntropy = &kernel{name: "cross_entropy", binds: "rrww", src: wgslReduce + `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<storage, read_write> dl: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
` + strings.Replace(crossEntropyBody, "OUT", "dl", 1)}

var kCrossEntropyInPlace = &kernel{name: "cross_entropy_inplace", binds: "wrw", src: wgslReduce + `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
` + strings.Replace(crossEntropyBody, "OUT", "x", 1)}

// CrossEntropy records train.CrossEntropy for logits [N, V] and the target
// ids in targets (an Index over V rows): losses [N, 1] gets each row's
// negative log-likelihood and dLogits the gradient of the mean loss. The
// mean itself is Sum(out, i, losses, false, false, 1/N). dLogits may be
// logits itself, in which case the logits are overwritten.
func (b *CommandBatch) CrossEntropy(losses, dLogits, logits *Tensor, targets *Index) {
	b.same("CrossEntropy", logits, dLogits)
	if losses.Len() != logits.Rows || targets.N != logits.Rows || targets.TableRows != logits.Cols {
		panic(fmt.Sprintf("gpu: CrossEntropy of %s with %d targets over %d ids into losses %s", logits.Shape(), targets.N, targets.TableRows, losses.Shape()))
	}
	params := []uint32{uint32(logits.Rows), uint32(logits.Cols), fbits(1 / float64(logits.Rows)), 0}
	if dLogits == logits {
		b.dispatch(kCrossEntropyInPlace, grid(logits.Rows), params, logits.buf, targets.ids, losses.buf)
		return
	}
	b.dispatch(kCrossEntropy, grid(logits.Rows), params, logits.buf, targets.ids, losses.buf, dLogits.buf)
}
