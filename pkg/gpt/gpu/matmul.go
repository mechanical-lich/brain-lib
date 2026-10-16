package gpu

import (
	"fmt"
	"strings"
)

// The matmul kernels start from the unrolled 64x64-tile kernel in
// experiments/gpu-spike: a 16x16 workgroup computes a 64x64 tile of C, each
// thread owning a 4x4 block in four named vec4 accumulators. Both operand
// tiles are staged in workgroup memory k-major as vec4s, A as [16 k][16 x 4
// rows] and B as [16 k][16 x 4 cols], so each inner step is two vector
// loads and four vector FMAs. (The spike staged A as scalars [m][k], four
// scalar loads per step; the vec4 A tile measured 1.38x faster on the M5.)
//
// The three variants differ only in how they fill those two tiles, so the
// transposed operands are handled by dedicated loads rather than by a
// separate transpose pass (see readme.md for the measurement). An operand
// stored along the tile's vec4 axis is read as one vec4 of four consecutive
// floats per thread; one stored the other way is gathered from four stored
// rows at one k, with consecutive threads taking consecutive k so the loads
// stay coalesced.
//
// Every variant is strided and batched: workgroup z multiplies the matrices
// at element offset Offset + (z / Inner)*Outer + (z % Inner)*InnerStride of
// each operand, with row stride LD. That covers per-(batch, head) attention
// products on the [B*T, 3D] qkv layout in one dispatch.
const matmulTemplate = `
struct P {
    m: u32, n: u32, k: u32, acc: u32,
    lda: u32, ldb: u32, ldc: u32, inner: u32,
    offa: u32, offb: u32, offc: u32, mode: u32,
    sao: u32, sai: u32, sbo: u32, sbi: u32,
    sco: u32, sci: u32, hasb: u32, seed: u32,
    step: u32, pad0: u32, dp: f32, dscale: f32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> aux: array<f32>;
@group(0) @binding(5) var<storage, read_write> c2: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

const BM: u32 = 64u;
const BN: u32 = 64u;
const BK: u32 = 16u;
var<workgroup> au: array<vec4<f32>, 256>; // BK x BM/4, [k][m/4]
var<workgroup> bu: array<vec4<f32>, 256>; // BK x BN/4, [k][n/4]

// finish applies the epilogue to one result element at offset o, column gc
// of its matrix: accumulate, bias, then the mode (see EpilogueMode).
fn finish(o: u32, gc: u32, v: f32) {
    var x = v;
    if (p.acc != 0u) { x = x + c[o]; }
    if (p.hasb != 0u) { x = x + bias[gc]; }
    if (p.mode == 1u) {
        c2[o] = gelu(x);
    } else if (p.mode == 2u) {
        let mk = dropmask(o, p.dp, p.dscale, p.seed, p.step);
        c2[o] = mk;
        x = fma(x, mk, aux[o]);
    } else if (p.mode == 3u) {
        x = x + aux[o];
    } else if (p.mode == 4u) {
        x = x * gelu_grad(aux[o]);
    }
    c[o] = x;
}

fn store_row(cbase: u32, gr: u32, gc: u32, v: vec4<f32>) {
    if (gr >= p.m) { return; }
    let o = cbase + gr * p.ldc + gc;
    if (gc < p.n) { finish(o, gc, v.x); }
    if (gc + 1u < p.n) { finish(o + 1u, gc + 1u, v.y); }
    if (gc + 2u < p.n) { finish(o + 2u, gc + 2u, v.z); }
    if (gc + 3u < p.n) { finish(o + 3u, gc + 3u, v.w); }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let zo = wid.z / p.inner;
    let zi = wid.z % p.inner;
    let abase = p.offa + zo * p.sao + zi * p.sai;
    let bbase = p.offb + zo * p.sbo + zi * p.sbi;
    let cbase = p.offc + zo * p.sco + zi * p.sci;
    let rowT = wid.y * BM;
    let colT = wid.x * BN;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let tiles = (p.k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * BK;
LOAD_A
LOAD_B
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let av = au[kk * 16u + lid.y];
            let bv = bu[kk * 16u + lid.x];
            acc0 = fma(vec4<f32>(av.x), bv, acc0);
            acc1 = fma(vec4<f32>(av.y), bv, acc1);
            acc2 = fma(vec4<f32>(av.z), bv, acc2);
            acc3 = fma(vec4<f32>(av.w), bv, acc3);
        }
        workgroupBarrier();
    }
    let row0 = rowT + lid.y * 4u;
    let col0 = colT + lid.x * 4u;
    store_row(cbase, row0, col0, acc0);
    store_row(cbase, row0 + 1u, col0, acc1);
    store_row(cbase, row0 + 2u, col0, acc2);
    store_row(cbase, row0 + 3u, col0, acc3);
}
`

const matmulTemplate128 = `
struct P {
    m: u32, n: u32, k: u32, acc: u32,
    lda: u32, ldb: u32, ldc: u32, inner: u32,
    offa: u32, offb: u32, offc: u32, mode: u32,
    sao: u32, sai: u32, sbo: u32, sbi: u32,
    sco: u32, sci: u32, hasb: u32, seed: u32,
    step: u32, pad0: u32, dp: f32, dscale: f32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> aux: array<f32>;
@group(0) @binding(5) var<storage, read_write> c2: array<f32>;
@group(0) @binding(6) var<uniform> p: P;

const BM: u32 = 128u;
const BN: u32 = 64u;
const BK: u32 = 16u;
var<workgroup> au: array<vec4<f32>, 512>; // BK x BM/4, [k][m/4]
var<workgroup> bu: array<vec4<f32>, 256>; // BK x BN/4, [k][n/4]

// finish applies the epilogue to one result element at offset o, column gc
// of its matrix: accumulate, bias, then the mode (see EpilogueMode).
fn finish(o: u32, gc: u32, v: f32) {
    var x = v;
    if (p.acc != 0u) { x = x + c[o]; }
    if (p.hasb != 0u) { x = x + bias[gc]; }
    if (p.mode == 1u) {
        c2[o] = gelu(x);
    } else if (p.mode == 2u) {
        let mk = dropmask(o, p.dp, p.dscale, p.seed, p.step);
        c2[o] = mk;
        x = fma(x, mk, aux[o]);
    } else if (p.mode == 3u) {
        x = x + aux[o];
    } else if (p.mode == 4u) {
        x = x * gelu_grad(aux[o]);
    }
    c[o] = x;
}

fn store_row(cbase: u32, gr: u32, gc: u32, v: vec4<f32>) {
    if (gr >= p.m) { return; }
    let o = cbase + gr * p.ldc + gc;
    if (gc < p.n) { finish(o, gc, v.x); }
    if (gc + 1u < p.n) { finish(o + 1u, gc + 1u, v.y); }
    if (gc + 2u < p.n) { finish(o + 2u, gc + 2u, v.z); }
    if (gc + 3u < p.n) { finish(o + 3u, gc + 3u, v.w); }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let zo = wid.z / p.inner;
    let zi = wid.z % p.inner;
    let abase = p.offa + zo * p.sao + zi * p.sai;
    let bbase = p.offb + zo * p.sbo + zi * p.sbi;
    let cbase = p.offc + zo * p.sco + zi * p.sci;
    let rowT = wid.y * BM;
    let colT = wid.x * BN;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var acc4 = vec4<f32>(0.0);
    var acc5 = vec4<f32>(0.0);
    var acc6 = vec4<f32>(0.0);
    var acc7 = vec4<f32>(0.0);
    let tiles = (p.k + BK - 1u) / BK;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let k0 = t * BK;
LOAD_A
LOAD_B
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let av = au[kk * 32u + lid.y];
            let aw = au[kk * 32u + lid.y + 16u];
            let bv = bu[kk * 16u + lid.x];
            acc0 = fma(vec4<f32>(av.x), bv, acc0);
            acc1 = fma(vec4<f32>(av.y), bv, acc1);
            acc2 = fma(vec4<f32>(av.z), bv, acc2);
            acc3 = fma(vec4<f32>(av.w), bv, acc3);
            acc4 = fma(vec4<f32>(aw.x), bv, acc4);
            acc5 = fma(vec4<f32>(aw.y), bv, acc5);
            acc6 = fma(vec4<f32>(aw.z), bv, acc6);
            acc7 = fma(vec4<f32>(aw.w), bv, acc7);
        }
        workgroupBarrier();
    }
    let row0 = rowT + lid.y * 4u;
    let row1 = rowT + 64u + lid.y * 4u;
    let col0 = colT + lid.x * 4u;
    store_row(cbase, row0, col0, acc0);
    store_row(cbase, row0 + 1u, col0, acc1);
    store_row(cbase, row0 + 2u, col0, acc2);
    store_row(cbase, row0 + 3u, col0, acc3);
    store_row(cbase, row1, col0, acc4);
    store_row(cbase, row1 + 1u, col0, acc5);
    store_row(cbase, row1 + 2u, col0, acc6);
    store_row(cbase, row1 + 3u, col0, acc7);
}
`

// 128-row tile loads: the A tile is 32 row groups of 4; each thread loads
// two of them (its group and the one 16 groups later).
const loadA128 = `
        {
            let kk = li % 16u;
            let r4 = li / 16u;
            let gk = k0 + kk;
            for (var half = 0u; half < 2u; half = half + 1u) {
                let rg = r4 + half * 16u;
                let gr = rowT + rg * 4u;
                var v = vec4<f32>(0.0);
                if (gk < p.k) {
                    let o = abase + gr * p.lda + gk;
                    if (gr < p.m) { v.x = a[o]; }
                    if (gr + 1u < p.m) { v.y = a[o + p.lda]; }
                    if (gr + 2u < p.m) { v.z = a[o + 2u * p.lda]; }
                    if (gr + 3u < p.m) { v.w = a[o + 3u * p.lda]; }
                }
                au[kk * 32u + rg] = v;
            }
        }`

const loadAT128 = `
        {
            let kk = li / 16u;
            let r4 = li % 16u;
            let gk = k0 + kk;
            for (var half = 0u; half < 2u; half = half + 1u) {
                let rg = r4 + half * 16u;
                let gr = rowT + rg * 4u;
                var v = vec4<f32>(0.0);
                if (gk < p.k) {
                    let o = abase + gk * p.lda + gr;
                    if (gr + 3u < p.m) {
                        v = vec4<f32>(a[o], a[o + 1u], a[o + 2u], a[o + 3u]);
                    } else {
                        if (gr < p.m) { v.x = a[o]; }
                        if (gr + 1u < p.m) { v.y = a[o + 1u]; }
                        if (gr + 2u < p.m) { v.z = a[o + 2u]; }
                    }
                }
                au[kk * 32u + rg] = v;
            }
        }`

func matmulKernel128(name, la, lb string) *kernel {
	src := strings.Replace(strings.Replace(matmulTemplate128, "LOAD_A", la, 1), "LOAD_B", lb, 1)
	return &kernel{name: name, binds: "rrwrrw", src: geluWGSL + dropoutWGSL + src}
}

var (
	kMatMulNN128 = matmulKernel128("matmul_nn_128", loadA128, loadB)
	kMatMulNT128 = matmulKernel128("matmul_nt_128", loadA128, loadBT)
	kMatMulTN128 = matmulKernel128("matmul_tn_128", loadAT128, loadB)
)

// tallTileRows is the M at or above which the 128-row tile kernels are
// used; below it the 64-row tile keeps more workgroups in flight.
var tallTileRows = 1024

// A stored [m][k]: each thread gathers one k from four stored rows;
// consecutive threads take consecutive k.
const loadA = `
        {
            let kk = li % 16u;
            let r4 = li / 16u;
            let gk = k0 + kk;
            let gr = rowT + r4 * 4u;
            var v = vec4<f32>(0.0);
            if (gk < p.k) {
                let o = abase + gr * p.lda + gk;
                if (gr < p.m) { v.x = a[o]; }
                if (gr + 1u < p.m) { v.y = a[o + p.lda]; }
                if (gr + 2u < p.m) { v.z = a[o + 2u * p.lda]; }
                if (gr + 3u < p.m) { v.w = a[o + 3u * p.lda]; }
            }
            au[kk * 16u + r4] = v;
        }`

// A stored [k][m] (A^T): one vec4 of four consecutive rows of the tile per
// thread.
const loadAT = `
        {
            let kk = li / 16u;
            let r4 = li % 16u;
            let gk = k0 + kk;
            let gr = rowT + r4 * 4u;
            var v = vec4<f32>(0.0);
            if (gk < p.k) {
                let o = abase + gk * p.lda + gr;
                if (gr + 3u < p.m) {
                    v = vec4<f32>(a[o], a[o + 1u], a[o + 2u], a[o + 3u]);
                } else {
                    if (gr < p.m) { v.x = a[o]; }
                    if (gr + 1u < p.m) { v.y = a[o + 1u]; }
                    if (gr + 2u < p.m) { v.z = a[o + 2u]; }
                }
            }
            au[kk * 16u + r4] = v;
        }`

// B stored [k][n]: one vec4 of four consecutive columns per thread.
const loadB = `
        {
            let kk = li / 16u;
            let c4 = li % 16u;
            let gk = k0 + kk;
            let gc = colT + c4 * 4u;
            var v = vec4<f32>(0.0);
            if (gk < p.k) {
                let o = bbase + gk * p.ldb + gc;
                if (gc + 3u < p.n) {
                    v = vec4<f32>(b[o], b[o + 1u], b[o + 2u], b[o + 3u]);
                } else {
                    if (gc < p.n) { v.x = b[o]; }
                    if (gc + 1u < p.n) { v.y = b[o + 1u]; }
                    if (gc + 2u < p.n) { v.z = b[o + 2u]; }
                }
            }
            bu[kk * 16u + c4] = v;
        }`

// B stored [n][k] (B^T): each thread gathers one k from four stored rows;
// consecutive threads take consecutive k.
const loadBT = `
        {
            let kk = li % 16u;
            let c4 = li / 16u;
            let gk = k0 + kk;
            let gc = colT + c4 * 4u;
            var v = vec4<f32>(0.0);
            if (gk < p.k) {
                let o = bbase + gc * p.ldb + gk;
                if (gc < p.n) { v.x = b[o]; }
                if (gc + 1u < p.n) { v.y = b[o + p.ldb]; }
                if (gc + 2u < p.n) { v.z = b[o + 2u * p.ldb]; }
                if (gc + 3u < p.n) { v.w = b[o + 3u * p.ldb]; }
            }
            bu[kk * 16u + c4] = v;
        }`

func matmulKernel(name, la, lb string) *kernel {
	src := strings.Replace(strings.Replace(matmulTemplate, "LOAD_A", la, 1), "LOAD_B", lb, 1)
	return &kernel{name: name, binds: "rrwrrw", src: geluWGSL + dropoutWGSL + src}
}

var (
	kMatMulNN = matmulKernel("matmul_nn", loadA, loadB)
	kMatMulNT = matmulKernel("matmul_nt", loadA, loadBT)
	kMatMulTN = matmulKernel("matmul_tn", loadAT, loadB)
)

// Operand describes where one matrix of a GEMM lives inside its tensor.
type Operand struct {
	Offset int // element offset of the first batch entry's matrix
	LD     int // row stride of the stored matrix; 0 means densely packed
	// Batch entry z starts at Offset + (z/Inner)*Outer + (z%Inner)*InnerStride.
	Outer, InnerStride int
}

// GEMM is a general strided, batched matrix product
// C (+)= op(A) @ op(B) for Batch independent entries, where op(A) is
// [M, K], op(B) is [K, N] and C is [M, N]. TransA means A is stored [K, M]
// and TransB means B is stored [N, K]; both together are not supported.
//
// The epilogue runs on each result element as it is stored, in this order:
// the accumulate (C += ...), then Bias added by column, then Mode. Fusing
// these into the store saves a full read-and-write pass over C per
// operation, which on a bandwidth-bound GPU is most of what those
// operations cost.
type GEMM struct {
	M, N, K        int
	TransA, TransB bool
	Accumulate     bool // C += product rather than C = product
	Batch          int  // number of entries; 0 means 1
	Inner          int  // entries per outer step (e.g. heads); 0 means 1
	A, B, C        Operand

	Bias *Tensor       // optional [1, N]: added to every row of the result
	Mode EpilogueMode  // what else to do with each result element
	Aux  *Tensor       // residual (EpilogueDropoutResidual, EpilogueResidual) or pre-activation (EpilogueGELUGrad); laid out like C
	Out2 *Tensor       // second output: gelu(result) (EpilogueGELU) or the dropout mask (EpilogueDropoutResidual); laid out like C
	Drop DropoutParams // for EpilogueDropoutResidual
}

// EpilogueMode selects what a GEMM does with each result element after the
// accumulate and the bias.
type EpilogueMode uint32

const (
	// EpilogueNone stores the result.
	EpilogueNone EpilogueMode = iota
	// EpilogueGELU stores the result in C and gelu(result) in Out2 (the
	// MLP's fc layer: pre-activation for backward, activation for the next
	// matmul).
	EpilogueGELU
	// EpilogueDropoutResidual draws a dropout mask from Drop, stores it in
	// Out2 and stores Aux + mask*result in C (a sub-layer's output through
	// dropout plus its residual connection).
	EpilogueDropoutResidual
	// EpilogueResidual stores Aux + result in C (the same without dropout).
	EpilogueResidual
	// EpilogueGELUGrad stores result * gelu'(Aux) in C: the gradient
	// through the GELU folded into the matmul that produces it.
	EpilogueGELUGrad
)

// DropoutParams are the dropout settings of a fused epilogue: the same
// counter-based mask as Dropout, indexed by the element's offset in C.
type DropoutParams struct {
	P          float64
	Seed, Step uint32
}

// GEMM records a strided batched matrix product. See GEMM.
func (b *CommandBatch) GEMM(c, a, bm *Tensor, g GEMM) {
	if g.TransA && g.TransB {
		panic("gpu: GEMM with both operands transposed")
	}
	if g.M <= 0 || g.N <= 0 || g.K <= 0 {
		panic(fmt.Sprintf("gpu: GEMM shape %dx%dx%d", g.M, g.N, g.K))
	}
	g.Batch, g.Inner = max(g.Batch, 1), max(g.Inner, 1)
	// Stored shapes as rows x cols, for strides and bounds.
	ar, ac := g.M, g.K
	if g.TransA {
		ar, ac = g.K, g.M
	}
	br, bc := g.K, g.N
	if g.TransB {
		br, bc = g.N, g.K
	}
	lda := g.A.check("A", a, ar, ac, g)
	ldb := g.B.check("B", bm, br, bc, g)
	ldc := g.C.check("C", c, g.M, g.N, g)
	k, bm64 := kMatMulNN, 64
	switch {
	case g.TransA:
		k = kMatMulTN
	case g.TransB:
		k = kMatMulNT
	}
	if g.M >= tallTileRows {
		bm64 = 128
		switch {
		case g.TransA:
			k = kMatMulTN128
		case g.TransB:
			k = kMatMulNT128
		default:
			k = kMatMulNN128
		}
	}
	if g.Batch == 1 && g.Mode == EpilogueNone && g.Bias == nil {
		tiles := ceilDiv(g.M, bm64) * ceilDiv(g.N, 64)
		if splits := splitKFor(tiles, g.K); splits > 1 {
			b.gemmSplitK(c, a, bm, g, k, bm64, lda, ldb, ldc, splits)
			return
		}
	}
	biasBuf, auxBuf, out2Buf := b.d.dummyR, b.d.dummyR, b.d.dummyW
	hasb := uint32(0)
	if g.Bias != nil {
		if g.Bias.Len() != g.N {
			panic(fmt.Sprintf("gpu: GEMM bias %s for N=%d", g.Bias.Shape(), g.N))
		}
		biasBuf, hasb = g.Bias.buf, 1
	}
	var dp, dscale float64
	switch g.Mode {
	case EpilogueNone:
		if g.Aux != nil || g.Out2 != nil {
			panic("gpu: GEMM Aux/Out2 given without a Mode")
		}
	case EpilogueGELU:
		b.same("GEMM Out2", c, g.Out2)
		out2Buf = g.Out2.buf
	case EpilogueDropoutResidual:
		b.same("GEMM Aux", c, g.Aux)
		b.same("GEMM Out2", c, g.Out2)
		if !(g.Drop.P >= 0 && g.Drop.P < 1) {
			panic(fmt.Sprintf("gpu: GEMM dropout probability %g", g.Drop.P))
		}
		auxBuf, out2Buf = g.Aux.buf, g.Out2.buf
		dp, dscale = g.Drop.P, 1/(1-g.Drop.P)
	case EpilogueResidual, EpilogueGELUGrad:
		b.same("GEMM Aux", c, g.Aux)
		auxBuf = g.Aux.buf
	default:
		panic(fmt.Sprintf("gpu: GEMM epilogue mode %d", g.Mode))
	}
	params := []uint32{
		uint32(g.M), uint32(g.N), uint32(g.K), flag(g.Accumulate),
		uint32(lda), uint32(ldb), uint32(ldc), uint32(g.Inner),
		uint32(g.A.Offset), uint32(g.B.Offset), uint32(g.C.Offset), uint32(g.Mode),
		uint32(g.A.Outer), uint32(g.A.InnerStride), uint32(g.B.Outer), uint32(g.B.InnerStride),
		uint32(g.C.Outer), uint32(g.C.InnerStride), hasb, g.Drop.Seed,
		g.Drop.Step, 0, fbits(dp), fbits(dscale),
	}
	b.dispatch(k, [3]uint32{uint32(ceilDiv(g.N, 64)), uint32(ceilDiv(g.M, bm64)), uint32(g.Batch)}, params, a.buf, bm.buf, c.buf, biasBuf, auxBuf, out2Buf)
}

// check validates that a rows x cols matrix per batch entry fits in t and
// returns its row stride.
func (o Operand) check(name string, t *Tensor, rows, cols int, g GEMM) int {
	ld := o.LD
	if ld == 0 {
		ld = cols
	}
	if ld < cols || o.Offset < 0 || o.Outer < 0 || o.InnerStride < 0 {
		panic(fmt.Sprintf("gpu: GEMM %s: ld %d for %d cols, offset %d, strides %d/%d", name, ld, cols, o.Offset, o.Outer, o.InnerStride))
	}
	maxOuter := (g.Batch - 1) / g.Inner
	maxInner := min(g.Inner, g.Batch) - 1
	last := o.Offset + maxOuter*o.Outer + maxInner*o.InnerStride + (rows-1)*ld + cols - 1
	if last >= t.Len() {
		panic(fmt.Sprintf("gpu: GEMM %s reaches element %d of a %s tensor", name, last, t.Shape()))
	}
	return ld
}

// MatMul records c (+)= a @ bm for a [m,k], bm [k,n], c [m,n].
func (b *CommandBatch) MatMul(c, a, bm *Tensor, accumulate bool) {
	if a.Cols != bm.Rows || c.Rows != a.Rows || c.Cols != bm.Cols {
		panic(fmt.Sprintf("gpu: MatMul %s = %s @ %s", c.Shape(), a.Shape(), bm.Shape()))
	}
	b.GEMM(c, a, bm, GEMM{M: a.Rows, N: bm.Cols, K: a.Cols, Accumulate: accumulate})
}

// MatMulT records c (+)= a @ bm^T for a [m,k], bm [n,k], c [m,n].
func (b *CommandBatch) MatMulT(c, a, bm *Tensor, accumulate bool) {
	if a.Cols != bm.Cols || c.Rows != a.Rows || c.Cols != bm.Rows {
		panic(fmt.Sprintf("gpu: MatMulT %s = %s @ %s^T", c.Shape(), a.Shape(), bm.Shape()))
	}
	b.GEMM(c, a, bm, GEMM{M: a.Rows, N: bm.Rows, K: a.Cols, TransB: true, Accumulate: accumulate})
}

// MatMulTA records c (+)= a^T @ bm for a [k,m], bm [k,n], c [m,n].
func (b *CommandBatch) MatMulTA(c, a, bm *Tensor, accumulate bool) {
	if a.Rows != bm.Rows || c.Rows != a.Cols || c.Cols != bm.Cols {
		panic(fmt.Sprintf("gpu: MatMulTA %s = %s^T @ %s", c.Shape(), a.Shape(), bm.Shape()))
	}
	b.GEMM(c, a, bm, GEMM{M: a.Cols, N: bm.Cols, K: a.Rows, TransA: true, Accumulate: accumulate})
}

var kTranspose = &kernel{name: "transpose", binds: "rw", src: `
struct P { rows: u32, cols: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

var<workgroup> tile: array<f32, 272>; // 16 x 17, padded against bank conflicts

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.y * 16u + lid.y;
    let c = wid.x * 16u + lid.x;
    if (r < p.rows && c < p.cols) { tile[lid.y * 17u + lid.x] = src[r * p.cols + c]; }
    workgroupBarrier();
    let orow = wid.x * 16u + lid.y; // row of dst = column of src
    let ocol = wid.y * 16u + lid.x;
    if (orow < p.cols && ocol < p.rows) { dst[orow * p.rows + ocol] = tile[lid.x * 17u + lid.y]; }
}
`}

// Transpose records dst = src^T for src [r,c], dst [c,r].
func (b *CommandBatch) Transpose(dst, src *Tensor) {
	if dst.Rows != src.Cols || dst.Cols != src.Rows {
		panic(fmt.Sprintf("gpu: Transpose %s into %s", src.Shape(), dst.Shape()))
	}
	b.dispatch(kTranspose, [3]uint32{uint32(ceilDiv(src.Cols, 16)), uint32(ceilDiv(src.Rows, 16)), 1},
		[]uint32{uint32(src.Rows), uint32(src.Cols), 0, 0}, src.buf, dst.buf)
}
