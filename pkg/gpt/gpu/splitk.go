package gpu

import "fmt"

// Split-K: a product with few output tiles (the weight gradients, whose
// M and N are the layer widths while K is the whole batch) launches too
// few workgroups to fill the GPU. Such a product is run as splits batch
// entries over K slices into a scratch of partial results, then reduced
// into C in a fixed order. It applies only to plain products: no bias, no
// epilogue mode, one batch entry.
const (
	splitKMinTiles = 512  // tiles at or above this fill the GPU without splitting
	splitKMinK     = 1024 // K below this is not worth splitting
)

// splitKFor returns how many K slices to use for a product with tiles
// output tiles and k depth, or 1 for no split.
func splitKFor(tiles, k int) int {
	if tiles >= splitKMinTiles || k < splitKMinK {
		return 1
	}
	for _, s := range []int{2, 4, 8} {
		if k%s != 0 || (k/s)%16 != 0 {
			continue
		}
		if tiles*s >= splitKMinTiles {
			return s
		}
	}
	if k%8 == 0 && (k/8)%16 == 0 {
		return 8
	}
	return 1
}

var kSplitKReduce = &kernel{name: "splitk_reduce", binds: "rw", src: `
struct P { mn: u32, n: u32, splits: u32, acc: u32, offc: u32, ldc: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let i = flat(wid, nwg) * 256u + li;
    if (i >= p.mn) { return; }
    var s: f32 = 0.0;
    for (var z = 0u; z < p.splits; z = z + 1u) { s = s + partial[z * p.mn + i]; }
    let o = p.offc + (i / p.n) * p.ldc + i % p.n;
    if (p.acc != 0u) { c[o] = c[o] + s; } else { c[o] = s; }
}
`}

// splitScratch returns the device's split-K partials scratch, grown to at
// least n floats. A scratch outgrown while a batch is recording is still
// bound by that batch's earlier dispatches, so it is retired and released
// only after the batch is submitted.
func (d *Device) splitScratch(n int) *Tensor {
	if d.splitk == nil || d.splitk.Len() < n {
		if d.splitk != nil {
			d.retired = append(d.retired, d.splitk)
		}
		d.splitk = d.NewTensor(n, 1)
	}
	return d.splitk
}

// releaseRetired frees scratch tensors outgrown during the last batch.
func (d *Device) releaseRetired() {
	for _, t := range d.retired {
		t.Release()
	}
	d.retired = nil
}

// gemmSplitK records g (already validated, one batch entry, no epilogue)
// as splits K-slices into scratch followed by the reduction into c.
func (b *CommandBatch) gemmSplitK(c, a, bm *Tensor, g GEMM, k *kernel, bmRows, lda, ldb, ldc, splits int) {
	kslice := g.K / splits
	mn := g.M * g.N
	scratch := b.d.splitScratch(splits * mn)
	sao, sbo := kslice, kslice*ldb // A [M,K]: slice is a column offset; B [K,N]: a row offset
	if g.TransA {
		sao = kslice * lda // A stored [K,M]: slice is a row offset
	}
	if g.TransB {
		sbo = kslice // B stored [N,K]: slice is a column offset
	}
	params := []uint32{
		uint32(g.M), uint32(g.N), uint32(kslice), 0,
		uint32(lda), uint32(ldb), uint32(g.N), 1,
		uint32(g.A.Offset), uint32(g.B.Offset), 0, 0,
		uint32(sao), 0, uint32(sbo), 0,
		uint32(mn), 0, 0, 0,
		0, 0, 0, 0,
	}
	b.dispatch(k, [3]uint32{uint32(ceilDiv(g.N, 64)), uint32(ceilDiv(g.M, bmRows)), uint32(splits)}, params,
		a.buf, bm.buf, scratch.buf, b.d.dummyR, b.d.dummyR, b.d.dummyW)
	b.dispatch(kSplitKReduce, elems(mn), []uint32{uint32(mn), uint32(g.N), uint32(splits), flag(g.Accumulate), uint32(g.C.Offset), uint32(ldc), 0, 0},
		scratch.buf, c.buf)
}

func init() {
	if splitKMinK%16 != 0 {
		panic(fmt.Sprintf("gpu: splitKMinK %d", splitKMinK))
	}
}
