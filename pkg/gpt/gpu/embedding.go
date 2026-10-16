package gpu

import "fmt"

var kGather = &kernel{name: "gather", binds: "rrw", src: `
struct P { n: u32, cols: u32, acc: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
` + elemHead + `    let r = i / p.cols;
    let j = i % p.cols;
    let v = table[ids[r] * p.cols + j];
    if (p.acc != 0u) { out[i] = out[i] + v; } else { out[i] = v; }
}
`}

// One workgroup per (table row, 256-column chunk) sums the source rows
// that index it, in the order the CSR lists them.
var kScatterAdd = &kernel{name: "scatter_add", binds: "rrrw", src: `
struct P { rows: u32, cols: u32, chunks: u32, pad0: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> positions: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad: array<f32>;
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
    let w = flat(wid, nwg);
    let row = w / p.chunks;
    let j = (w % p.chunks) * 256u + li;
    if (row >= p.rows || j >= p.cols) { return; }
    let lo = offsets[row];
    let hi = offsets[row + 1u];
    if (lo == hi) { return; }
    var s: f32 = 0.0;
    for (var q = lo; q < hi; q = q + 1u) { s = s + src[positions[q] * p.cols + j]; }
    grad[row * p.cols + j] = grad[row * p.cols + j] + s;
}
`}

// Gather records out[r] (+)= table[ids[r]] for out [ids.N, D] and table
// [ids.TableRows, D]: the token embedding lookup, and with accumulate and
// ids[r] = r % T the position embedding added on top.
func (b *CommandBatch) Gather(out, table *Tensor, ids *Index, accumulate bool) {
	if out.Rows != ids.N || table.Rows != ids.TableRows || out.Cols != table.Cols {
		panic(fmt.Sprintf("gpu: Gather %d ids from %s into %s (index over %d rows)", ids.N, table.Shape(), out.Shape(), ids.TableRows))
	}
	b.dispatch(kGather, elems(out.Len()), []uint32{uint32(out.Len()), uint32(out.Cols), flag(accumulate), 0}, table.buf, ids.ids, out.buf)
}

// ScatterAdd records grad[ids[r]] += src[r] for every r: the gradient of
// Gather. Rows that share an id are summed in a fixed order, so the result
// is deterministic.
func (b *CommandBatch) ScatterAdd(grad, src *Tensor, ids *Index) {
	if src.Rows != ids.N || grad.Rows != ids.TableRows || src.Cols != grad.Cols {
		panic(fmt.Sprintf("gpu: ScatterAdd %s by %d ids into %s (index over %d rows)", src.Shape(), ids.N, grad.Shape(), ids.TableRows))
	}
	chunks := ceilDiv(src.Cols, 256)
	b.dispatch(kScatterAdd, grid(grad.Rows*chunks), []uint32{uint32(grad.Rows), uint32(grad.Cols), uint32(chunks), 0}, src.buf, ids.offsets, ids.positions, grad.buf)
}
