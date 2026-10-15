package tensor

import "fmt"

// The three matmul kernels below share one idea: compute the output in 4x4
// register tiles. Each tile keeps 16 independent accumulators live across the
// whole k loop, so every loaded value feeds four multiply-adds and the
// compiler can pipeline them, and nothing is stored until the tile is done.
// Every accumulator still sums its products in increasing k order, so the
// results are bit-for-bit identical to the straightforward triple loop.
const (
	tileRows = 4 // output rows per register tile
	tileCols = 4 // output columns per register tile
)

// MatMul returns a @ b for a [n,k] and b [k,m].
func MatMul(a, b *Mat) *Mat {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("tensor: MatMul %s @ %s", a.Shape(), b.Shape()))
	}
	// Transposing b once costs k*m, which is nothing next to the n*k*m
	// multiply-adds, and lets the kernel stream both operands along k.
	out := New(a.Rows, b.Cols)
	rowDotTiled(out, a, transposeOf(b))
	return out
}

// MatMulT returns a @ b^T for a [n,k] and b [m,k].
func MatMulT(a, b *Mat) *Mat {
	if a.Cols != b.Cols {
		panic(fmt.Sprintf("tensor: MatMulT %s @ %s^T", a.Shape(), b.Shape()))
	}
	out := New(a.Rows, b.Rows)
	rowDotTiled(out, a, b)
	return out
}

// MatMulTA returns a^T @ b for a [k,n] and b [k,m].
func MatMulTA(a, b *Mat) *Mat {
	if a.Rows != b.Rows {
		panic(fmt.Sprintf("tensor: MatMulTA %s^T @ %s", a.Shape(), b.Shape()))
	}
	kk, n, m := a.Rows, a.Cols, b.Cols
	out := New(n, m)
	// Both operands are read down their columns here, so a naive loop
	// streams all of b once per output row. Instead each worker owns a block
	// of output columns and walks k in slabs of taKBlock rows: within a slab
	// the rows of a it needs and its columns of b stay in L1 while every 4x4
	// output tile accumulates the slab in registers.
	parallelTiles(m, tileCols, func(j0, j1 int) {
		for k0 := 0; k0 < kk; k0 += taKBlock {
			k1 := min(k0+taKBlock, kk)
			i := 0
			for ; i+tileRows <= n; i += tileRows {
				j := j0
				for ; j+tileCols <= j1; j += tileCols {
					outer4x4(out, a, b, i, j, k0, k1)
				}
				for ; j < j1; j++ {
					o0, o1, o2, o3 := out.Row(i), out.Row(i+1), out.Row(i+2), out.Row(i+3)
					c0, c1, c2, c3 := o0[j], o1[j], o2[j], o3[j]
					for k := k0; k < k1; k++ {
						ar := a.Data[k*n+i : k*n+i+4 : k*n+i+4]
						y := b.Data[k*m+j]
						c0 += ar[0] * y
						c1 += ar[1] * y
						c2 += ar[2] * y
						c3 += ar[3] * y
					}
					o0[j], o1[j], o2[j], o3[j] = c0, c1, c2, c3
				}
			}
			for ; i < n; i++ {
				oi := out.Row(i)[j0:j1]
				for k := k0; k < k1; k++ {
					x := a.Data[k*n+i]
					for j, y := range b.Row(k)[j0:j1] {
						oi[j] += x * y
					}
				}
			}
		}
	})
	return out
}

// taKBlock is the number of k rows MatMulTA accumulates per pass over its
// output block. 32 rows of a 512-wide operand is 128KB, which keeps both
// slabs near L1; it measured best of 8..64 on an Apple M5.
const taKBlock = 32

// outer4x4 accumulates sum_{k in [k0,k1)} a[k][i+r] * b[k][j+c] into
// out[i+r][j+c] for the 4x4 tile at (i, j) of a^T @ b. Rows i..i+3 of a and
// columns j..j+3 of b must be in range. Each output accumulates in increasing
// k order, continuing from the value already in out.
func outer4x4(out, a, b *Mat, i, j, k0, k1 int) {
	o0 := out.Row(i)[j : j+4 : j+4]
	o1 := out.Row(i + 1)[j : j+4 : j+4]
	o2 := out.Row(i + 2)[j : j+4 : j+4]
	o3 := out.Row(i + 3)[j : j+4 : j+4]
	c00, c01, c02, c03 := o0[0], o0[1], o0[2], o0[3]
	c10, c11, c12, c13 := o1[0], o1[1], o1[2], o1[3]
	c20, c21, c22, c23 := o2[0], o2[1], o2[2], o2[3]
	c30, c31, c32, c33 := o3[0], o3[1], o3[2], o3[3]
	n, m := a.Cols, b.Cols
	ad, bd := a.Data, b.Data
	ao, bo := k0*n+i, k0*m+j
	for k := k0; k < k1; k++ {
		ar := ad[ao : ao+4 : ao+4]
		br := bd[bo : bo+4 : bo+4]
		ao += n
		bo += m
		x0, x1, x2, x3 := ar[0], ar[1], ar[2], ar[3]
		y0, y1, y2, y3 := br[0], br[1], br[2], br[3]
		c00 += x0 * y0
		c01 += x0 * y1
		c02 += x0 * y2
		c03 += x0 * y3
		c10 += x1 * y0
		c11 += x1 * y1
		c12 += x1 * y2
		c13 += x1 * y3
		c20 += x2 * y0
		c21 += x2 * y1
		c22 += x2 * y2
		c23 += x2 * y3
		c30 += x3 * y0
		c31 += x3 * y1
		c32 += x3 * y2
		c33 += x3 * y3
	}
	o0[0], o0[1], o0[2], o0[3] = c00, c01, c02, c03
	o1[0], o1[1], o1[2], o1[3] = c10, c11, c12, c13
	o2[0], o2[1], o2[2], o2[3] = c20, c21, c22, c23
	o3[0], o3[1], o3[2], o3[3] = c30, c31, c32, c33
}

// rowDotTiled sets out[i][j] = Dot(a.Row(i), b.Row(j)) for a [n,k], b [m,k].
func rowDotTiled(out, a, b *Mat) {
	n, m := a.Rows, b.Rows
	parallelTiles(n, tileRows, func(lo, hi int) {
		i := lo
		for ; i+tileRows <= hi; i += tileRows {
			a0, a1, a2, a3 := a.Row(i), a.Row(i+1), a.Row(i+2), a.Row(i+3)
			o0, o1, o2, o3 := out.Row(i), out.Row(i+1), out.Row(i+2), out.Row(i+3)
			j := 0
			for ; j+tileCols <= m; j += tileCols {
				dot4x4(a0, a1, a2, a3, b.Row(j), b.Row(j+1), b.Row(j+2), b.Row(j+3),
					o0[j:j+4], o1[j:j+4], o2[j:j+4], o3[j:j+4])
			}
			for ; j < m; j++ {
				bj := b.Row(j)
				o0[j], o1[j], o2[j], o3[j] = Dot(a0, bj), Dot(a1, bj), Dot(a2, bj), Dot(a3, bj)
			}
		}
		for ; i < hi; i++ {
			ai, oi := a.Row(i), out.Row(i)
			for j := 0; j < m; j++ {
				oi[j] = Dot(ai, b.Row(j))
			}
		}
	})
}

// dot4x4 writes the 16 inner products of rows a0..a3 with rows b0..b3 into
// o0..o3 (o_r[c] = a_r . b_c). All eight input rows have the same length and
// each output slice holds at least 4 values.
func dot4x4(a0, a1, a2, a3, b0, b1, b2, b3, o0, o1, o2, o3 []float64) {
	// Reslicing everything to len(a0) lets the compiler drop the bounds
	// checks inside the loop.
	k := len(a0)
	a1, a2, a3 = a1[:k], a2[:k], a3[:k]
	b0, b1, b2, b3 = b0[:k], b1[:k], b2[:k], b3[:k]
	var c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33 float64
	for i := range a0 {
		x0, x1, x2, x3 := a0[i], a1[i], a2[i], a3[i]
		y0, y1, y2, y3 := b0[i], b1[i], b2[i], b3[i]
		c00 += x0 * y0
		c01 += x0 * y1
		c02 += x0 * y2
		c03 += x0 * y3
		c10 += x1 * y0
		c11 += x1 * y1
		c12 += x1 * y2
		c13 += x1 * y3
		c20 += x2 * y0
		c21 += x2 * y1
		c22 += x2 * y2
		c23 += x2 * y3
		c30 += x3 * y0
		c31 += x3 * y1
		c32 += x3 * y2
		c33 += x3 * y3
	}
	o0, o1, o2, o3 = o0[:4], o1[:4], o2[:4], o3[:4]
	o0[0], o0[1], o0[2], o0[3] = c00, c01, c02, c03
	o1[0], o1[1], o1[2], o1[3] = c10, c11, c12, c13
	o2[0], o2[1], o2[2], o2[3] = c20, c21, c22, c23
	o3[0], o3[1], o3[2], o3[3] = c30, c31, c32, c33
}

// transposeOf returns a new matrix holding m^T.
func transposeOf(m *Mat) *Mat {
	t := New(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j, v := range m.Row(i) {
			t.Data[j*m.Rows+i] = v
		}
	}
	return t
}

// parallelTiles is Parallel over [0, n) with chunk boundaries aligned to
// multiples of tile, so every chunk but the last consists of whole tiles.
func parallelTiles(n, tile int, fn func(lo, hi int)) {
	Parallel((n+tile-1)/tile, func(lo, hi int) {
		fn(lo*tile, min(hi*tile, n))
	})
}
