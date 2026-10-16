package gpu

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

// float32 against float64: products over k up to 4096 stay well inside this.
const tol = 1e-4

func TestMatMulVariants(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(2))
	// m, k, n of the product C[m,n] = op(A) op(B)
	shapes := [][3]int{
		{1, 1, 1}, {3, 5, 7}, {37, 53, 29}, {65, 17, 130}, {64, 16, 64},
		{tRows, tWidth, 4 * tWidth}, // MLP fc forward
		{tRows, tWidth, tVocab},     // logits (MatMulT against the embedding)
		{tWidth, tRows, 4 * tWidth}, // dW = x^T dy (MatMulTA)
	}
	type variant struct {
		name string
		// stored operand shapes, CPU reference, GPU op
		a, b func(m, k, n int) (int, int)
		cpu  func(a, b *tensor.Mat) *tensor.Mat
		gpu  func(bt *CommandBatch, c, a, b *Tensor, acc bool)
	}
	variants := []variant{
		{"MatMul", func(m, k, n int) (int, int) { return m, k }, func(m, k, n int) (int, int) { return k, n }, tensor.MatMul, (*CommandBatch).MatMul},
		{"MatMulT", func(m, k, n int) (int, int) { return m, k }, func(m, k, n int) (int, int) { return n, k }, tensor.MatMulT, (*CommandBatch).MatMulT},
		{"MatMulTA", func(m, k, n int) (int, int) { return k, m }, func(m, k, n int) (int, int) { return k, n }, tensor.MatMulTA, (*CommandBatch).MatMulTA},
	}
	for _, v := range variants {
		for _, s := range shapes {
			m, k, n := s[0], s[1], s[2]
			t.Run(fmt.Sprintf("%s/%dx%dx%d", v.name, m, k, n), func(t *testing.T) {
				ar, ac := v.a(m, k, n)
				br, bc := v.b(m, k, n)
				a, b := randn(rng, ar, ac, 1), randn(rng, br, bc, 1)
				want := v.cpu(a, b)
				c0 := randn(rng, m, n, 1)
				ga, gb := d.FromMat(a), d.FromMat(b)
				gc, gacc := d.NewTensor(m, n), d.FromMat(c0)
				defer func() { ga.Release(); gb.Release(); gc.Release(); gacc.Release() }()
				run(t, d, func(bt *CommandBatch) {
					v.gpu(bt, gc, ga, gb, false)
					v.gpu(bt, gacc, ga, gb, true)
				})
				assertClose(t, "C = product", download(t, gc), want, tol)
				want.Add(c0)
				assertClose(t, "C += product", download(t, gacc), want, tol)
			})
		}
	}
}

// TestGEMMAttentionLayout runs the per-(batch, head) attention products in
// one batched GEMM dispatch each, straight off the [B*T, 3D] qkv layout,
// and compares against tensor kernels run head by head.
func TestGEMMAttentionLayout(t *testing.T) {
	d := dev(t)
	for _, c := range []struct{ batch, seq, heads, hs int }{{2, 67, 3, 16}, {tBatch, tCtx, tHeads, tWidth / tHeads}} {
		t.Run(fmt.Sprintf("B%d_T%d_H%d_hs%d", c.batch, c.seq, c.heads, c.hs), func(t *testing.T) {
			rng := rand.New(rand.NewSource(3))
			B, T, H, hs := c.batch, c.seq, c.heads, c.hs
			D := H * hs
			qkv := randn(rng, B*T, 3*D, 1)
			dOut := randn(rng, B*T, D, 1) // gradient w.r.t. concatenated head outputs
			gq, gdo := d.FromMat(qkv), d.FromMat(dOut)
			scores := d.NewTensor(B*H*T, T) // row (b*H+h)*T + t
			out := d.NewTensor(B*T, D)
			dv := d.NewTensor(B*T, D)
			defer func() { gq.Release(); gdo.Release(); scores.Release(); out.Release(); dv.Release() }()
			// operand of head (b, h) at column col of a [B*T, ld] activation
			act := func(col, ld int) Operand {
				return Operand{Offset: col, LD: ld, Outer: T * ld, InnerStride: hs}
			}
			sq := Operand{Outer: H * T * T, InnerStride: T * T}
			run(t, d, func(bt *CommandBatch) {
				// scores = q k^T
				bt.GEMM(scores, gq, gq, GEMM{M: T, N: T, K: hs, TransB: true, Batch: B * H, Inner: H, A: act(0, 3*D), B: act(D, 3*D), C: sq})
				// out = scores v (scores used as attention weights directly)
				bt.GEMM(out, scores, gq, GEMM{M: T, N: hs, K: T, Batch: B * H, Inner: H, A: sq, B: act(2*D, 3*D), C: act(0, D)})
				// dV = scores^T dOut
				bt.GEMM(dv, scores, gdo, GEMM{M: T, N: hs, K: T, TransA: true, Batch: B * H, Inner: H, A: sq, B: act(0, D), C: act(0, D)})
			})
			wantS := tensor.New(B*H*T, T)
			wantO := tensor.New(B*T, D)
			wantDV := tensor.New(B*T, D)
			head := func(m *tensor.Mat, b, col int) *tensor.Mat {
				h := tensor.New(T, hs)
				for t := 0; t < T; t++ {
					copy(h.Row(t), m.Row(b*T + t)[col:col+hs])
				}
				return h
			}
			put := func(dst *tensor.Mat, src *tensor.Mat, b, col int) {
				for t := 0; t < T; t++ {
					copy(dst.Row(b*T + t)[col:col+hs], src.Row(t))
				}
			}
			for b := 0; b < B; b++ {
				for h := 0; h < H; h++ {
					q, k, v := head(qkv, b, h*hs), head(qkv, b, D+h*hs), head(qkv, b, 2*D+h*hs)
					s := tensor.MatMulT(q, k)
					copy(wantS.Data[(b*H+h)*T*T:], s.Data)
					put(wantO, tensor.MatMul(s, v), b, h*hs)
					put(wantDV, tensor.MatMulTA(s, head(dOut, b, h*hs)), b, h*hs)
				}
			}
			assertClose(t, "scores", download(t, scores), wantS, tol)
			assertClose(t, "out", download(t, out), wantO, tol)
			assertClose(t, "dV", download(t, dv), wantDV, tol)
		})
	}
}

func TestTranspose(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(4))
	for _, s := range [][2]int{{1, 1}, {17, 33}, {tWidth, tVocab}} {
		m := randn(rng, s[0], s[1], 1)
		x, y := d.FromMat(m), d.NewTensor(s[1], s[0])
		run(t, d, func(b *CommandBatch) { b.Transpose(y, x) })
		want := tensor.New(s[1], s[0])
		tensor.TransposeInto(want, m)
		assertClose(t, "transpose", download(t, y), want, 1e-7)
		x.Release()
		y.Release()
	}
}

func TestElementwise(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(5))
	for _, s := range [][2]int{{1, 1}, {7, 13}, {tRows, 4 * tWidth}} {
		rows, cols := s[0], s[1]
		t.Run(fmt.Sprintf("%dx%d", rows, cols), func(t *testing.T) {
			xm, ym := randn(rng, rows, cols, 1), randn(rng, rows, cols, 1)
			bias := randn(rng, 1, cols, 1)
			x, bs := d.FromMat(xm), d.FromMat(bias)
			defer x.Release()
			defer bs.Release()
			check := func(name string, op func(b *CommandBatch, y *Tensor), cpu func(y *tensor.Mat)) {
				y := d.FromMat(ym)
				defer y.Release()
				run(t, d, func(b *CommandBatch) { op(b, y) })
				want := ym.Clone()
				cpu(want)
				assertClose(t, name, download(t, y), want, 1e-6)
			}
			check("Fill", func(b *CommandBatch, y *Tensor) { b.Fill(y, -1.25) }, func(y *tensor.Mat) { y.Fill(-1.25) })
			check("Zero", func(b *CommandBatch, y *Tensor) { b.Zero(y) }, func(y *tensor.Mat) { y.Zero() })
			check("Copy", func(b *CommandBatch, y *Tensor) { b.Copy(y, x) }, func(y *tensor.Mat) { copy(y.Data, xm.Data) })
			check("Add", func(b *CommandBatch, y *Tensor) { b.Add(y, x) }, func(y *tensor.Mat) { y.Add(xm) })
			check("AddScaled", func(b *CommandBatch, y *Tensor) { b.AddScaled(y, x, -0.3) }, func(y *tensor.Mat) { y.AddScaled(xm, -0.3) })
			check("Scale", func(b *CommandBatch, y *Tensor) { b.Scale(y, 0.7) }, func(y *tensor.Mat) { y.Scale(0.7) })
			check("Mul", func(b *CommandBatch, y *Tensor) { b.Mul(y, x) }, func(y *tensor.Mat) {
				for i := range y.Data {
					y.Data[i] *= xm.Data[i]
				}
			})
			check("AddRow", func(b *CommandBatch, y *Tensor) { b.AddRow(y, bs) }, func(y *tensor.Mat) { y.AddRow(bias.Data) })

			// ColSum, plain and accumulating
			c0 := randn(rng, 1, cols, 1)
			out, acc := d.NewTensor(1, cols), d.FromMat(c0)
			defer out.Release()
			defer acc.Release()
			run(t, d, func(b *CommandBatch) {
				b.ColSum(out, x, false)
				b.ColSum(acc, x, true)
			})
			want := tensor.FromSlice(1, cols, xm.ColSum())
			assertClose(t, "ColSum", download(t, out), want, tol)
			want.Add(c0)
			assertClose(t, "ColSum accumulate", download(t, acc), want, tol)
		})
	}
}

// gelu and geluGrad mirror the unexported functions in pkg/gpt/model
// (mlp.go), the tanh approximation GPT-2 uses.
func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(0.7978845608028654*(x+0.044715*x*x*x)))
}

func geluGrad(x float64) float64 {
	const c, k = 0.7978845608028654, 0.044715
	u := c * (x + k*x*x*x)
	th := math.Tanh(u)
	return 0.5*(1+th) + 0.5*x*(1-th*th)*c*(1+3*k*x*x)
}

func TestGELU(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(6))
	for _, s := range [][2]int{{3, 5}, {tRows, 4 * tWidth}} {
		pre := randn(rng, s[0], s[1], 2)
		pre.Data[0] = 40 // large arguments must not overflow tanh
		if len(pre.Data) > 1 {
			pre.Data[1] = -40
		}
		dy := randn(rng, s[0], s[1], 1)
		gp, gy, gd := d.FromMat(pre), d.NewTensor(s[0], s[1]), d.FromMat(dy)
		run(t, d, func(b *CommandBatch) {
			b.GELU(gy, gp)
			b.GELUBackward(gd, gp)
		})
		wantY, wantD := tensor.New(s[0], s[1]), tensor.New(s[0], s[1])
		for i, v := range pre.Data {
			wantY.Data[i] = gelu(v)
			wantD.Data[i] = dy.Data[i] * geluGrad(v)
		}
		assertClose(t, "GELU", download(t, gy), wantY, 1e-5)
		assertClose(t, "GELUBackward", download(t, gd), wantD, 1e-5)
		gp.Release()
		gy.Release()
		gd.Release()
	}
}

// causalSoftmax and causalSoftmaxBackward mirror pkg/gpt/model
// (attention.go) on a full row: scale, softmax over the first n entries,
// zero the rest.
func causalSoftmax(row []float64, n int, scale float64) {
	maxv := math.Inf(-1)
	for s := 0; s < n; s++ {
		row[s] *= scale
		maxv = max(maxv, row[s])
	}
	var sum float64
	for s := 0; s < n; s++ {
		row[s] = math.Exp(row[s] - maxv)
		sum += row[s]
	}
	for s := 0; s < n; s++ {
		row[s] /= sum
	}
	clear(row[n:])
}

func causalSoftmaxBackward(dAtt, att []float64, n int, scale float64) {
	var dot float64
	for s := 0; s < n; s++ {
		dot += att[s] * dAtt[s]
	}
	for s := 0; s < n; s++ {
		dAtt[s] = att[s] * (dAtt[s] - dot) * scale
	}
	clear(dAtt[n:])
}

func TestSoftmax(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(7))
	cases := []struct {
		rows, cols int
		causal     bool
	}{
		{1, 1, true}, {2 * 3 * 67, 67, true}, {tBatch * tHeads * tCtx, tCtx, true},
		{5, 1, false}, {100, tVocab, false}, {33, 300, false},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("%dx%d_causal=%v", c.rows, c.cols, c.causal), func(t *testing.T) {
			scale := 1 / math.Sqrt(64)
			x := randn(rng, c.rows, c.cols, 3)
			dAtt := randn(rng, c.rows, c.cols, 1)
			gx, gd := d.FromMat(x), d.FromMat(dAtt)
			defer gx.Release()
			defer gd.Release()
			run(t, d, func(b *CommandBatch) {
				b.Softmax(gx, scale, c.causal)
				b.SoftmaxBackward(gd, gx, scale, c.causal)
			})
			wantA, wantD := x.Clone(), dAtt.Clone()
			for r := 0; r < c.rows; r++ {
				n := c.cols
				if c.causal {
					n = r%c.cols + 1
				}
				causalSoftmax(wantA.Row(r), n, scale)
				causalSoftmaxBackward(wantD.Row(r), wantA.Row(r), n, scale)
			}
			assertClose(t, "Softmax", download(t, gx), wantA, 1e-5)
			assertClose(t, "SoftmaxBackward", download(t, gd), wantD, 1e-4)
		})
	}
}

func TestLayerNorm(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(8))
	for _, s := range [][2]int{{1, 3}, {37, 53}, {5, 1000}, {tRows, tWidth}} {
		n, dim := s[0], s[1]
		t.Run(fmt.Sprintf("%dx%d", n, dim), func(t *testing.T) {
			ln := model.NewLayerNorm("ln", dim)
			copy(ln.G.Value.Data, randn(rng, 1, dim, 1).Data)
			copy(ln.B.Value.Data, randn(rng, 1, dim, 1).Data)
			// start the gradients non-zero to check they accumulate
			dg0, db0 := randn(rng, 1, dim, 1), randn(rng, 1, dim, 1)
			copy(ln.G.Grad.Data, dg0.Data)
			copy(ln.B.Grad.Data, db0.Data)
			x := randn(rng, n, dim, 2)
			for i := range x.Data {
				x.Data[i] += 0.5 // non-zero mean
			}
			dy := randn(rng, n, dim, 1)
			wantY := ln.Forward(x)
			wantDX := ln.Backward(dy)

			gx, gg, gb := d.FromMat(x), d.FromMat(ln.G.Value), d.FromMat(ln.B.Value)
			gy, xhat, rstd := d.NewTensor(n, dim), d.NewTensor(n, dim), d.NewTensor(n, 1)
			gdy, gdx, gdg, gdb := d.FromMat(dy), d.NewTensor(n, dim), d.FromMat(dg0), d.FromMat(db0)
			defer func() {
				for _, x := range []*Tensor{gx, gg, gb, gy, xhat, rstd, gdy, gdx, gdg, gdb} {
					x.Release()
				}
			}()
			run(t, d, func(b *CommandBatch) {
				b.LayerNorm(gy, xhat, rstd, gx, gg, gb, ln.Eps)
				b.LayerNormBackward(gdx, gdg, gdb, gdy, xhat, rstd, gg)
			})
			assertClose(t, "y", download(t, gy), wantY, 1e-5)
			assertClose(t, "dx", download(t, gdx), wantDX, tol)
			assertClose(t, "dg", download(t, gdg), ln.G.Grad, tol)
			assertClose(t, "db", download(t, gdb), ln.B.Grad, tol)
		})
	}
}

func TestCrossEntropy(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(9))
	for _, s := range [][2]int{{1, 1}, {3, 7}, {77, 300}, {tRows, tVocab}} {
		n, v := s[0], s[1]
		t.Run(fmt.Sprintf("%dx%d", n, v), func(t *testing.T) {
			logits := randn(rng, n, v, 3)
			targets := make([]int, n)
			for i := range targets {
				targets[i] = rng.Intn(v)
			}
			wantLoss, wantD := train.CrossEntropy(logits, targets)
			wantRow := tensor.New(n, 1)
			for i := 0; i < n; i++ {
				wantRow.Data[i], _ = train.CrossEntropy(logits.RowRange(i, i+1), targets[i:i+1])
			}

			gl, gl2 := d.FromMat(logits), d.FromMat(logits)
			ids := d.NewIndex(targets, v)
			losses, losses2, dl, mean := d.NewTensor(n, 1), d.NewTensor(n, 1), d.NewTensor(n, v), d.NewTensor(1, 2)
			defer func() {
				for _, x := range []*Tensor{gl, gl2, losses, losses2, dl, mean} {
					x.Release()
				}
				ids.Release()
			}()
			run(t, d, func(b *CommandBatch) {
				b.CrossEntropy(losses, dl, gl, ids)
				b.CrossEntropy(losses2, gl2, gl2, ids) // in place
				b.Sum(mean, 0, losses, false, false, 1/float64(n))
				b.Sum(mean, 1, losses2, false, false, 1/float64(n))
			})
			assertClose(t, "losses", download(t, losses), wantRow, tol)
			assertClose(t, "dLogits", download(t, dl), wantD, tol)
			assertClose(t, "losses (in place)", download(t, losses2), wantRow, tol)
			assertClose(t, "dLogits (in place)", download(t, gl2), wantD, tol)
			assertClose(t, "mean loss", download(t, mean), tensor.FromSlice(1, 2, []float64{wantLoss, wantLoss}), tol)
		})
	}
}

func TestSum(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(10))
	for _, n := range []int{1, 255, 1025, tRows, 4*1024*1024 + 7} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			x := randn(rng, 1, n, 1)
			for i := range x.Data {
				x.Data[i] += 0.1 // non-zero mean so the sum is not tiny
			}
			var sum, sq float64
			for _, v := range x.Data {
				sum += v
				sq += v * v
			}
			gx, out := d.FromMat(x), d.FromMat(tensor.FromSlice(1, 4, []float64{0, 0, 0, 10}))
			defer gx.Release()
			defer out.Release()
			run(t, d, func(b *CommandBatch) {
				b.Sum(out, 0, gx, false, false, 1)
				b.Sum(out, 1, gx, true, false, 1)
				b.Sum(out, 2, gx, false, false, 0.5)
				b.Sum(out, 3, gx, true, true, 1) // accumulate onto 10
			})
			want := tensor.FromSlice(1, 4, []float64{sum, sq, sum / 2, sq + 10})
			assertClose(t, "sums", download(t, out), want, 1e-5)
		})
	}
}

func TestGatherScatter(t *testing.T) {
	d := dev(t)
	for _, c := range []struct{ batch, seq, vocab, dim int }{{1, 1, 1, 1}, {3, 5, 11, 300}, {tBatch, tCtx, tVocab, tWidth}} {
		t.Run(fmt.Sprintf("B%d_T%d_V%d_D%d", c.batch, c.seq, c.vocab, c.dim), func(t *testing.T) {
			rng := rand.New(rand.NewSource(11))
			n := c.batch * c.seq
			tok, pos := randn(rng, c.vocab, c.dim, 1), randn(rng, c.seq, c.dim, 1)
			ids, posIDs := make([]int, n), make([]int, n)
			for r := range ids {
				ids[r] = rng.Intn(c.vocab)
				posIDs[r] = r % c.seq
			}
			ids[0] = ids[n-1] // at least one repeated id
			dx := randn(rng, n, c.dim, 1)
			tg0, pg0 := randn(rng, c.vocab, c.dim, 1), randn(rng, c.seq, c.dim, 1)

			// the lookups and their gradient as GPT.Forward and GPT.Backward do them
			wantX := tensor.New(n, c.dim)
			wantTG, wantPG := tg0.Clone(), pg0.Clone()
			for r, id := range ids {
				xr := wantX.Row(r)
				copy(xr, tok.Row(id))
				for j, v := range pos.Row(r % c.seq) {
					xr[j] += v
				}
				te, pe := wantTG.Row(id), wantPG.Row(r%c.seq)
				for j, v := range dx.Row(r) {
					te[j] += v
					pe[j] += v
				}
			}

			gt, gp, gx, gdx := d.FromMat(tok), d.FromMat(pos), d.NewTensor(n, c.dim), d.FromMat(dx)
			gtg, gpg := d.FromMat(tg0), d.FromMat(pg0)
			ix, px := d.NewIndex(ids, c.vocab), d.NewIndex(posIDs, c.seq)
			defer func() {
				for _, x := range []*Tensor{gt, gp, gx, gdx, gtg, gpg} {
					x.Release()
				}
				ix.Release()
				px.Release()
			}()
			run(t, d, func(b *CommandBatch) {
				b.Gather(gx, gt, ix, false)
				b.Gather(gx, gp, px, true)
				b.ScatterAdd(gtg, gdx, ix)
				b.ScatterAdd(gpg, gdx, px)
			})
			assertClose(t, "embeddings", download(t, gx), wantX, 1e-6)
			assertClose(t, "token grad", download(t, gtg), wantTG, 1e-5)
			assertClose(t, "position grad", download(t, gpg), wantPG, 1e-5)
		})
	}
}

func TestDropout(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(12))
	const p, seed = 0.1, 1234
	n, cols := tRows, tWidth
	x := randn(rng, n, cols, 1)
	dy := randn(rng, n, cols, 1)
	gx, mask, gx2, mask2, gdy := d.FromMat(x), d.NewTensor(n, cols), d.FromMat(x), d.NewTensor(n, cols), d.FromMat(dy)
	defer func() {
		for _, t := range []*Tensor{gx, mask, gx2, mask2, gdy} {
			t.Release()
		}
	}()
	run(t, d, func(b *CommandBatch) {
		b.Dropout(gx, mask, p, seed, 7)
		b.Dropout(gx2, mask2, p, seed, 8)
		b.Mul(gdy, mask) // backward
	})
	// the mask must be the Go mirror of the kernel's hash, exactly
	wantMask := tensor.New(n, cols)
	scale := float64(float32(1 / (1 - p)))
	dropped := 0
	for i := range wantMask.Data {
		u := float32(dropoutHash(uint32(i), seed, 7)>>8) * (1.0 / 16777216.0)
		if u >= float32(p) {
			wantMask.Data[i] = scale
		} else {
			dropped++
		}
	}
	gotMask := download(t, mask)
	assertClose(t, "mask", gotMask, wantMask, 0)
	if frac := float64(dropped) / float64(n*cols); math.Abs(frac-p) > 0.005 {
		t.Fatalf("dropped fraction %.4f, want about %.2f", frac, p)
	}
	wantY, wantD := x.Clone(), dy.Clone()
	for i, m := range wantMask.Data {
		wantY.Data[i] *= m
		wantD.Data[i] *= m
	}
	assertClose(t, "y", download(t, gx), wantY, 1e-6)
	assertClose(t, "dx", download(t, gdy), wantD, 1e-6)
	// a different step draws a different mask
	other := download(t, mask2)
	same := 0
	for i, m := range other.Data {
		if (m == 0) == (gotMask.Data[i] == 0) {
			same++
		}
	}
	if same == len(other.Data) {
		t.Fatal("dropout masks for steps 7 and 8 are identical")
	}
}

// TestAdamW runs three optimizer steps with and without gradient clipping
// and compares the weights with train.ClipGradNorm + train.AdamW.
func TestAdamW(t *testing.T) {
	d := dev(t)
	for _, maxNorm := range []float64{0, 1} {
		t.Run(fmt.Sprintf("maxNorm=%g", maxNorm), func(t *testing.T) {
			rng := rand.New(rand.NewSource(13))
			shapes := [][2]int{{tVocab, tWidth}, {1, tWidth}, {37, 53}}
			params := make([]*model.Param, len(shapes))
			for i, s := range shapes {
				params[i] = &model.Param{Name: fmt.Sprint(i), Value: randn(rng, s[0], s[1], 0.02), Grad: tensor.New(s[0], s[1]), Decay: i != 1}
			}
			cfg := train.DefaultAdamW()
			cfg.LR = 1e-3
			opt := train.NewAdamW(params, cfg)

			type gpuParam struct{ w, g, m, v *Tensor }
			gps := make([]gpuParam, len(params))
			for i, p := range params {
				gps[i] = gpuParam{d.FromMat(p.Value), d.NewTensor(p.Value.Rows, p.Value.Cols), d.NewTensor(p.Value.Rows, p.Value.Cols), d.NewTensor(p.Value.Rows, p.Value.Cols)}
			}
			sumSq := d.NewTensor(1, 1)
			defer func() {
				for _, g := range gps {
					g.w.Release()
					g.g.Release()
					g.m.Release()
					g.v.Release()
				}
				sumSq.Release()
			}()
			for step := 1; step <= 3; step++ {
				for i, p := range params {
					g := randn(rng, p.Value.Rows, p.Value.Cols, 0.05)
					copy(p.Grad.Data, g.Data)
					gps[i].g.Upload(g)
				}
				norm := train.ClipGradNorm(params, maxNorm)
				opt.Step()
				b := d.Batch()
				for i, g := range gps {
					b.Sum(sumSq, 0, g.g, true, i > 0, 1)
				}
				for i, g := range gps {
					b.AdamW(g.w, g.g, g.m, g.v, sumSq, AdamWStep{
						LR: cfg.LR, Beta1: cfg.Beta1, Beta2: cfg.Beta2, Eps: cfg.Eps,
						WeightDecay: cfg.WeightDecay, Decay: params[i].Decay, Step: step, MaxNorm: maxNorm,
					})
				}
				r := b.Read(sumSq)
				if err := b.Submit(); err != nil {
					t.Fatal(err)
				}
				if got := math.Sqrt(float64(r.Values()[0])); math.Abs(got-norm) > 1e-5*norm {
					t.Fatalf("step %d: gradient norm %v, want %v", step, got, norm)
				}
				for i, p := range params {
					assertClose(t, fmt.Sprintf("step %d param %d", step, i), download(t, gps[i].w), p.Value, tol)
				}
			}
		})
	}
}
