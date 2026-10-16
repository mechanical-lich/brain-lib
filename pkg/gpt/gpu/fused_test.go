package gpu

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

func geluCPU(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(0.7978845608028654*(x+0.044715*x*x*x)))
}

func geluGradCPU(x float64) float64 {
	u := 0.7978845608028654 * (x + 0.044715*x*x*x)
	t := math.Tanh(u)
	du := 0.7978845608028654 * (1 + 3*0.044715*x*x)
	return 0.5*(1+t) + 0.5*x*(1-t*t)*du
}

// TestGEMMEpilogue checks every fused epilogue against the plain product
// followed by the CPU operations it replaces.
func TestGEMMEpilogue(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(21))
	for _, s := range [][3]int{{37, 29, 53}, {tRows, tWidth, 4 * tWidth}} {
		m, k, n := s[0], s[1], s[2]
		t.Run(fmt.Sprintf("%dx%dx%d", m, k, n), func(t *testing.T) {
			a, bm := randn(rng, m, k, 1), randn(rng, k, n, 1)
			bias := randn(rng, 1, n, 1)
			res := randn(rng, m, n, 1)
			pre := randn(rng, m, n, 2)
			ga, gb, gbias, gres, gpre := d.FromMat(a), d.FromMat(bm), d.FromMat(bias), d.FromMat(res), d.FromMat(pre)
			c, c2 := d.NewTensor(m, n), d.NewTensor(m, n)
			defer func() {
				for _, x := range []*Tensor{ga, gb, gbias, gres, gpre, c, c2} {
					x.Release()
				}
			}()
			prod := tensor.MatMul(a, bm)
			withBias := prod.Clone()
			withBias.AddRow(bias.Data)
			base := GEMM{M: m, N: n, K: k, Bias: gbias}

			// bias only
			run(t, d, func(b *CommandBatch) { b.GEMM(c, ga, gb, base) })
			assertClose(t, "bias", download(t, c), withBias, tol)

			// bias + accumulate: c = c + prod + bias
			want := withBias.Clone()
			want.Add(withBias)
			g := base
			g.Accumulate = true
			run(t, d, func(b *CommandBatch) { b.GEMM(c, ga, gb, g) })
			assertClose(t, "bias+acc", download(t, c), want, tol)

			// gelu: c = pre, c2 = gelu(pre)
			g = base
			g.Mode, g.Out2 = EpilogueGELU, c2
			run(t, d, func(b *CommandBatch) { b.GEMM(c, ga, gb, g) })
			wantAct := withBias.Clone()
			for i, v := range wantAct.Data {
				wantAct.Data[i] = geluCPU(v)
			}
			assertClose(t, "gelu c", download(t, c), withBias, tol)
			assertClose(t, "gelu c2", download(t, c2), wantAct, 1e-5)

			// residual: c = res + prod + bias
			g = base
			g.Mode, g.Aux = EpilogueResidual, gres
			run(t, d, func(b *CommandBatch) { b.GEMM(c, ga, gb, g) })
			want = withBias.Clone()
			want.Add(res)
			assertClose(t, "residual", download(t, c), want, tol)

			// dropout + residual: c = res + mask*(prod+bias), c2 = mask (from the hash)
			g = base
			g.Mode, g.Aux, g.Out2 = EpilogueDropoutResidual, gres, c2
			g.Drop = DropoutParams{P: 0.3, Seed: 7, Step: 11}
			run(t, d, func(b *CommandBatch) { b.GEMM(c, ga, gb, g) })
			mask := download(t, c2)
			want = tensor.New(m, n)
			dropped := 0
			for i := range want.Data {
				h := dropoutHash(uint32(i), 7, 11)
				u := float64(h>>8) * (1.0 / 16777216.0)
				mk := float64(float32(1 / (1 - 0.3)))
				if u < 0.3 {
					mk, dropped = 0, dropped+1
				}
				if mask.Data[i] != mk {
					t.Fatalf("mask[%d] = %v, want %v", i, mask.Data[i], mk)
				}
				want.Data[i] = res.Data[i] + withBias.Data[i]*mk
			}
			if frac := float64(dropped) / float64(m*n); m*n > 1000 && math.Abs(frac-0.3) > 0.02 {
				t.Fatalf("drop rate %.3f, want about 0.3", frac)
			}
			assertClose(t, "dropout+residual", download(t, c), want, tol)

			// gelu grad: c = (prod + bias) * gelu'(pre)
			g = base
			g.Mode, g.Aux = EpilogueGELUGrad, gpre
			run(t, d, func(b *CommandBatch) { b.GEMM(c, ga, gb, g) })
			want = withBias.Clone()
			for i := range want.Data {
				want.Data[i] *= geluGradCPU(pre.Data[i])
			}
			assertClose(t, "gelu grad", download(t, c), want, tol)

			// no bias, no mode still works (the plain path through the same kernel)
			run(t, d, func(b *CommandBatch) { b.MatMul(c, ga, gb, false) })
			assertClose(t, "plain", download(t, c), prod, tol)
		})
	}
}

func TestMulInto(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(22))
	x, y := randn(rng, 33, 47, 1), randn(rng, 33, 47, 1)
	gx, gy, gz := d.FromMat(x), d.FromMat(y), d.NewTensor(33, 47)
	defer func() { gx.Release(); gy.Release(); gz.Release() }()
	run(t, d, func(b *CommandBatch) { b.MulInto(gz, gx, gy) })
	want := x.Clone()
	for i := range want.Data {
		want.Data[i] *= y.Data[i]
	}
	assertClose(t, "mul_into", download(t, gz), want, 1e-7)
}

// TestLayerNormBackwardAdd covers the residual fold and the wide-row
// fallback (cols > regCols) for both LayerNorm passes.
func TestLayerNormBackwardAdd(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(23))
	for _, s := range [][2]int{{19, 384}, {7, regCols + 100}} {
		n, dim := s[0], s[1]
		t.Run(fmt.Sprintf("%dx%d", n, dim), func(t *testing.T) {
			ln := model.NewLayerNorm("ln", dim)
			copy(ln.G.Value.Data, randn(rng, 1, dim, 1).Data)
			copy(ln.B.Value.Data, randn(rng, 1, dim, 1).Data)
			x := randn(rng, n, dim, 2)
			dy := randn(rng, n, dim, 1)
			res := randn(rng, n, dim, 1)
			wantY := ln.Forward(x)
			wantDX := ln.Backward(dy)
			wantDX.Add(res)
			gx, gg, gb := d.FromMat(x), d.FromMat(ln.G.Value), d.FromMat(ln.B.Value)
			gy, xhat, rstd := d.NewTensor(n, dim), d.NewTensor(n, dim), d.NewTensor(n, 1)
			gdy, gres, gdx, gdg, gdb := d.FromMat(dy), d.FromMat(res), d.NewTensor(n, dim), d.NewTensor(1, dim), d.NewTensor(1, dim)
			defer func() {
				for _, x := range []*Tensor{gx, gg, gb, gy, xhat, rstd, gdy, gres, gdx, gdg, gdb} {
					x.Release()
				}
			}()
			run(t, d, func(b *CommandBatch) {
				b.LayerNorm(gy, xhat, rstd, gx, gg, gb, ln.Eps)
				b.LayerNormBackwardAdd(gdx, gdg, gdb, gdy, xhat, rstd, gg, gres)
			})
			assertClose(t, "y", download(t, gy), wantY, 1e-5)
			assertClose(t, "dx+res", download(t, gdx), wantDX, tol)
			assertClose(t, "dg", download(t, gdg), ln.G.Grad, tol)
			assertClose(t, "db", download(t, gdb), ln.B.Grad, tol)
		})
	}
}

// TestSoftmaxWide exercises the multi-pass softmax kernels, which handle
// rows wider than regCols, against the single-pass ones' CPU reference.
func TestSoftmaxWide(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(24))
	T := regCols + 64
	x := randn(rng, 2*T, T, 1) // B*H = 2 heads of T queries, causal
	dAtt := randn(rng, 2*T, T, 1)
	gx, gd := d.FromMat(x), d.FromMat(dAtt)
	defer func() { gx.Release(); gd.Release() }()
	run(t, d, func(b *CommandBatch) {
		b.Softmax(gx, 0.5, true)
		b.SoftmaxBackward(gd, gx, 0.5, true)
	})
	wantAtt := x.Clone()
	wantD := dAtt.Clone()
	for r := 0; r < 2*T; r++ {
		n := r%T + 1
		row := wantAtt.Row(r)
		maxv := math.Inf(-1)
		for j := 0; j < n; j++ {
			row[j] *= 0.5
			maxv = max(maxv, row[j])
		}
		var sum float64
		for j := 0; j < n; j++ {
			row[j] = math.Exp(row[j] - maxv)
			sum += row[j]
		}
		for j := 0; j < n; j++ {
			row[j] /= sum
		}
		clear(row[n:])
		drow := wantD.Row(r)
		var dot float64
		for j := 0; j < n; j++ {
			dot += row[j] * drow[j]
		}
		for j := 0; j < n; j++ {
			drow[j] = row[j] * (drow[j] - dot) * 0.5
		}
		clear(drow[n:])
	}
	assertClose(t, "softmax wide", download(t, gx), wantAtt, tol)
	assertClose(t, "softmax backward wide", download(t, gd), wantD, tol)
}

// TestGEMMSplitK covers products that take the split-K path (few output
// tiles, deep K) in every variant, with and without accumulate, against the
// CPU kernels, plus the Offset/LD handling of a strided output.
func TestGEMMSplitK(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(25))
	type shape struct {
		variant string
		m, k, n int
	}
	for _, s := range []shape{{"TN", 384, 4096, 384}, {"TN", 384, 4096, 1536}, {"NN", 64, 2048, 64}, {"NT", 100, 4096, 70}} {
		t.Run(fmt.Sprintf("%s_%dx%dx%d", s.variant, s.m, s.k, s.n), func(t *testing.T) {
			tiles := ceilDiv(s.m, 64) * ceilDiv(s.n, 64)
			if splitKFor(tiles, s.k) < 2 {
				t.Fatalf("shape does not split (tiles %d)", tiles)
			}
			var a, bm, want *tensor.Mat
			g := GEMM{M: s.m, N: s.n, K: s.k}
			switch s.variant {
			case "TN":
				a, bm = randn(rng, s.k, s.m, 1), randn(rng, s.k, s.n, 1)
				want = tensor.MatMulTA(a, bm)
				g.TransA = true
			case "NT":
				a, bm = randn(rng, s.m, s.k, 1), randn(rng, s.n, s.k, 1)
				want = tensor.MatMulT(a, bm)
				g.TransB = true
			default:
				a, bm = randn(rng, s.m, s.k, 1), randn(rng, s.k, s.n, 1)
				want = tensor.MatMul(a, bm)
			}
			c0 := randn(rng, s.m, s.n, 1)
			ga, gb, gc := d.FromMat(a), d.FromMat(bm), d.FromMat(c0)
			defer func() { ga.Release(); gb.Release(); gc.Release() }()
			run(t, d, func(b *CommandBatch) { b.GEMM(gc, ga, gb, g) })
			assertClose(t, "splitk", download(t, gc), want, tol)
			g.Accumulate = true
			run(t, d, func(b *CommandBatch) { b.GEMM(gc, ga, gb, g) })
			want2 := want.Clone()
			want2.Add(want)
			assertClose(t, "splitk acc", download(t, gc), want2, tol)
		})
	}
	// strided output: write a 64x64 TN product into the middle of a wider tensor
	m, k, n := 64, 2048, 64
	a, bm := randn(rng, k, m, 1), randn(rng, k, n, 1)
	want := tensor.MatMulTA(a, bm)
	wide := tensor.New(m, 200)
	ga, gb, gw := d.FromMat(a), d.FromMat(bm), d.FromMat(wide)
	defer func() { ga.Release(); gb.Release(); gw.Release() }()
	run(t, d, func(b *CommandBatch) {
		b.GEMM(gw, ga, gb, GEMM{M: m, N: n, K: k, TransA: true, C: Operand{Offset: 50, LD: 200}})
	})
	got := download(t, gw)
	for i := 0; i < m; i++ {
		copy(wide.Row(i)[50:50+n], want.Row(i))
	}
	assertClose(t, "splitk strided", got, wide, tol)
}
