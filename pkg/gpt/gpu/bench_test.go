package gpu

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// benchReps is how many dispatches each benchmark iteration records into
// one batch, so the per-dispatch time excludes the submit-and-wait cost the
// way a whole training step in one submission would.
const benchReps = 20

// benchBatch times n batches of benchReps calls of op and reports the rate
// per dispatch as unit, given the work (FLOPs or bytes) of one call.
func benchBatch(b *testing.B, d *Device, work float64, unit string, op func(bt *CommandBatch)) {
	b.Helper()
	run(b, d, op) // compile the pipelines and fill the bind-group cache
	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		run(b, d, func(bt *CommandBatch) {
			for r := 0; r < benchReps; r++ {
				op(bt)
			}
		})
	}
	per := time.Since(start).Seconds() / float64(b.N*benchReps)
	b.ReportMetric(work/per/1e9, unit)
	b.ReportMetric(per*1e6, "µs/dispatch")
}

// The matmul shapes of a training step at the test model size: the
// forward products, the input gradients (MatMulT against the weights) and
// the weight gradients (MatMulTA over the batch).
var benchMatMuls = []struct {
	name    string
	variant string
	m, k, n int
}{
	{"fc_fwd", "NN", tRows, tWidth, 4 * tWidth},
	{"proj_fwd", "NN", tRows, 4 * tWidth, tWidth},
	{"logits_fwd", "NT", tRows, tWidth, tVocab},
	{"fc_dx", "NT", tRows, 4 * tWidth, tWidth},
	{"logits_dx", "NN", tRows, tVocab, tWidth},
	{"fc_dW", "TN", tWidth, tRows, 4 * tWidth},
	{"emb_dW", "TN", tVocab, tRows, tWidth},
}

func BenchmarkMatMul(b *testing.B) {
	d := dev(b)
	rng := rand.New(rand.NewSource(1))
	for _, s := range benchMatMuls {
		b.Run(fmt.Sprintf("%s_%s_%dx%dx%d", s.variant, s.name, s.m, s.k, s.n), func(b *testing.B) {
			ar, ac, br, bc := s.m, s.k, s.k, s.n
			switch s.variant {
			case "NT":
				br, bc = s.n, s.k
			case "TN":
				ar, ac = s.k, s.m
			}
			a, bm, c := d.FromMat(randn(rng, ar, ac, 1)), d.FromMat(randn(rng, br, bc, 1)), d.NewTensor(s.m, s.n)
			defer func() { a.Release(); bm.Release(); c.Release() }()
			op := map[string]func(bt *CommandBatch){
				"NN": func(bt *CommandBatch) { bt.MatMul(c, a, bm, false) },
				"NT": func(bt *CommandBatch) { bt.MatMulT(c, a, bm, false) },
				"TN": func(bt *CommandBatch) { bt.MatMulTA(c, a, bm, false) },
			}[s.variant]
			benchBatch(b, d, 2*float64(s.m)*float64(s.k)*float64(s.n), "GFLOP/s", op)
		})
	}
}

// BenchmarkMatMulViaTranspose times the alternative to the dedicated
// transposed kernels: transpose the operand, then run the plain kernel.
// The GFLOP/s count only the product, so they compare directly with
// BenchmarkMatMul.
func BenchmarkMatMulViaTranspose(b *testing.B) {
	d := dev(b)
	rng := rand.New(rand.NewSource(1))
	for _, s := range benchMatMuls {
		if s.variant == "NN" {
			continue
		}
		b.Run(fmt.Sprintf("%s_%s_%dx%dx%d", s.variant, s.name, s.m, s.k, s.n), func(b *testing.B) {
			var a, bm, tmp *Tensor
			c := d.NewTensor(s.m, s.n)
			var op func(bt *CommandBatch)
			if s.variant == "NT" {
				a, bm, tmp = d.FromMat(randn(rng, s.m, s.k, 1)), d.FromMat(randn(rng, s.n, s.k, 1)), d.NewTensor(s.k, s.n)
				op = func(bt *CommandBatch) { bt.Transpose(tmp, bm); bt.MatMul(c, a, tmp, false) }
			} else {
				a, bm, tmp = d.FromMat(randn(rng, s.k, s.m, 1)), d.FromMat(randn(rng, s.k, s.n, 1)), d.NewTensor(s.m, s.k)
				op = func(bt *CommandBatch) { bt.Transpose(tmp, a); bt.MatMul(c, tmp, bm, false) }
			}
			defer func() { a.Release(); bm.Release(); tmp.Release(); c.Release() }()
			benchBatch(b, d, 2*float64(s.m)*float64(s.k)*float64(s.n), "GFLOP/s", op)
		})
	}
}

// BenchmarkLayerNorm reports GB/s of logical traffic: the forward reads x
// and writes y and xhat; the backward reads dy and xhat and writes dx, then
// reads dy and xhat again for the gain and bias gradients.
func BenchmarkLayerNorm(b *testing.B) {
	d := dev(b)
	rng := rand.New(rand.NewSource(1))
	n, dim := tRows, tWidth
	x, g, bias := d.FromMat(randn(rng, n, dim, 1)), d.FromMat(randn(rng, 1, dim, 1)), d.FromMat(randn(rng, 1, dim, 1))
	y, xhat, rstd := d.NewTensor(n, dim), d.NewTensor(n, dim), d.NewTensor(n, 1)
	dy, dx, dg, db := d.FromMat(randn(rng, n, dim, 1)), d.NewTensor(n, dim), d.NewTensor(1, dim), d.NewTensor(1, dim)
	defer func() {
		for _, t := range []*Tensor{x, g, bias, y, xhat, rstd, dy, dx, dg, db} {
			t.Release()
		}
	}()
	mat := float64(n*dim) * 4
	b.Run("forward", func(b *testing.B) {
		benchBatch(b, d, 3*mat, "GB/s", func(bt *CommandBatch) { bt.LayerNorm(y, xhat, rstd, x, g, bias, 1e-5) })
	})
	b.Run("backward", func(b *testing.B) {
		benchBatch(b, d, 5*mat, "GB/s", func(bt *CommandBatch) { bt.LayerNormBackward(dx, dg, db, dy, xhat, rstd, g) })
	})
}

// BenchmarkSoftmax runs the causal attention softmax on [B*H*T, T] scores
// and reports GB/s of one read and one write of the matrix.
func BenchmarkSoftmax(b *testing.B) {
	d := dev(b)
	rng := rand.New(rand.NewSource(1))
	rows, cols := tBatch*tHeads*tCtx, tCtx
	x, dAtt := d.FromMat(randn(rng, rows, cols, 1)), d.FromMat(randn(rng, rows, cols, 1))
	defer x.Release()
	defer dAtt.Release()
	mat := float64(rows*cols) * 4
	b.Run("forward", func(b *testing.B) {
		benchBatch(b, d, 2*mat, "GB/s", func(bt *CommandBatch) { bt.Softmax(x, 0.125, true) })
	})
	b.Run("backward", func(b *testing.B) {
		benchBatch(b, d, 3*mat, "GB/s", func(bt *CommandBatch) { bt.SoftmaxBackward(dAtt, x, 0.125, true) })
	})
}
