package gpu

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// Training shapes the tests and benchmarks use: a 16 x 256 batch, width
// 384 with 6 heads, and a 2000-token BPE vocabulary, so nothing is a
// multiple of the 64-wide matmul tile in every dimension.
const (
	tBatch = 16
	tCtx   = 256
	tWidth = 384
	tHeads = 6
	tVocab = 2000
	tRows  = tBatch * tCtx
)

var (
	testDev    *Device
	testDevErr error
)

func TestMain(m *testing.M) {
	testDev, testDevErr = Open()
	if testDevErr == nil {
		fmt.Fprintf(os.Stderr, "gpu: testing on %s\n", testDev.Name())
	}
	code := m.Run()
	if testDev != nil {
		testDev.Close()
	}
	os.Exit(code)
}

// dev returns the shared test device, skipping the test when the machine
// has no GPU so the suite still passes there.
func dev(tb testing.TB) *Device {
	tb.Helper()
	if testDevErr != nil {
		if errors.Is(testDevErr, ErrNoAdapter) {
			tb.Skipf("skipping GPU kernel test: %v", testDevErr)
		}
		tb.Fatalf("gpu.Open: %v", testDevErr)
	}
	return testDev
}

func run(tb testing.TB, d *Device, fn func(b *CommandBatch)) {
	tb.Helper()
	if err := d.Run(fn); err != nil {
		tb.Fatal(err)
	}
}

func download(tb testing.TB, t *Tensor) *tensor.Mat {
	tb.Helper()
	m, err := t.ToMat()
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

// assertClose fails unless got matches want to tol relative to the largest
// magnitude in want (an absolute floor of 1 for all-small references would
// hide real errors, so the floor is tiny).
func assertClose(tb testing.TB, name string, got, want *tensor.Mat, tol float64) {
	tb.Helper()
	if !got.SameShape(want) {
		tb.Fatalf("%s: shape %s, want %s", name, got.Shape(), want.Shape())
	}
	var maxDiff, maxRef float64
	worst := 0
	for i, w := range want.Data {
		g := got.Data[i]
		if math.IsNaN(g) || math.IsInf(g, 0) {
			tb.Fatalf("%s: element %d is %v (want %v)", name, i, g, w)
		}
		if d := math.Abs(g - w); d > maxDiff {
			maxDiff, worst = d, i
		}
		maxRef = max(maxRef, math.Abs(w))
	}
	rel := maxDiff / max(maxRef, 1e-30)
	if rel > tol {
		tb.Fatalf("%s: max |gpu-cpu| %.3g at %d (gpu %v, cpu %v), relative %.2e > %.0e",
			name, maxDiff, worst, got.Data[worst], want.Data[worst], rel, tol)
	}
}

func randn(rng *rand.Rand, rows, cols int, std float64) *tensor.Mat {
	return tensor.Randn(rows, cols, std, rng)
}

func TestOpenSkipsWithoutGPU(t *testing.T) {
	// Documents the contract the rest of the suite relies on: Open either
	// works or reports ErrNoAdapter.
	if testDevErr != nil && !errors.Is(testDevErr, ErrNoAdapter) {
		t.Fatalf("Open failed with something other than ErrNoAdapter: %v", testDevErr)
	}
	if testDevErr != nil {
		t.Skipf("no GPU: %v", testDevErr)
	}
	t.Logf("device: %s", testDev.Name())
}

func TestUploadDownload(t *testing.T) {
	d := dev(t)
	rng := rand.New(rand.NewSource(1))
	for _, sh := range [][2]int{{1, 1}, {3, 5}, {tRows, tWidth}} {
		m := randn(rng, sh[0], sh[1], 1)
		x := d.FromMat(m)
		got := download(t, x)
		assertClose(t, "roundtrip", got, m, 1e-7)
		x.Release()
	}
}

func TestReadAtEndOfBatch(t *testing.T) {
	d := dev(t)
	x := d.NewTensor(1, 4)
	defer x.Release()
	b := d.Batch()
	b.Fill(x, 2.5)
	b.Scale(x, 2)
	r := b.ReadN(x, 2)
	if err := b.Submit(); err != nil {
		t.Fatal(err)
	}
	if v := r.Values(); len(v) != 2 || v[0] != 5 || v[1] != 5 {
		t.Fatalf("read %v, want [5 5]", v)
	}
}

// TestManyDispatchesOneBatch records more dispatches than one uniform arena
// chunk holds, each with its own parameters, and runs the batch twice so the
// second run goes through the bind-group cache.
func TestManyDispatchesOneBatch(t *testing.T) {
	d := dev(t)
	const n = arenaSlots + 500
	x := d.NewTensor(3, 7)
	y := d.NewTensor(3, 7)
	defer x.Release()
	defer y.Release()
	for pass := 0; pass < 2; pass++ {
		run(t, d, func(b *CommandBatch) {
			b.Fill(x, 1)
			b.Zero(y)
			for i := 0; i < n; i++ {
				b.AddScaled(y, x, float64(i%3)) // 0, 1, 2, 0, ...
			}
		})
		var want float64
		for i := 0; i < n; i++ {
			want += float64(i % 3)
		}
		got := download(t, y)
		for _, v := range got.Data {
			if v != want {
				t.Fatalf("pass %d: got %v, want %v", pass, v, want)
			}
		}
	}
}
