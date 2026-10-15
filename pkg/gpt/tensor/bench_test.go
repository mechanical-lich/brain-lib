package tensor

import (
	"math/rand"
	"testing"
)

// The shapes below are the ones a cmd/gpt-train default step (width 128,
// MLP width 512, batch*context 2048) spends most of its time in.
const benchRows, benchIn, benchOut = 2048, 128, 512

func BenchmarkMatMul(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := Randn(benchRows, benchIn, 1, rng)
	w := Randn(benchIn, benchOut, 1, rng)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		MatMul(x, w)
	}
}

func BenchmarkMatMulT(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	dy := Randn(benchRows, benchOut, 1, rng)
	w := Randn(benchIn, benchOut, 1, rng)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		MatMulT(dy, w)
	}
}

func BenchmarkMatMulTA(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	x := Randn(benchRows, benchIn, 1, rng)
	dy := Randn(benchRows, benchOut, 1, rng)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		MatMulTA(x, dy)
	}
}

// BenchmarkMatMulTAWide is the MLP output projection's weight gradient, where
// the wide operand is a rather than b.
func BenchmarkMatMulTAWide(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	act := Randn(benchRows, benchOut, 1, rng)
	dy := Randn(benchRows, benchIn, 1, rng)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		MatMulTA(act, dy)
	}
}
