package model

import (
	"math/rand"
	"testing"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// BenchmarkStep measures one forward+backward at the cmd/gpt-train default
// shape (4 layers, width 128, context 128, batch 16, char-sized vocab).
func BenchmarkStep(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	cfg := Config{VocabSize: 92, ContextLen: 128, NumLayers: 4, NumHeads: 4, EmbedDim: 128}
	g := New(cfg, rng)
	tokens := randomTokens(rng, 16, 128, cfg.VocabSize)
	dLogits := tensor.Randn(16*128, cfg.VocabSize, 1e-3, rng)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.ZeroGrads()
		g.Forward(tokens)
		g.Backward(dLogits)
	}
}

func BenchmarkForward(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	cfg := Config{VocabSize: 92, ContextLen: 128, NumLayers: 4, NumHeads: 4, EmbedDim: 128}
	g := New(cfg, rng)
	tokens := randomTokens(rng, 16, 128, cfg.VocabSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Forward(tokens)
	}
}
