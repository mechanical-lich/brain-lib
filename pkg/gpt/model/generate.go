package model

import (
	"math"
	"math/rand"
	"sort"
)

// GenerateOptions controls sampling in Generate.
type GenerateOptions struct {
	// Temperature scales logits before sampling. <= 0 means greedy (argmax).
	Temperature float64
	// TopK restricts sampling to the K most likely tokens. <= 0 disables.
	TopK int
}

// Generate autoregressively extends prompt by maxNew tokens and returns the
// full sequence (prompt followed by the new tokens). The model re-runs a full
// forward pass over the trailing ContextLen tokens each step; there is no
// KV cache, which is fine for small models.
func (g *GPT) Generate(prompt []int, maxNew int, opts GenerateOptions, rng *rand.Rand) []int {
	out := append([]int(nil), prompt...)
	if len(out) == 0 {
		panic("model: Generate needs a non-empty prompt")
	}
	for i := 0; i < maxNew; i++ {
		window := out
		if len(window) > g.Config.ContextLen {
			window = window[len(window)-g.Config.ContextLen:]
		}
		logits := g.Forward([][]int{window})
		last := logits.Row(logits.Rows - 1)
		out = append(out, sample(last, opts, rng))
	}
	return out
}

// sample picks a token id from a logit vector according to opts.
func sample(logits []float64, opts GenerateOptions, rng *rand.Rand) int {
	return Sample(logits, opts, rng)
}

// Sample picks a token id from one row of next-token logits according to
// opts: greedy argmax when Temperature <= 0, otherwise a draw from the
// temperature-scaled softmax, restricted to the TopK largest when TopK > 0.
// It is exported so other backends (pkg/gpt/gpumodel) sample identically.
func Sample(logits []float64, opts GenerateOptions, rng *rand.Rand) int {
	if opts.Temperature <= 0 {
		best := 0
		for i, v := range logits {
			if v > logits[best] {
				best = i
			}
		}
		return best
	}
	scaled := make([]float64, len(logits))
	for i, v := range logits {
		scaled[i] = v / opts.Temperature
	}
	if opts.TopK > 0 && opts.TopK < len(scaled) {
		idx := make([]int, len(scaled))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return scaled[idx[a]] > scaled[idx[b]] })
		cutoff := scaled[idx[opts.TopK-1]]
		for i, v := range scaled {
			if v < cutoff {
				scaled[i] = math.Inf(-1)
			}
		}
	}
	maxv := math.Inf(-1)
	for _, v := range scaled {
		if v > maxv {
			maxv = v
		}
	}
	var sum float64
	for i, v := range scaled {
		scaled[i] = math.Exp(v - maxv)
		sum += scaled[i]
	}
	r := rng.Float64() * sum
	for i, p := range scaled {
		r -= p
		if r <= 0 {
			return i
		}
	}
	return len(scaled) - 1
}
