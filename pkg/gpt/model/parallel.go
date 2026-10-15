package model

import (
	"sort"
	"sync"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// parallelRows runs fn over chunks of [0, n) rows in parallel. Each call gets
// its own zeroed accumulators, one of length width per slice in sums, and
// after every chunk has finished the per-chunk accumulators are added into
// sums in ascending row order. Rows are therefore processed concurrently
// while the reductions (bias and gain gradients) stay deterministic for a
// given chunking, independent of which worker ran which chunk.
func parallelRows(n, width int, sums [][]float64, fn func(lo, hi int, acc [][]float64)) {
	type partial struct {
		lo  int
		acc [][]float64
	}
	var mu sync.Mutex
	var parts []partial
	tensor.Parallel(n, func(lo, hi int) {
		acc := make([][]float64, len(sums))
		for i := range acc {
			acc[i] = make([]float64, width)
		}
		fn(lo, hi, acc)
		mu.Lock()
		parts = append(parts, partial{lo, acc})
		mu.Unlock()
	})
	sort.Slice(parts, func(i, j int) bool { return parts[i].lo < parts[j].lo })
	for _, p := range parts {
		for s, acc := range p.acc {
			dst := sums[s]
			for j, v := range acc {
				dst[j] += v
			}
		}
	}
}
