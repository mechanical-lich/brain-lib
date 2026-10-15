// Package data loads token streams and serves random training windows from
// them. It also knows how to fetch public-domain texts from Project
// Gutenberg to build a small corpus.
package data

import (
	"fmt"
	"math/rand"
)

// Dataset is a flat stream of token ids. Batches are random contiguous
// windows; the target for each position is the following token.
type Dataset struct {
	Tokens []int
}

// NewDataset wraps a token stream.
func NewDataset(tokens []int) *Dataset { return &Dataset{Tokens: tokens} }

// Len is the number of tokens.
func (d *Dataset) Len() int { return len(d.Tokens) }

// Batch samples batchSize random windows of contextLen tokens. It satisfies
// train.Batcher. Panics if the stream is shorter than contextLen+1.
func (d *Dataset) Batch(rng *rand.Rand, batchSize, contextLen int) (x, y [][]int) {
	if len(d.Tokens) < contextLen+1 {
		panic(fmt.Sprintf("data: %d tokens is too few for context length %d", len(d.Tokens), contextLen))
	}
	x = make([][]int, batchSize)
	y = make([][]int, batchSize)
	span := len(d.Tokens) - contextLen
	for i := range x {
		start := rng.Intn(span)
		x[i] = d.Tokens[start : start+contextLen]
		y[i] = d.Tokens[start+1 : start+contextLen+1]
	}
	return x, y
}

// Split divides tokens into a training and validation stream, with the last
// valFrac of the stream held out. valFrac of 0 returns a nil validation set.
func Split(tokens []int, valFrac float64) (train, val *Dataset) {
	if valFrac <= 0 {
		return NewDataset(tokens), nil
	}
	if valFrac >= 1 {
		panic("data: valFrac must be < 1")
	}
	n := int(float64(len(tokens)) * (1 - valFrac))
	return NewDataset(tokens[:n]), NewDataset(tokens[n:])
}
