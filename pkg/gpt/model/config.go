// Package model implements a minimal decoder-only transformer (GPT) with
// hand-written forward and backward passes on top of pkg/gpt/tensor.
//
// The model is corpus-agnostic: it consumes and predicts integer token ids
// and knows nothing about text. Vocabulary size, context length, depth, head
// count and width all come from Config, so the same code trains on prose with
// a character tokenizer or on a tiny sensor/action vocabulary.
package model

import "fmt"

// Config describes the shape of a GPT.
type Config struct {
	VocabSize  int `json:"vocabSize"`  // number of distinct token ids
	ContextLen int `json:"contextLen"` // maximum sequence length (block size)
	NumLayers  int `json:"numLayers"`  // transformer blocks
	NumHeads   int `json:"numHeads"`   // attention heads per block
	EmbedDim   int `json:"embedDim"`   // residual stream width
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error {
	switch {
	case c.VocabSize <= 0:
		return fmt.Errorf("model: VocabSize must be > 0, got %d", c.VocabSize)
	case c.ContextLen <= 0:
		return fmt.Errorf("model: ContextLen must be > 0, got %d", c.ContextLen)
	case c.NumLayers <= 0:
		return fmt.Errorf("model: NumLayers must be > 0, got %d", c.NumLayers)
	case c.NumHeads <= 0:
		return fmt.Errorf("model: NumHeads must be > 0, got %d", c.NumHeads)
	case c.EmbedDim <= 0:
		return fmt.Errorf("model: EmbedDim must be > 0, got %d", c.EmbedDim)
	case c.EmbedDim%c.NumHeads != 0:
		return fmt.Errorf("model: EmbedDim %d not divisible by NumHeads %d", c.EmbedDim, c.NumHeads)
	}
	return nil
}

// HeadDim is the per-head width.
func (c Config) HeadDim() int { return c.EmbedDim / c.NumHeads }
