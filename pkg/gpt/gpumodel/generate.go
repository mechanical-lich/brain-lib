package gpumodel

import (
	"fmt"
	"math/rand"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
)

// Generate autoregressively extends prompt by maxNew tokens on the GPU and
// returns the full sequence, with the same contract and sampler as
// model.GPT.Generate. Dropout is off while sampling regardless of training
// mode, which is restored afterward.
//
// Each token re-runs a full forward over the trailing window, padded on the
// right to a multiple of genPad positions so the activation buffers are
// reallocated a few times rather than once per token as the window grows
// (the causal mask makes the padding invisible to the positions before
// it), and only the last live row of logits is gathered and read back. One
// submission per token. Without a KV cache the cost per token is a forward
// over the whole window, about 8 ms at the BPE shape on an M5 for a full
// 256 window, so a 200-token sample takes about a second.
func (m *Model) Generate(prompt []int, maxNew int, opts model.GenerateOptions, rng *rand.Rand) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("gpumodel: Generate needs a non-empty prompt")
	}
	for _, id := range prompt {
		if id < 0 || id >= m.Config.VocabSize {
			return nil, fmt.Errorf("gpumodel: token id %d out of range [0, %d)", id, m.Config.VocabSize)
		}
	}
	out := append([]int(nil), prompt...)
	ctx := m.Config.ContextLen
	row := make([]float64, m.Config.VocabSize)
	for i := 0; i < maxNew; i++ {
		window := out
		if len(window) > ctx {
			window = window[len(window)-ctx:]
		}
		n := min(ctx, (len(window)+genPad-1)/genPad*genPad)
		padded := make([]int, n)
		copy(padded, window)
		a := m.ensure(1, n)
		a.ids.Set(padded)
		a.lastRow.Set([]int{len(window) - 1})
		r := m.newRec(false)
		m.recordForward(r, a, false, 0)
		r.b.Gather(a.lastLog, a.logits, a.lastRow, false)
		rb := r.b.Read(a.lastLog)
		if err := r.b.Submit(); err != nil {
			return nil, err
		}
		for j, v := range rb.Values() {
			row[j] = float64(v)
		}
		out = append(out, model.Sample(row, opts, rng))
	}
	return out, nil
}

// genPad is the granularity of the padded generation window.
const genPad = 64

// Generate samples from the stepper's GPU model; see Model.Generate.
func (s *Stepper) Generate(prompt []int, maxNew int, opts model.GenerateOptions, rng *rand.Rand) ([]int, error) {
	return s.Model.Generate(prompt, maxNew, opts, rng)
}
