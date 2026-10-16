// Package tokenizer converts between text (or any symbol stream) and integer
// token ids. The GPT model itself only ever sees []int, so swapping the
// tokenizer is enough to retarget the same architecture from prose to, say, a
// tiny vocabulary of sensor readings and robot actions.
//
// Three implementations ship today:
//
//   - Char: one token per rune, built from a sample corpus.
//   - Symbol: one token per whitespace-separated symbol from a fixed list,
//     intended for small discrete vocabularies (sensor/action tokens).
//   - BPE: GPT-2 style byte-level byte-pair encoding learned from a corpus,
//     so common words and word pieces become single tokens.
//
// Another tokenizer can be added by implementing Tokenizer and registering
// its kind in Marshal/Unmarshal.
package tokenizer

import (
	"encoding/json"
	"fmt"
)

// Tokenizer maps between text and token ids.
type Tokenizer interface {
	// Encode converts text into token ids.
	Encode(text string) []int
	// Decode converts token ids back into text.
	Decode(ids []int) string
	// VocabSize is the number of distinct ids this tokenizer can produce.
	VocabSize() int
	// Kind is a short identifier ("char", "symbol", ...) used when saving.
	Kind() string
}

// Kinds understood by Marshal / Unmarshal.
const (
	KindChar   = "char"
	KindSymbol = "symbol"
	KindBPE    = "bpe"
)

type envelope struct {
	Kind  string          `json:"kind"`
	State json.RawMessage `json:"state"`
}

// Marshal serializes a tokenizer (with its kind) to JSON.
func Marshal(t Tokenizer) ([]byte, error) {
	var state any
	switch v := t.(type) {
	case *Char:
		state = v.state()
	case *Symbol:
		state = v.state()
	case *BPE:
		state = v.state()
	default:
		return nil, fmt.Errorf("tokenizer: cannot marshal kind %q", t.Kind())
	}
	raw, err := json.Marshal(state)
	if err != nil {
		return nil, err
	}
	return json.Marshal(envelope{Kind: t.Kind(), State: raw})
}

// Unmarshal restores a tokenizer written by Marshal.
func Unmarshal(data []byte) (Tokenizer, error) {
	var env envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, fmt.Errorf("tokenizer: bad envelope: %w", err)
	}
	switch env.Kind {
	case KindChar:
		return charFromState(env.State)
	case KindSymbol:
		return symbolFromState(env.State)
	case KindBPE:
		return bpeFromState(env.State)
	default:
		return nil, fmt.Errorf("tokenizer: unknown kind %q", env.Kind)
	}
}
