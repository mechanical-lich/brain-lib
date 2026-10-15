package tokenizer

import (
	"encoding/json"
	"strings"
)

// Symbol tokenizes whitespace-separated symbols against a fixed vocabulary.
// It is meant for small discrete vocabularies such as
// "SENSOR_LEFT_NEAR SENSOR_RIGHT_FAR ACTION_TURN_LEFT", where each symbol is
// one token. Id 0 is reserved for unknown symbols.
type Symbol struct {
	symbols []string
	lookup  map[string]int
}

type symbolState struct {
	Symbols []string `json:"symbols"`
}

// NewSymbol builds a symbol tokenizer. Order defines the ids (offset by one
// for the unknown slot); duplicates are ignored.
func NewSymbol(symbols []string) *Symbol {
	t := &Symbol{symbols: []string{"<unk>"}, lookup: map[string]int{}}
	for _, s := range symbols {
		if _, dup := t.lookup[s]; dup || s == "" {
			continue
		}
		t.lookup[s] = len(t.symbols)
		t.symbols = append(t.symbols, s)
	}
	return t
}

func (t *Symbol) Encode(text string) []int {
	fields := strings.Fields(text)
	ids := make([]int, len(fields))
	for i, f := range fields {
		if id, ok := t.lookup[f]; ok {
			ids[i] = id
		} else {
			ids[i] = UnknownToken
		}
	}
	return ids
}

func (t *Symbol) Decode(ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		if id < 0 || id >= len(t.symbols) {
			id = UnknownToken
		}
		parts[i] = t.symbols[id]
	}
	return strings.Join(parts, " ")
}

func (t *Symbol) VocabSize() int { return len(t.symbols) }
func (t *Symbol) Kind() string   { return KindSymbol }

func (t *Symbol) state() symbolState { return symbolState{Symbols: t.symbols[1:]} }

func symbolFromState(raw json.RawMessage) (*Symbol, error) {
	var s symbolState
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, err
	}
	return NewSymbol(s.Symbols), nil
}
