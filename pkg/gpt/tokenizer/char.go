package tokenizer

import (
	"encoding/json"
	"sort"
	"strings"
)

// UnknownToken is the id reserved for runes/symbols not in the vocabulary.
const UnknownToken = 0

// Char is a character-level tokenizer: one id per distinct rune seen in the
// corpus it was built from, plus id 0 reserved for unknown runes.
type Char struct {
	runes  []rune       // id -> rune (index 0 is the unknown placeholder)
	lookup map[rune]int // rune -> id
}

type charState struct {
	Runes string `json:"runes"` // vocabulary runes in id order, excluding the unknown slot
}

// NewChar builds a character tokenizer from the distinct runes in corpus.
func NewChar(corpus string) *Char {
	seen := map[rune]bool{}
	for _, r := range corpus {
		seen[r] = true
	}
	runes := make([]rune, 0, len(seen))
	for r := range seen {
		runes = append(runes, r)
	}
	sort.Slice(runes, func(i, j int) bool { return runes[i] < runes[j] })
	return newCharFromRunes(runes)
}

func newCharFromRunes(runes []rune) *Char {
	t := &Char{
		runes:  append([]rune{'�'}, runes...),
		lookup: make(map[rune]int, len(runes)),
	}
	for id, r := range t.runes {
		if id == UnknownToken {
			continue
		}
		t.lookup[r] = id
	}
	return t
}

func (t *Char) Encode(text string) []int {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		if id, ok := t.lookup[r]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, UnknownToken)
		}
	}
	return ids
}

func (t *Char) Decode(ids []int) string {
	var sb strings.Builder
	sb.Grow(len(ids))
	for _, id := range ids {
		if id < 0 || id >= len(t.runes) {
			id = UnknownToken
		}
		sb.WriteRune(t.runes[id])
	}
	return sb.String()
}

func (t *Char) VocabSize() int { return len(t.runes) }
func (t *Char) Kind() string   { return KindChar }

func (t *Char) state() charState { return charState{Runes: string(t.runes[1:])} }

func charFromState(raw json.RawMessage) (*Char, error) {
	var s charState
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, err
	}
	return newCharFromRunes([]rune(s.Runes)), nil
}
