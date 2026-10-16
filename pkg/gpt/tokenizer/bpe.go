package tokenizer

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// bpeBase is the number of base ids: one per byte value.
const bpeBase = 256

// BPE is a byte-level byte-pair-encoding tokenizer in the style of GPT-2.
// Ids 0-255 are the raw byte values, so any input encodes without an unknown
// token (id 0 is the NUL byte, not "unknown"). Each learned merge adds one id,
// 256 + its rank, standing for the concatenation of two existing tokens.
//
// Text is first split into words by preTokenize, which keeps a leading space
// on each word and separates punctuation runs and newlines, so a merge never
// spans a word boundary. Encode then applies merges inside each word in the
// order they were learned.
type BPE struct {
	merges []bpePair       // rank -> pair; merged id = bpeBase + rank
	ranks  map[bpePair]int // pair -> rank
	vocab  [][]byte        // id -> bytes

	mu    sync.Mutex
	cache map[string][]int // word -> ids
}

type bpePair struct{ a, b int }

type bpeState struct {
	Merges [][2]int `json:"merges"` // in rank order; merge i produces id 256+i
}

// NewBPE learns merges from corpus until the vocabulary holds vocabSize ids
// (256 base bytes plus merges), or until every word is a single token.
//
// Learning follows Sennrich et al.: corpus words are counted once, and each
// step merges the most frequent adjacent pair across that word dictionary,
// weighted by word frequency. Ties go to the pair with the smaller ids, so the
// result is deterministic.
func NewBPE(corpus string, vocabSize int) *BPE {
	return newBPEFromMerges(learnMerges(corpus, vocabSize-bpeBase))
}

func newBPEFromMerges(merges []bpePair) *BPE {
	t := &BPE{
		merges: merges,
		ranks:  make(map[bpePair]int, len(merges)),
		vocab:  make([][]byte, bpeBase, bpeBase+len(merges)),
		cache:  map[string][]int{},
	}
	for b := range bpeBase {
		t.vocab[b] = []byte{byte(b)}
	}
	for rank, p := range merges {
		t.ranks[p] = rank
		tok := make([]byte, 0, len(t.vocab[p.a])+len(t.vocab[p.b]))
		t.vocab = append(t.vocab, append(append(tok, t.vocab[p.a]...), t.vocab[p.b]...))
	}
	return t
}

func (t *BPE) Encode(text string) []int {
	ids := make([]int, 0, len(text)/3)
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, w := range preTokenize(text) {
		enc, ok := t.cache[w]
		if !ok {
			enc = t.encodeWord(w)
			t.cache[w] = enc
		}
		ids = append(ids, enc...)
	}
	return ids
}

// encodeWord repeatedly merges the lowest-ranked adjacent pair in w until no
// learned merge applies, which reproduces the order merges were learned in.
func (t *BPE) encodeWord(w string) []int {
	syms := make([]int, len(w))
	for i := range len(w) {
		syms[i] = int(w[i])
	}
	for len(syms) > 1 {
		best, bestRank := -1, len(t.merges)
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[bpePair{syms[i], syms[i+1]}]; ok && r < bestRank {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			break
		}
		syms = mergeAll(syms, t.merges[bestRank], bpeBase+bestRank)
	}
	return syms
}

// Decode concatenates the bytes of each id. Ids outside the vocabulary
// decode as U+FFFD. A generated sequence can end partway through a multibyte
// rune, so the result is not guaranteed to be valid UTF-8.
func (t *BPE) Decode(ids []int) string {
	var sb strings.Builder
	sb.Grow(len(ids) * 4)
	for _, id := range ids {
		if id < 0 || id >= len(t.vocab) {
			sb.WriteRune(utf8.RuneError)
			continue
		}
		sb.Write(t.vocab[id])
	}
	return sb.String()
}

func (t *BPE) VocabSize() int { return len(t.vocab) }
func (t *BPE) Kind() string   { return KindBPE }

// Vocab returns each id's token as a string, indexed by id. Entries 0-255 are
// single bytes; later ones are learned merges in rank order.
func (t *BPE) Vocab() []string {
	out := make([]string, len(t.vocab))
	for id, b := range t.vocab {
		out[id] = string(b)
	}
	return out
}

func (t *BPE) state() bpeState {
	s := bpeState{Merges: make([][2]int, len(t.merges))}
	for i, p := range t.merges {
		s.Merges[i] = [2]int{p.a, p.b}
	}
	return s
}

func bpeFromState(raw json.RawMessage) (*BPE, error) {
	var s bpeState
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, err
	}
	merges := make([]bpePair, len(s.Merges))
	for i, m := range s.Merges {
		limit := bpeBase + i // a merge can only use ids that exist before it
		if m[0] < 0 || m[0] >= limit || m[1] < 0 || m[1] >= limit {
			return nil, fmt.Errorf("tokenizer: bpe merge %d references id outside [0, %d)", i, limit)
		}
		merges[i] = bpePair{m[0], m[1]}
	}
	return newBPEFromMerges(merges), nil
}

// mergeAll replaces every non-overlapping occurrence of p in syms, scanning
// left to right, with id. It reuses syms' backing array.
func mergeAll(syms []int, p bpePair, id int) []int {
	out := syms[:0]
	for i := 0; i < len(syms); i++ {
		if i+1 < len(syms) && syms[i] == p.a && syms[i+1] == p.b {
			out = append(out, id)
			i++
		} else {
			out = append(out, syms[i])
		}
	}
	return out
}

// learnMerges runs up to n merge steps over the word dictionary of corpus.
//
// Pair counts are kept incrementally: each step touches only the words that
// contain the chosen pair, subtracting their old pairs and adding their new
// ones, instead of recounting the whole dictionary.
func learnMerges(corpus string, n int) []bpePair {
	if n <= 0 {
		return nil
	}
	freq := map[string]int{}
	for _, w := range preTokenize(corpus) {
		freq[w]++
	}
	type word struct {
		syms  []int
		count int
	}
	words := make([]word, 0, len(freq))
	for w, c := range freq {
		if len(w) < 2 {
			continue // a single byte has no pairs to merge
		}
		syms := make([]int, len(w))
		for i := range len(w) {
			syms[i] = int(w[i])
		}
		words = append(words, word{syms, c})
	}

	counts := map[bpePair]int{}
	where := map[bpePair][]int{} // pair -> indices of words that have contained it
	addPairs := func(wi int, sign int) {
		w := words[wi]
		for i := 0; i+1 < len(w.syms); i++ {
			p := bpePair{w.syms[i], w.syms[i+1]}
			counts[p] += sign * w.count
			if sign > 0 {
				where[p] = append(where[p], wi)
			} else if counts[p] == 0 {
				delete(counts, p)
			}
		}
	}
	for wi := range words {
		addPairs(wi, 1)
	}

	merges := make([]bpePair, 0, n)
	seen := make([]int, len(words)) // last merge step that rewrote each word, +1
	for step := 1; step <= n && len(counts) > 0; step++ {
		var best bpePair
		bestCount := 0
		for p, c := range counts {
			if c > bestCount || c == bestCount && (p.a < best.a || p.a == best.a && p.b < best.b) {
				best, bestCount = p, c
			}
		}
		id := bpeBase + len(merges)
		merges = append(merges, best)
		for _, wi := range where[best] {
			if seen[wi] == step {
				continue // listed twice
			}
			seen[wi] = step
			addPairs(wi, -1)
			words[wi].syms = mergeAll(words[wi].syms, best, id)
			addPairs(wi, 1)
		}
		delete(where, best)
	}
	return merges
}

// preTokenize splits text into words the way GPT-2's pattern does:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// A word is an optional single leading space followed by a run of letters, a
// run of digits, or a run of other non-space characters; English contractions
// are split off on their own. A whitespace run that precedes a word leaves its
// last space for that word, and a newline followed by text is its own word.
// Concatenating the result gives back text exactly.
func preTokenize(text string) []string {
	var out []string
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		if r == '\'' {
			if n := contractionLen(text[i+size:]); n > 0 {
				out = append(out, text[i:i+size+n])
				i += size + n
				continue
			}
		}
		start := i
		if r == ' ' && i+size < len(text) {
			if next, _ := utf8.DecodeRuneInString(text[i+size:]); !unicode.IsSpace(next) {
				i += size
				r, size = next, 0
			}
		}
		if !unicode.IsSpace(r) {
			class := runeClass(r)
			for i < len(text) {
				c, sz := utf8.DecodeRuneInString(text[i:])
				if unicode.IsSpace(c) || runeClass(c) != class {
					break
				}
				i += sz
			}
			out = append(out, text[start:i])
			continue
		}
		// Whitespace run. If text follows it, the last rune is left for the
		// next word (a space joins it; anything else stands alone).
		end, last := i, i
		for end < len(text) {
			c, sz := utf8.DecodeRuneInString(text[end:])
			if !unicode.IsSpace(c) {
				break
			}
			last = end
			end += sz
		}
		if end < len(text) && last > i {
			end = last
		}
		out = append(out, text[i:end])
		i = end
	}
	return out
}

// contractionLen returns the byte length of a contraction suffix ("s", "t",
// "re", "ve", "m", "ll", "d") at the start of s, or 0.
func contractionLen(s string) int {
	for _, suf := range [...]string{"re", "ve", "ll", "s", "t", "m", "d"} {
		if strings.HasPrefix(s, suf) {
			return len(suf)
		}
	}
	return 0
}

const (
	classLetter = iota
	classNumber
	classOther
)

func runeClass(r rune) int {
	switch {
	case unicode.IsLetter(r):
		return classLetter
	case unicode.IsNumber(r):
		return classNumber
	default:
		return classOther
	}
}
