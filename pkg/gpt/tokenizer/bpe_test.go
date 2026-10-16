package tokenizer

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPreTokenize(t *testing.T) {
	text := "Hello, world! It's 42.\n\nNew  line don't\tend "
	words := preTokenize(text)
	assert.Equal(t, []string{
		"Hello", ",", " world", "!", " It", "'s", " 42", ".", "\n", "\n",
		"New", " ", " line", " don", "'t", "\t", "end", " ",
	}, words)
	assert.Equal(t, text, strings.Join(words, ""))
}

// The classic Sennrich example. After pre-tokenization the word dictionary is
//
//	"low" 1, " low" 4, " lower" 2, " newest" 6, " widest" 3
//
// so the byte pairs "es" and "st" both occur 9 times; the tie goes to the
// smaller ids, 'e' < 's', giving "es" first. "es"+"t" is then 9, and "lo" and
// "ow" tie at 7, so "lo" then "lo"+"w".
func TestBPEKnownMerges(t *testing.T) {
	corpus := "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
	tok := NewBPE(corpus, 256+4)
	require.Equal(t, 260, tok.VocabSize())
	vocab := tok.Vocab()
	assert.Equal(t, []string{"es", "est", "lo", "low"}, vocab[256:])
	assert.Equal(t, "a", vocab['a'])

	assert.Equal(t, []int{259, ' ', 'n', 'e', 'w', 257}, tok.Encode("low newest"))
	assert.Equal(t, []int{' ', 259, 'e', 'r'}, tok.Encode(" lower"))
}

func TestBPEStopsWhenNothingToMerge(t *testing.T) {
	tok := NewBPE("ab ab", 1000)
	// "ab", " ab" -> merge a+b, then " "+"ab": every word is one token.
	assert.Equal(t, []string{"ab", " ab"}, tok.Vocab()[256:])
	assert.Equal(t, 256, NewBPE("anything", 10).VocabSize(), "vocab below 256 means bytes only")
}

func TestBPERoundTrip(t *testing.T) {
	corpus := strings.Repeat("The ancient horror slept beneath the sea; the stars were right.\n", 20) +
		"Ph'nglui mglw'nafh Cthulhu R'lyeh wgah'nagl fhtagn!\n"
	tok := NewBPE(corpus, 400)
	assert.Greater(t, tok.VocabSize(), 256)

	text := "The horror—it was Ångström's dæmon, 東京 and 🐙!\n\n  Strange... \"quotes\" (and) [brackets]\r\n\x00end"
	ids := tok.Encode(text)
	assert.Equal(t, text, tok.Decode(ids))
	assert.Less(t, len(tok.Encode("the stars were right")), len("the stars were right"), "merges shorten learned text")
	for _, id := range ids {
		assert.True(t, id >= 0 && id < tok.VocabSize())
	}
	assert.Equal(t, "�", tok.Decode([]int{99999}), "out-of-range id decodes safely")
	assert.Equal(t, ids, tok.Encode(text), "cached encode matches first encode")
}

func TestBPEMarshalUnmarshal(t *testing.T) {
	corpus := strings.Repeat("That is not dead which can eternal lie, and with strange aeons even death may die.\n", 10)
	orig := NewBPE(corpus, 320)
	raw, err := Marshal(orig)
	require.NoError(t, err)
	back, err := Unmarshal(raw)
	require.NoError(t, err)
	assert.Equal(t, KindBPE, back.Kind())
	assert.Equal(t, orig.VocabSize(), back.VocabSize())
	assert.Equal(t, orig.Vocab(), back.(*BPE).Vocab())
	sample := "With strange aeons, even death may die. Ünïcödé too."
	assert.Equal(t, orig.Encode(sample), back.Encode(sample))
	assert.Equal(t, sample, back.Decode(back.Encode(sample)))

	_, err = Unmarshal([]byte(`{"kind":"bpe","state":{"merges":[[1,256]]}}`))
	assert.Error(t, err, "a merge may not reference its own or a later id")
}

const lovecraftPath = "/Users/johngodsey/repos/mechanical-lich-repos/critters-project/brain-lib/local_data/training_data/lovecraft.txt"

func TestBPELearnLovecraft(t *testing.T) {
	if testing.Short() {
		t.Skip("learns BPE on a 3 MB corpus")
	}
	raw, err := os.ReadFile(lovecraftPath)
	if os.IsNotExist(err) {
		t.Skip("corpus not present: " + lovecraftPath)
	}
	require.NoError(t, err)
	text := string(raw)

	start := time.Now()
	tok := NewBPE(text, 2000)
	learn := time.Since(start)
	start = time.Now()
	ids := tok.Encode(text)
	encode := time.Since(start)
	t.Logf("learn %v, encode %v, vocab %d, %d bytes -> %d tokens (%.2f bytes/token)",
		learn, encode, tok.VocabSize(), len(text), len(ids), float64(len(text))/float64(len(ids)))

	assert.Equal(t, 2000, tok.VocabSize())
	assert.Less(t, learn, 2*time.Minute)
	assert.Equal(t, text, tok.Decode(ids))
	assert.Greater(t, float64(len(text))/float64(len(ids)), 3.0, "sub-word tokens should average several bytes")
}
