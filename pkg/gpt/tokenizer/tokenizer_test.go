package tokenizer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCharRoundTrip(t *testing.T) {
	tok := NewChar("hello, wörld\n")
	assert.Equal(t, 12, tok.VocabSize(), "11 distinct runes + unknown")
	text := "wöld, hello\n"
	assert.Equal(t, text, tok.Decode(tok.Encode(text)))
	assert.Equal(t, []int{UnknownToken}, tok.Encode("z"), "unseen rune maps to unknown")
	assert.Equal(t, "�", tok.Decode([]int{999}), "out-of-range id decodes safely")
}

func TestCharDeterministicIds(t *testing.T) {
	a := NewChar("cba")
	b := NewChar("abc")
	assert.Equal(t, a.Encode("abc"), b.Encode("abc"))
}

func TestSymbolRoundTrip(t *testing.T) {
	tok := NewSymbol([]string{"SENSOR_L_NEAR", "SENSOR_L_FAR", "ACTION_TURN_L", "ACTION_FWD", "ACTION_FWD"})
	assert.Equal(t, 5, tok.VocabSize(), "duplicates ignored, plus unknown")
	ids := tok.Encode("SENSOR_L_NEAR   ACTION_TURN_L\nSENSOR_L_FAR ACTION_FWD")
	assert.Equal(t, []int{1, 3, 2, 4}, ids)
	assert.Equal(t, "SENSOR_L_NEAR ACTION_TURN_L SENSOR_L_FAR ACTION_FWD", tok.Decode(ids))
	assert.Equal(t, []int{UnknownToken}, tok.Encode("NOPE"))
}

func TestMarshalUnmarshal(t *testing.T) {
	for _, orig := range []Tokenizer{NewChar("The horror!"), NewSymbol([]string{"a", "b"})} {
		raw, err := Marshal(orig)
		require.NoError(t, err)
		back, err := Unmarshal(raw)
		require.NoError(t, err)
		assert.Equal(t, orig.Kind(), back.Kind())
		assert.Equal(t, orig.VocabSize(), back.VocabSize())
		sample := "The a b"
		assert.Equal(t, orig.Encode(sample), back.Encode(sample))
	}
	_, err := Unmarshal([]byte(`{"kind":"wordpiece","state":{}}`))
	assert.Error(t, err)
}
