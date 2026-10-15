package data

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBatchWindows(t *testing.T) {
	tokens := make([]int, 50)
	for i := range tokens {
		tokens[i] = i
	}
	d := NewDataset(tokens)
	rng := rand.New(rand.NewSource(1))
	x, y := d.Batch(rng, 4, 8)
	require.Len(t, x, 4)
	for i := range x {
		require.Len(t, x[i], 8)
		for j := range x[i] {
			assert.Equal(t, x[i][j]+1, y[i][j], "target is the next token")
		}
	}
	assert.Panics(t, func() { NewDataset(tokens[:5]).Batch(rng, 1, 8) })
}

func TestSplit(t *testing.T) {
	tokens := make([]int, 100)
	tr, val := Split(tokens, 0.1)
	assert.Equal(t, 90, tr.Len())
	assert.Equal(t, 10, val.Len())
	tr, val = Split(tokens, 0)
	assert.Equal(t, 100, tr.Len())
	assert.Nil(t, val)
	assert.Panics(t, func() { Split(tokens, 1) })
}

func TestStripBoilerplate(t *testing.T) {
	raw := "junk\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\nBody line 1\r\nBody line 2\r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nlicense"
	assert.Equal(t, "Body line 1\nBody line 2\n", StripBoilerplate(raw))
	assert.Equal(t, "plain\n", StripBoilerplate("  plain  "), "no markers: unchanged apart from trim")
}

func TestReadText(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "b.txt"), []byte("second"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "a.txt"), []byte("first"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "ignored.md"), []byte("nope"), 0o644))
	got, err := ReadText(dir)
	require.NoError(t, err)
	assert.Equal(t, "first\n\nsecond\n\n", got)

	single, err := ReadText(filepath.Join(dir, "a.txt"))
	require.NoError(t, err)
	assert.Equal(t, "first", single)

	_, err = ReadText(t.TempDir())
	assert.Error(t, err, "empty dir")
}
