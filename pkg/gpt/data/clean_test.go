package data

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCleanDefaults(t *testing.T) {
	cases := []struct {
		name, in, want string
	}{
		{"crlf", "one\r\ntwo\r\n\r\nthree", "one two\n\nthree\n"},
		{
			"multi-line illustration",
			"Before it.\n\n[Illustration: \"The ring of worshipers moved\nbetween the ring of bodies.\"][1]\n\nAfter it.",
			"Before it.\n\nAfter it.\n",
		},
		{"bare illustration", "A\n\n[Illustration]\n\nB", "A\n\nB\n"},
		{
			"footnote block",
			"Text here.[1]\n\n[Footnote 1: Found among the papers of the late\nFrancis Wayland Thurston, of Boston.]\n\nMore.",
			"Text here.\n\nMore.\n",
		},
		{
			"transcriber note",
			"[Transcriber's Note: This etext was produced from\nWeird Tales, February 1928.]\n\nStory.",
			"Story.\n",
		},
		{"other brackets kept", "the thing [it ran] fled", "the thing [it ran] fled\n"},
		{"italics", "it was _not_ a _dream_ at all", "it was not a dream at all\n"},
		{"curly quotes", "“Don’t,” she said. ‘No.’", "\"Don't,\" she said. 'No.'\n"},
		{"acute apostrophe", "don´t", "don't\n"},
		{"dashes", "dark—deep–cold", "dark--deep--cold\n"},
		{"mojibake repaired", "the AthenĂ¦um and WidmÃ¤nstÃ¤tten", "the Athenæum and Widmänstätten\n"},
		{"mojibake dropped", "odd¤ markÃ", "odd mark\n"},
		{"accents survive", "Phænician naïve café Noël señor façade 5°", "Phænician naïve café Noël señor façade 5°\n"},
		{
			"unwrap paragraph",
			"The most merciful thing in the world, I think, is the\ninability of the human mind.\n\nSecond paragraph\nwraps too.",
			"The most merciful thing in the world, I think, is the inability of the human mind.\n\nSecond paragraph wraps too.\n",
		},
		{
			"verse stays line-broken",
			"He recited:\n\n    That is not dead which can eternal lie,\n    And with strange aeons even death may die.\n\nThen silence.",
			"He recited:\n\n    That is not dead which can eternal lie,\n    And with strange aeons even death may die.\n\nThen silence.\n",
		},
		{
			"prose after verse in same block",
			"    a verse line\nprose that\nwraps",
			"    a verse line\nprose that wraps\n",
		},
		{"collapse spaces", "too   many \t spaces", "too many spaces\n"},
		{"collapse newlines", "a\n\n\n\n\nb", "a\n\nb\n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			assert.Equal(t, c.want, Clean(c.in, DefaultCleanOptions()))
		})
	}
}

func TestCleanOptionsToggle(t *testing.T) {
	opts := DefaultCleanOptions()
	opts.NormalizePunctuation = false
	assert.Equal(t, "“dark—deep”\n", Clean("“dark—deep”", opts))

	opts = DefaultCleanOptions()
	opts.Unwrap = false
	assert.Equal(t, "one\ntwo\n", Clean("one\ntwo", opts))
}

func TestCleanDropHeader(t *testing.T) {
	long := strings.Repeat("word ", 50)
	in := "The CALL of CTHULHU\n\nBy H.P. LOVECRAFT\n\n" + long + "\n\nShort.\n"

	assert.True(t, strings.HasPrefix(Clean(in, DefaultCleanOptions()), "The CALL of CTHULHU\n\n"), "off by default")

	opts := DefaultCleanOptions()
	opts.DropHeader = true
	got := Clean(in, opts)
	assert.True(t, strings.HasPrefix(got, "word word"), got)
	assert.True(t, strings.HasSuffix(got, "\n\nShort.\n"), got)

	assert.Equal(t, "Tiny.\n\nText.\n", Clean("Tiny.\n\nText.", opts), "no long paragraph keeps everything")
}

// TestCleanGutenbergSample runs the defaults over a Gutenberg-shaped excerpt
// and checks the properties the training corpus relies on.
func TestCleanGutenbergSample(t *testing.T) {
	raw := `The CALL of CTHULHU

By H.P. LOVECRAFT

[Transcriber's Note: This etext was produced from
Weird Tales, February 1928.]


    "Of such great powers or beings there may be conceivably a
    survival ... a survival of a hugely remote period."

                                                --_Algernon Blackwood._


[Illustration: "The ring of worshipers moved in endless bacchanale
between the ring of bodies and the ring of fire."][1]

[Footnote 1: Found among the papers of the late Francis Wayland
Thurston, of Boston.]


                       _1. The Horror in Clay._

The most merciful thing in the world, I think, is the inability of the
human mind to correlate all its contents. We live on a placid island
of ignorance in the midst of black seas of infinity, and it was _not_
meant that we should voyage far.

Theosophists have guessed at the awesome grandeur of the cosmic cycle
wherein our world and human race form transient incidents.
`
	dir := t.TempDir()
	path := filepath.Join(dir, "book.txt")
	require.NoError(t, os.WriteFile(path, []byte(raw), 0o644))
	text, err := ReadText(path)
	require.NoError(t, err)

	got := Clean(text, DefaultCleanOptions())
	assert.NotContains(t, got, "_")
	assert.NotContains(t, got, "[Illustration")
	assert.NotContains(t, got, "[Footnote")
	assert.NotContains(t, got, "[1]")
	assert.Contains(t, got, "The most merciful thing in the world, I think, is the inability of the human mind to correlate all its contents. We live on a placid island of ignorance in the midst of black seas of infinity, and it was not meant that we should voyage far.\n")
	assert.Contains(t, got, "\n    \"Of such great powers or beings there may be conceivably a\n    survival")
	assert.Contains(t, got, "\n    1. The Horror in Clay.\n")
}
