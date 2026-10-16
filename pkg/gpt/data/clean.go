package data

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// CleanOptions selects the transformations Clean applies. A char tokenizer's
// vocabulary is the set of distinct runes in the corpus, so the same options
// must be used for the first training run and for any text added later.
type CleanOptions struct {
	// RemoveEditorial drops bracketed blocks whose opening bracket starts
	// with one of EditorialPrefixes (matched across lines to the closing
	// bracket) and footnote reference markers such as [1].
	RemoveEditorial   bool
	EditorialPrefixes []string
	// StripItalics removes _underscore_ italic markers, keeping the words.
	StripItalics bool
	// NormalizePunctuation turns curly quotes and the acute accent used as an
	// apostrophe into ASCII quotes, en/em dashes into "--" and the ellipsis
	// character into "...".
	NormalizePunctuation bool
	// DropMojibake repairs the common two-rune mis-decodings of UTF-8
	// (e.g. "Ã¤" for "ä") and then deletes the runes in MojibakeRunes.
	// Legitimate accented letters and ligatures are kept.
	DropMojibake  bool
	MojibakeRunes string
	// Unwrap joins hard-wrapped lines so each paragraph is one line. A blank
	// line is a paragraph break; a line starting with whitespace (verse,
	// epigraphs) stays on its own line with a fixed indent.
	Unwrap bool
	// CollapseWhitespace squeezes runs of spaces to one and runs of more than
	// two newlines to two.
	CollapseWhitespace bool
	// DropHeader removes everything before the first paragraph longer than
	// HeaderMinParagraph runes (title, author, dedication lines). It is a
	// heuristic, so it is off by default.
	DropHeader         bool
	HeaderMinParagraph int
}

// DefaultCleanOptions is tuned for Project Gutenberg prose: everything on
// except DropHeader.
func DefaultCleanOptions() CleanOptions {
	return CleanOptions{
		RemoveEditorial:      true,
		EditorialPrefixes:    []string{"[Illustration", "[Footnote", "[Transcriber", "[Source"},
		StripItalics:         true,
		NormalizePunctuation: true,
		DropMojibake:         true,
		MojibakeRunes:        "ÃĂÂ¤¦�",
		Unwrap:               true,
		CollapseWhitespace:   true,
		HeaderMinParagraph:   200,
	}
}

// verseIndent replaces the leading whitespace of an indented line when
// unwrapping, so verse stays distinguishable without spending many tokens on
// alignment spaces.
const verseIndent = "    "

var (
	footnoteRef = regexp.MustCompile(`\[\d+\]`)
	manyNL      = regexp.MustCompile(`\n{3,}`)
	manySpaces  = regexp.MustCompile(` {2,}`)
)

var punctReplacer = strings.NewReplacer(
	"‘", "'", "’", "'", "‚", "'", "‛", "'", "´", "'",
	"“", `"`, "”", `"`, "„", `"`, "‟", `"`,
	"–", "--", "—", "--", "―", "--",
	"…", "...",
)

// Clean normalizes text for training. Steps run in this order: CRLF
// normalization, editorial block removal, italic markers, punctuation,
// mojibake, unwrapping, whitespace collapsing, header dropping.
func Clean(text string, opts CleanOptions) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	if opts.RemoveEditorial {
		text = removeBracketed(text, opts.EditorialPrefixes)
		text = footnoteRef.ReplaceAllString(text, "")
	}
	if opts.StripItalics {
		text = strings.ReplaceAll(text, "_", "")
	}
	if opts.NormalizePunctuation {
		text = punctReplacer.Replace(text)
	}
	if opts.DropMojibake {
		text = fixMojibake(text, opts.MojibakeRunes)
	}
	if opts.CollapseWhitespace {
		text = strings.Map(func(r rune) rune {
			if r == '\t' || r == ' ' {
				return ' '
			}
			return r
		}, text)
	}
	if opts.Unwrap {
		text = unwrap(text)
	}
	if opts.CollapseWhitespace {
		text = collapseSpaces(text)
		text = manyNL.ReplaceAllString(text, "\n\n")
	}
	if opts.DropHeader {
		text = dropHeader(text, opts.HeaderMinParagraph)
	}
	return strings.TrimRight(strings.TrimLeft(text, "\n"), " \n") + "\n"
}

// removeBracketed deletes every "[prefix ... ]" block, tracking nested
// brackets so an inner [1] doesn't end the block early. An unterminated
// block is left alone.
func removeBracketed(text string, prefixes []string) string {
	var b strings.Builder
	for i := 0; i < len(text); {
		if text[i] == '[' && hasAnyPrefix(text[i:], prefixes) {
			if end := matchBracket(text, i); end >= 0 {
				i = end + 1
				continue
			}
		}
		b.WriteByte(text[i])
		i++
	}
	return b.String()
}

func hasAnyPrefix(s string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

// matchBracket returns the index of the ']' closing the '[' at start, or -1.
func matchBracket(text string, start int) int {
	depth := 0
	for j := start; j < len(text); j++ {
		switch text[j] {
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return j
			}
		}
	}
	return -1
}

// fixMojibake repairs UTF-8 text that was decoded as Latin-1/Windows-1250
// and re-encoded: a lead rune (Ã or Ă, both byte 0xC3) followed by a rune in
// U+0080..U+00BF is decoded back to the original two-byte character. Any
// remaining rune in drop is deleted.
func fixMojibake(text, drop string) string {
	rs := []rune(text)
	var b strings.Builder
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		if (r == 'Ã' || r == 'Ă') && i+1 < len(rs) && rs[i+1] >= 0x80 && rs[i+1] <= 0xBF {
			if fixed, _ := utf8.DecodeRune([]byte{0xC3, byte(rs[i+1])}); fixed != utf8.RuneError {
				b.WriteRune(fixed)
				i++
				continue
			}
		}
		if strings.ContainsRune(drop, r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// unwrap joins the non-indented lines of each paragraph with single spaces.
// Indented lines start their own output line with a fixed indent, and the
// line after one starts a new line too.
func unwrap(text string) string {
	var b strings.Builder
	open := false // a joinable prose line is being built
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "":
			if open || b.Len() > 0 {
				b.WriteString("\n\n")
			}
			open = false
		case unicode.IsSpace(rune(line[0])):
			if b.Len() > 0 && !strings.HasSuffix(b.String(), "\n") {
				b.WriteByte('\n')
			}
			b.WriteString(verseIndent)
			b.WriteString(trimmed)
			b.WriteByte('\n')
			open = false
		case open:
			b.WriteByte(' ')
			b.WriteString(trimmed)
		default:
			if b.Len() > 0 && !strings.HasSuffix(b.String(), "\n") {
				b.WriteByte('\n')
			}
			b.WriteString(trimmed)
			open = true
		}
	}
	return b.String()
}

// collapseSpaces squeezes runs of spaces to one and trims trailing spaces on
// each line, preserving a leading verse indent.
func collapseSpaces(text string) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		body := strings.TrimLeft(line, " ")
		indent := ""
		if len(body) < len(line) && body != "" {
			indent = verseIndent
		}
		lines[i] = indent + strings.TrimRight(manySpaces.ReplaceAllString(body, " "), " ")
	}
	return strings.Join(lines, "\n")
}

// dropHeader removes the paragraphs before the first one longer than minLen
// runes. If no paragraph is that long the text is returned unchanged.
func dropHeader(text string, minLen int) string {
	paras := strings.Split(text, "\n\n")
	for i, p := range paras {
		if utf8.RuneCountInString(strings.TrimSpace(p)) > minLen {
			return strings.Join(paras[i:], "\n\n")
		}
	}
	return text
}
