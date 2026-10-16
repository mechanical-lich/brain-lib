package data

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Book identifies a Project Gutenberg text.
type Book struct {
	ID    int
	Title string
}

// HorrorBooks is a starter corpus of public-domain horror. Lovecraft's later
// works entered the US public domain recently, so the available set grows
// each January; add IDs here as they appear on gutenberg.org.
var HorrorBooks = []Book{
	{ID: 68283, Title: "The Call of Cthulhu (Lovecraft)"},
	{ID: 50133, Title: "The Dunwich Horror (Lovecraft)"},
	{ID: 31469, Title: "The Shunned House (Lovecraft)"},
	{ID: 68236, Title: "The Colour Out of Space (Lovecraft)"},
	{ID: 70652, Title: "At the Mountains of Madness (Lovecraft)"},
	{ID: 73181, Title: "The Shadow over Innsmouth (Lovecraft)"},
	{ID: 73547, Title: "The Case of Charles Dexter Ward (Lovecraft)"},
	{ID: 345, Title: "Dracula (Stoker)"},
	{ID: 84, Title: "Frankenstein (Shelley)"},
	{ID: 10007, Title: "Carmilla (Le Fanu)"},
	{ID: 8492, Title: "The King in Yellow (Chambers)"},
	{ID: 209, Title: "The Turn of the Screw (James)"},
	{ID: 2148, Title: "The Works of Edgar Allan Poe, Vol. 2"},
}

// GutenbergURL returns the plain-text download URL for a book id.
func GutenbergURL(id int) string {
	return fmt.Sprintf("https://www.gutenberg.org/cache/epub/%d/pg%d.txt", id, id)
}

// Fetch downloads book into dir as <id>.txt with the Gutenberg header and
// footer stripped and the body run through Clean with DefaultCleanOptions,
// and returns the path. An existing file is left alone
// and returned as-is.
func Fetch(ctx context.Context, client *http.Client, book Book, dir string) (string, error) {
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, fmt.Sprintf("%d.txt", book.ID))
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, GutenbergURL(book.ID), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "brain-lib-gpt-fetch/1.0 (corpus download)")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("data: fetch %d: %s", book.ID, resp.Status)
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	text := Clean(StripBoilerplate(string(raw)), DefaultCleanOptions())
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("data: fetch %d: empty text after stripping", book.ID)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// StripBoilerplate removes the Project Gutenberg license header and footer,
// leaving the body between the "*** START OF" and "*** END OF" markers. If
// the markers are missing the text is returned unchanged (minus CRLFs).
func StripBoilerplate(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	if i := strings.Index(text, "*** START OF"); i >= 0 {
		if nl := strings.IndexByte(text[i:], '\n'); nl >= 0 {
			text = text[i+nl+1:]
		}
	}
	if i := strings.Index(text, "*** END OF"); i >= 0 {
		text = text[:i]
	}
	return strings.TrimSpace(text) + "\n"
}
