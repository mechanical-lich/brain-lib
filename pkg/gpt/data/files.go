package data

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ReadText returns the concatenated contents of path. If path is a
// directory, every *.txt file directly inside it is read in name order and
// joined with blank lines; otherwise the single file is read.
func ReadText(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		b, err := os.ReadFile(path)
		return string(b), err
	}
	matches, err := filepath.Glob(filepath.Join(path, "*.txt"))
	if err != nil {
		return "", err
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("data: no *.txt files in %s", path)
	}
	sort.Strings(matches)
	var sb strings.Builder
	for _, m := range matches {
		b, err := os.ReadFile(m)
		if err != nil {
			return "", err
		}
		sb.Write(b)
		sb.WriteString("\n\n")
	}
	return sb.String(), nil
}
