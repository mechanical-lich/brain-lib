// Command gpt-fetch downloads public-domain texts from Project Gutenberg
// into a corpus directory for cmd/gpt-train.
//
//	go run ./cmd/gpt-fetch                      # default horror corpus
//	go run ./cmd/gpt-fetch -ids 345,84 -out ./x # specific book ids
//	go run ./cmd/gpt-fetch -list                # show the default list
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/data"
)

func main() {
	out := flag.String("out", "local_data/gutenberg", "Directory to write <id>.txt files into")
	ids := flag.String("ids", "", "Comma-separated Gutenberg ids (default: built-in horror list)")
	list := flag.Bool("list", false, "Print the default book list and exit")
	flag.Parse()

	books := data.HorrorBooks
	if *list {
		for _, b := range books {
			fmt.Printf("%6d  %s\n", b.ID, b.Title)
		}
		return
	}
	if *ids != "" {
		books = nil
		for _, s := range strings.Split(*ids, ",") {
			id, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil {
				fmt.Fprintf(os.Stderr, "bad id %q: %v\n", s, err)
				os.Exit(2)
			}
			books = append(books, data.Book{ID: id, Title: fmt.Sprintf("gutenberg #%d", id)})
		}
	}

	ctx := context.Background()
	failed := 0
	for _, b := range books {
		path, err := data.Fetch(ctx, nil, b, *out)
		if err != nil {
			slog.Error("fetch failed", "id", b.ID, "title", b.Title, "err", err)
			failed++
			continue
		}
		slog.Info("fetched", "id", b.ID, "title", b.Title, "path", path)
	}
	if failed > 0 {
		os.Exit(1)
	}
}
