// Command gpt-generate samples text from a checkpoint written by gpt-train.
//
//	go run ./cmd/gpt-generate -prompt "The old house" -n 400 -temp 0.8
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tokenizer"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

func main() {
	ckptPath := flag.String("ckpt", "local_data/gpt/model.ckpt", "Checkpoint path")
	prompt := flag.String("prompt", "\n", "Prompt text")
	n := flag.Int("n", 300, "Tokens to generate")
	temp := flag.Float64("temp", 0.8, "Sampling temperature (0 = greedy)")
	topK := flag.Int("topk", 40, "Top-k sampling cutoff (0 disables)")
	seed := flag.Int64("seed", 0, "RNG seed (0 = time-based)")
	flag.Parse()

	ck, err := train.Load(*ckptPath)
	if err != nil {
		fail("load checkpoint", err)
	}
	g, err := ck.Model()
	if err != nil {
		fail("restore model", err)
	}
	if len(ck.Tokenizer) == 0 {
		fail("tokenizer", fmt.Errorf("checkpoint has no tokenizer; cannot decode text"))
	}
	tok, err := tokenizer.Unmarshal(ck.Tokenizer)
	if err != nil {
		fail("restore tokenizer", err)
	}

	if *seed == 0 {
		*seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(*seed))
	ids := tok.Encode(*prompt)
	if len(ids) == 0 {
		ids = []int{1}
	}
	out := g.Generate(ids, *n, model.GenerateOptions{Temperature: *temp, TopK: *topK}, rng)
	fmt.Println(tok.Decode(out))
}

func fail(msg string, err error) {
	fmt.Fprintf(os.Stderr, "%s: %v\n", msg, err)
	os.Exit(1)
}
