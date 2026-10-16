// Command gpt-generate samples text from a checkpoint written by gpt-train.
//
//	go run ./cmd/gpt-generate -prompt "The old house" -n 400 -temp 0.8
//	go run ./cmd/gpt-generate -backend gpu -prompt "The old house" -n 400
//
// -backend gpu runs the forward passes on the machine's GPU through the
// pure-Go WebGPU backend; the output is sampled identically to the CPU.
package main

import (
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/gpu"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/gpumodel"
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
	backend := flag.String("backend", "cpu", "Where to run the model: cpu or gpu")
	flag.Parse()
	if *backend != "cpu" && *backend != "gpu" {
		fail("flags", fmt.Errorf("unknown -backend %q (want cpu or gpu)", *backend))
	}

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
	opts := model.GenerateOptions{Temperature: *temp, TopK: *topK}
	var out []int
	if *backend == "gpu" {
		dev, err := gpu.Open()
		if err != nil {
			if errors.Is(err, gpu.ErrNoAdapter) {
				fail("-backend gpu", fmt.Errorf("no hardware GPU adapter (Linux needs the Vulkan loader and a driver such as Mesa RADV); use -backend cpu"))
			}
			fail("-backend gpu: open device", err)
		}
		defer dev.Close()
		gm, err := gpumodel.New(dev, g.Config)
		if err != nil {
			fail("-backend gpu: allocate model", err)
		}
		defer gm.Close()
		if err := gm.LoadFrom(g); err != nil {
			fail("-backend gpu: load weights", err)
		}
		if out, err = gm.Generate(ids, *n, opts, rng); err != nil {
			fail("-backend gpu: generate", err)
		}
	} else {
		out = g.Generate(ids, *n, opts, rng)
	}
	fmt.Println(tok.Decode(out))
}

func fail(msg string, err error) {
	fmt.Fprintf(os.Stderr, "%s: %v\n", msg, err)
	os.Exit(1)
}
