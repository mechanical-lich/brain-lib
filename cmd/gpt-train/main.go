// Command gpt-train trains a small GPT on a text corpus.
//
//	go run ./cmd/gpt-fetch
//	go run ./cmd/gpt-train -data local_data/gutenberg -steps 2000
//	go run ./cmd/gpt-train -resume -steps 4000   # continue from the checkpoint
//	go run ./cmd/gpt-train -data my_corpus -clean   # normalize user-supplied text first
//
// The -tokenizer flag picks how text becomes tokens: "char" (one token per
// rune), "bpe" (GPT-2 style byte-pair encoding learned from the corpus, with
// -bpe-vocab ids), or "symbol" (one token per whitespace-separated
// word/symbol, which is the mode to use for a sensor/action vocabulary). The
// tokenizer is built on a fresh run only; -resume restores the checkpoint's.
//
// -backend gpu runs every training step on the machine's GPU through the
// pure-Go WebGPU backend (pkg/gpt/gpu, pkg/gpt/gpumodel). Checkpoints are
// the same files either way, so a run can switch backends at any -resume.
package main

import (
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/data"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/gpu"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/gpumodel"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tokenizer"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

func main() {
	dataPath := flag.String("data", "local_data/gutenberg", "Corpus: a .txt file or a directory of .txt files")
	out := flag.String("out", "local_data/gpt/model.ckpt", "Checkpoint path")
	resume := flag.Bool("resume", false, "Resume from -out if it exists (model shape and -dropout flags are ignored)")
	tokKind := flag.String("tokenizer", tokenizer.KindChar, "Tokenizer kind for a new model: char, bpe or symbol (ignored on resume)")
	bpeVocab := flag.Int("bpe-vocab", 2000, "Vocabulary size to learn with -tokenizer bpe: 256 byte ids plus merges (ignored on resume)")
	clean := flag.Bool("clean", false, "Run data.Clean with default options on the corpus before tokenizing (use the same setting when resuming)")

	layers := flag.Int("layers", 4, "Transformer blocks")
	heads := flag.Int("heads", 4, "Attention heads")
	dim := flag.Int("dim", 128, "Embedding dimension")
	ctxLen := flag.Int("ctx", 128, "Context length")
	dropout := flag.Float64("dropout", 0.1, "Dropout probability in [0, 1) for a new model (ignored on resume)")

	steps := flag.Int("steps", 1000, "Total optimizer steps")
	batch := flag.Int("batch", 16, "Batch size")
	lr := flag.Float64("lr", 1e-3, "Peak learning rate")
	minLR := flag.Float64("min-lr", 1e-4, "Final learning rate of the cosine schedule")
	warmup := flag.Int("warmup", 100, "Linear warmup steps")
	wd := flag.Float64("wd", 0.1, "Weight decay")
	clip := flag.Float64("clip", 1.0, "Gradient clip norm (0 disables)")
	valFrac := flag.Float64("val-frac", 0.1, "Fraction of corpus held out for validation")
	evalEvery := flag.Int("eval-every", 100, "Validate every N steps (0 disables)")
	evalBatches := flag.Int("eval-batches", 8, "Batches per validation")
	ckptEvery := flag.Int("ckpt-every", 100, "Checkpoint every N steps (0 disables)")
	sampleLen := flag.Int("sample", 200, "Tokens to sample after each validation (0 disables)")
	seed := flag.Int64("seed", 1, "RNG seed")
	backend := flag.String("backend", "cpu", "Where training steps run: cpu (float64) or gpu (float32, via pure-Go WebGPU)")
	flag.Parse()
	if *backend != "cpu" && *backend != "gpu" {
		fmt.Fprintf(os.Stderr, "unknown -backend %q (want cpu or gpu)\n", *backend)
		os.Exit(2)
	}

	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	rng := rand.New(rand.NewSource(*seed))

	text, err := data.ReadText(*dataPath)
	if err != nil {
		fatal(log, "read corpus", err)
	}
	if *clean {
		before := distinctRunes(text)
		raw := len(text)
		text = data.Clean(text, data.DefaultCleanOptions())
		log.Info("cleaned corpus", "bytes_before", raw, "bytes_after", len(text), "runes_before", before, "runes_after", distinctRunes(text))
	}

	var (
		g       *model.GPT
		opt     *train.AdamW
		tok     tokenizer.Tokenizer
		started int
	)
	optCfg := train.DefaultAdamW()
	optCfg.LR, optCfg.WeightDecay = *lr, *wd

	ck, err := loadIfResuming(*resume, *out)
	if err != nil {
		fatal(log, "load checkpoint", err)
	}
	if ck != nil {
		if g, err = ck.Model(); err != nil {
			fatal(log, "restore model", err)
		}
		if opt, err = ck.OptimizerFor(g, optCfg); err != nil {
			fatal(log, "restore optimizer", err)
		}
		if tok, err = tokenizer.Unmarshal(ck.Tokenizer); err != nil {
			fatal(log, "restore tokenizer", err)
		}
		// Draw dropout masks from the run's seeded rng rather than the
		// fixed one the checkpoint loader builds the model with.
		g.SetRNG(rng)
		started = ck.Step
		log.Info("resumed", "path", *out, "step", started, "tokenizer", tok.Kind(), "config", fmt.Sprintf("%+v", g.Config))
	} else {
		tok = newTokenizer(log, *tokKind, text, *bpeVocab)
		cfg := model.Config{VocabSize: tok.VocabSize(), ContextLen: *ctxLen, NumLayers: *layers, NumHeads: *heads, EmbedDim: *dim, Dropout: *dropout}
		if err := cfg.Validate(); err != nil {
			fatal(log, "config", err)
		}
		g = model.New(cfg, rng)
		opt = train.NewAdamW(g.Params(), optCfg)
		log.Info("new model", "config", fmt.Sprintf("%+v", cfg), "params", g.NumParams())
	}
	tokBytes, err := tokenizer.Marshal(tok)
	if err != nil {
		fatal(log, "marshal tokenizer", err)
	}

	tokens := tok.Encode(text)
	trainSet, valSet := data.Split(tokens, *valFrac)
	unknown := 0
	if tok.Kind() != tokenizer.KindBPE { // BPE has no unknown id; 0 is the NUL byte
		for _, id := range tokens {
			if id == tokenizer.UnknownToken {
				unknown++
			}
		}
	}
	log.Info("corpus", "chars", len(text), "tokens", len(tokens), "bytes_per_token", round(float64(len(text))/float64(max(len(tokens), 1))), "vocab", tok.VocabSize(), "unknown", unknown)
	if unknown > 0 {
		log.Warn("corpus has tokens outside the vocabulary; was -clean set the same way as the first run?", "unknown", unknown)
	}

	opts := train.Options{
		Steps: *steps, StartStep: started, BatchSize: *batch,
		GradClip: *clip, LR: train.Cosine(*warmup, *steps, *lr, *minLR),
		EvalEvery: *evalEvery, EvalBatches: *evalBatches,
		CheckpointEvery: *ckptEvery, CheckpointPath: *out, Tokenizer: tokBytes,
	}
	stepper := train.NewCPUStepper(g, opt, *clip)
	// generate produces the periodic sample: on the CPU model (refreshed
	// by Sync before each eval), or on the GPU when that is the backend,
	// where a 200-token sample takes well under a second instead of the
	// tens of seconds a CPU forward per token costs at this model size.
	generate := func(prompt []int, n int, opts model.GenerateOptions) ([]int, error) {
		return g.Generate(prompt, n, opts, rng), nil
	}
	if *backend == "gpu" {
		dev, err := gpu.Open()
		if err != nil {
			if errors.Is(err, gpu.ErrNoAdapter) {
				fatal(log, "-backend gpu: no hardware GPU adapter (Linux needs the Vulkan loader and a driver such as Mesa RADV); use -backend cpu", err)
			}
			fatal(log, "-backend gpu: open device", err)
		}
		defer dev.Close()
		gm, err := gpumodel.New(dev, g.Config)
		if err != nil {
			fatal(log, "-backend gpu: allocate model", err)
		}
		defer gm.Close()
		gm.SetSeed(uint32(*seed))
		gs, err := gpumodel.NewStepper(gm, g, opt, *clip)
		if err != nil {
			fatal(log, "-backend gpu: load model", err)
		}
		stepper = gs
		generate = func(prompt []int, n int, opts model.GenerateOptions) ([]int, error) {
			return gs.Generate(prompt, n, opts, rng)
		}
		log.Info("backend", "backend", "gpu", "adapter", dev.Name(), "params", g.NumParams())
	} else {
		log.Info("backend", "backend", "cpu", "params", g.NumParams())
	}

	start := time.Now()
	err = train.RunStepper(stepper, g, opt, trainSet, valSet, opts, rng, func(m train.Metrics) {
		if math.IsNaN(m.ValLoss) {
			if m.Step%10 == 0 {
				perStep := time.Since(start) / time.Duration(m.Step-started)
				log.Info("step", "step", m.Step, "loss", round(m.Loss), "lr", m.LR, "gnorm", round(m.GradNorm), "ms_per_step", perStep.Milliseconds(), "elapsed", time.Since(start).Round(time.Second))
			}
			return
		}
		log.Info("eval", "step", m.Step, "loss", round(m.Loss), "val_loss", round(m.ValLoss), "lr", m.LR)
		if *sampleLen > 0 {
			prompt := tok.Encode("\n")
			if len(prompt) == 0 || prompt[0] == tokenizer.UnknownToken {
				prompt = []int{1}
			}
			t0 := time.Now()
			ids, err := generate(prompt, *sampleLen, model.GenerateOptions{Temperature: 0.8, TopK: 40})
			if err != nil {
				log.Error("sample", "err", err)
				return
			}
			fmt.Fprintf(os.Stderr, "---- sample @ step %d (%d tokens in %v) ----\n%s\n----\n", m.Step, *sampleLen, time.Since(t0).Round(time.Millisecond), strings.TrimSpace(tok.Decode(ids)))
		}
	})
	if err != nil {
		fatal(log, "train", err)
	}
	log.Info("done", "steps", *steps, "checkpoint", *out, "elapsed", time.Since(start).Round(time.Second))
}

func loadIfResuming(resume bool, path string) (*train.Checkpoint, error) {
	if !resume {
		return nil, nil
	}
	ck, err := train.Load(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	return ck, err
}

func newTokenizer(log *slog.Logger, kind, text string, bpeVocab int) tokenizer.Tokenizer {
	switch kind {
	case tokenizer.KindChar:
		return tokenizer.NewChar(text)
	case tokenizer.KindBPE:
		start := time.Now()
		tok := tokenizer.NewBPE(text, bpeVocab)
		log.Info("learned bpe", "elapsed", time.Since(start).Round(time.Millisecond), "vocab", tok.VocabSize(), "requested", bpeVocab)
		return tok
	case tokenizer.KindSymbol:
		return tokenizer.NewSymbol(strings.Fields(text))
	default:
		fmt.Fprintf(os.Stderr, "unknown tokenizer %q (want char, bpe or symbol)\n", kind)
		os.Exit(2)
		return nil
	}
}

func distinctRunes(s string) int {
	seen := map[rune]struct{}{}
	for _, r := range s {
		seen[r] = struct{}{}
	}
	return len(seen)
}

func round(v float64) float64 { return math.Round(v*1e4) / 1e4 }

func fatal(log *slog.Logger, msg string, err error) {
	log.Error(msg, "err", err)
	os.Exit(1)
}
