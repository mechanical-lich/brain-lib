package train

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// MatData is a serializable matrix.
type MatData struct {
	Rows, Cols int
	Data       []float64
}

// OptimizerState is the serializable part of an AdamW.
type OptimizerState struct {
	Config AdamWConfig
	Step   int
	M, V   []MatData // parallel to GPT.Params()
}

// Checkpoint is a complete snapshot of a training run.
//
// Tokenizer is opaque to this package: callers typically store the output of
// tokenizer.Marshal so the generate command can decode samples, but a
// robot-control vocabulary might store nothing at all.
type Checkpoint struct {
	Config    model.Config
	Step      int
	Params    map[string]MatData
	Optimizer *OptimizerState
	Tokenizer []byte
}

// Snapshot captures g (and optionally opt) into a Checkpoint. Parameter
// values are copied, so the snapshot is unaffected by further training.
func Snapshot(g *model.GPT, opt *AdamW, step int, tokenizer []byte) *Checkpoint {
	ck := &Checkpoint{
		Config:    g.Config,
		Step:      step,
		Params:    make(map[string]MatData),
		Tokenizer: tokenizer,
	}
	for _, p := range g.Params() {
		ck.Params[p.Name] = toMatData(p.Value)
	}
	if opt != nil {
		st := &OptimizerState{Config: opt.Config, Step: opt.step}
		for i := range opt.params {
			st.M = append(st.M, toMatData(opt.m[i]))
			st.V = append(st.V, toMatData(opt.v[i]))
		}
		ck.Optimizer = st
	}
	return ck
}

// Save writes a checkpoint atomically (write to temp file, then rename).
func Save(path string, ck *Checkpoint) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".ckpt-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if err := gob.NewEncoder(tmp).Encode(ck); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("train: encode checkpoint: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, path)
}

// Load reads a checkpoint written by Save.
func Load(path string) (*Checkpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var ck Checkpoint
	if err := gob.NewDecoder(f).Decode(&ck); err != nil {
		return nil, fmt.Errorf("train: decode checkpoint %s: %w", path, err)
	}
	if err := ck.Config.Validate(); err != nil {
		return nil, err
	}
	return &ck, nil
}

// Model builds a GPT from the checkpoint's config and loads its weights.
func (ck *Checkpoint) Model() (*model.GPT, error) {
	g := model.New(ck.Config, newRand(0))
	for _, p := range g.Params() {
		md, ok := ck.Params[p.Name]
		if !ok {
			return nil, fmt.Errorf("train: checkpoint missing param %q", p.Name)
		}
		if md.Rows != p.Value.Rows || md.Cols != p.Value.Cols {
			return nil, fmt.Errorf("train: param %q is %dx%d in checkpoint, want %s",
				p.Name, md.Rows, md.Cols, p.Value.Shape())
		}
		copy(p.Value.Data, md.Data)
	}
	return g, nil
}

// OptimizerFor rebuilds an AdamW over g's params from the checkpoint. If the
// checkpoint has no optimizer state, a fresh optimizer with cfg is returned;
// otherwise cfg overrides the saved hyperparameters but moments are kept.
func (ck *Checkpoint) OptimizerFor(g *model.GPT, cfg AdamWConfig) (*AdamW, error) {
	opt := NewAdamW(g.Params(), cfg)
	st := ck.Optimizer
	if st == nil {
		return opt, nil
	}
	if len(st.M) != len(opt.params) || len(st.V) != len(opt.params) {
		return nil, fmt.Errorf("train: optimizer state has %d params, model has %d", len(st.M), len(opt.params))
	}
	for i := range opt.params {
		if err := fromMatData(st.M[i], opt.m[i]); err != nil {
			return nil, err
		}
		if err := fromMatData(st.V[i], opt.v[i]); err != nil {
			return nil, err
		}
	}
	opt.step = st.Step
	return opt, nil
}

func toMatData(m *tensor.Mat) MatData {
	return MatData{Rows: m.Rows, Cols: m.Cols, Data: append([]float64(nil), m.Data...)}
}

func fromMatData(md MatData, dst *tensor.Mat) error {
	if md.Rows != dst.Rows || md.Cols != dst.Cols || len(md.Data) != len(dst.Data) {
		return fmt.Errorf("train: matrix %dx%d does not fit %s", md.Rows, md.Cols, dst.Shape())
	}
	copy(dst.Data, md.Data)
	return nil
}
