package train

import (
	"math"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// AdamWConfig holds optimizer hyperparameters.
type AdamWConfig struct {
	LR          float64 `json:"lr"`
	Beta1       float64 `json:"beta1"`
	Beta2       float64 `json:"beta2"`
	Eps         float64 `json:"eps"`
	WeightDecay float64 `json:"weightDecay"` // applied only to params with Decay == true
}

// DefaultAdamW returns sensible defaults for small models.
func DefaultAdamW() AdamWConfig {
	return AdamWConfig{LR: 3e-4, Beta1: 0.9, Beta2: 0.95, Eps: 1e-8, WeightDecay: 0.1}
}

// AdamW is Adam with decoupled weight decay (Loshchilov & Hutter).
type AdamW struct {
	Config AdamWConfig
	params []*model.Param
	m, v   []*tensor.Mat // first/second moment estimates, parallel to params
	step   int
}

// NewAdamW creates an optimizer over params. The order of params must be
// stable across runs (GPT.Params guarantees this) for checkpoint restore.
func NewAdamW(params []*model.Param, cfg AdamWConfig) *AdamW {
	o := &AdamW{Config: cfg, params: params}
	o.m = make([]*tensor.Mat, len(params))
	o.v = make([]*tensor.Mat, len(params))
	for i, p := range params {
		o.m[i] = tensor.New(p.Value.Rows, p.Value.Cols)
		o.v[i] = tensor.New(p.Value.Rows, p.Value.Cols)
	}
	return o
}

// Step applies one update using the gradients currently stored in each
// Param. It does not zero the gradients.
func (o *AdamW) Step() {
	o.step++
	c := o.Config
	bc1 := 1 - math.Pow(c.Beta1, float64(o.step))
	bc2 := 1 - math.Pow(c.Beta2, float64(o.step))
	for i, p := range o.params {
		w, g := p.Value.Data, p.Grad.Data
		m, v := o.m[i].Data, o.v[i].Data
		decay := 0.0
		if p.Decay {
			decay = c.LR * c.WeightDecay
		}
		for j := range w {
			m[j] = c.Beta1*m[j] + (1-c.Beta1)*g[j]
			v[j] = c.Beta2*v[j] + (1-c.Beta2)*g[j]*g[j]
			mhat := m[j] / bc1
			vhat := v[j] / bc2
			w[j] -= c.LR * mhat / (math.Sqrt(vhat) + c.Eps)
			if decay != 0 {
				w[j] -= decay * w[j]
			}
		}
	}
}

// StepCount returns the number of updates applied so far.
func (o *AdamW) StepCount() int { return o.step }

// ClipGradNorm rescales all gradients so their global L2 norm is at most
// maxNorm. It returns the norm before clipping.
func ClipGradNorm(params []*model.Param, maxNorm float64) float64 {
	var sq float64
	for _, p := range params {
		for _, g := range p.Grad.Data {
			sq += g * g
		}
	}
	norm := math.Sqrt(sq)
	if maxNorm > 0 && norm > maxNorm {
		s := maxNorm / (norm + 1e-6)
		for _, p := range params {
			p.Grad.Scale(s)
		}
	}
	return norm
}

// Schedule maps a step number to a learning rate.
type Schedule func(step int) float64

// Constant returns a schedule that always yields lr.
func Constant(lr float64) Schedule { return func(int) float64 { return lr } }

// Cosine returns linear warmup over warmup steps to maxLR, then cosine
// decay to minLR at total steps, then minLR thereafter.
func Cosine(warmup, total int, maxLR, minLR float64) Schedule {
	return func(step int) float64 {
		if step < warmup && warmup > 0 {
			return maxLR * float64(step+1) / float64(warmup)
		}
		if step >= total {
			return minLR
		}
		progress := float64(step-warmup) / float64(max(1, total-warmup))
		coeff := 0.5 * (1 + math.Cos(math.Pi*progress))
		return minLR + coeff*(maxLR-minLR)
	}
}
