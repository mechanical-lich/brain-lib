package gpu

import (
	"fmt"
	"math"
)

var kAdamW = &kernel{name: "adamw", binds: "wrwwr", src: `
struct P {
    n: u32, lr: f32, beta1: f32, beta2: f32,
    eps: f32, decay: f32, bc1: f32, bc2: f32,
    max_norm: f32, use_norm: u32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read> sumsq: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
` + elemHead + `    var gs: f32 = 1.0;
    if (p.use_norm != 0u) {
        let norm = sqrt(sumsq[0]);
        if (p.max_norm > 0.0 && norm > p.max_norm) { gs = p.max_norm / (norm + 1e-6); }
    }
    let gi = g[i] * gs;
    let mi = p.beta1 * m[i] + (1.0 - p.beta1) * gi;
    let vi = p.beta2 * v[i] + (1.0 - p.beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    var wi = w[i] - p.lr * (mi / p.bc1) / (sqrt(vi / p.bc2) + p.eps);
    if (p.decay != 0.0) { wi = wi - p.decay * wi; }
    w[i] = wi;
}
`}

// AdamWStep holds the hyperparameters of one train.AdamW update.
type AdamWStep struct {
	LR, Beta1, Beta2, Eps float64
	WeightDecay           float64 // applied when Decay is set, as LR*WeightDecay
	Decay                 bool
	Step                  int // 1-based update number, for bias correction
	// MaxNorm > 0 clips the gradient to that global L2 norm, as
	// train.ClipGradNorm does, using the sum of squared gradients over all
	// parameters already reduced into SumSq[0] by Sum(..., squares=true).
	MaxNorm float64
}

// AdamW records one train.AdamW update of w with gradient g and moments m
// and v. The bias-correction terms are computed here on the host from
// h.Step, so the whole update needs no readback. With clipping, sumSq is
// the one-element tensor holding the squared global gradient norm; the
// gradient is scaled on the fly and g itself is left unscaled. sumSq is
// ignored when h.MaxNorm is 0 and may be nil.
func (b *CommandBatch) AdamW(w, g, m, v, sumSq *Tensor, h AdamWStep) {
	b.same("AdamW", w, g)
	b.same("AdamW", w, m)
	b.same("AdamW", w, v)
	if h.Step < 1 {
		panic(fmt.Sprintf("gpu: AdamW step %d, want >= 1", h.Step))
	}
	decay := 0.0
	if h.Decay {
		decay = h.LR * h.WeightDecay
	}
	bc1 := 1 - math.Pow(h.Beta1, float64(h.Step))
	bc2 := 1 - math.Pow(h.Beta2, float64(h.Step))
	norm := g.buf // any read-only binding will do when not clipping
	useNorm := h.MaxNorm > 0
	if useNorm {
		norm = sumSq.buf
	}
	params := []uint32{
		uint32(w.Len()), fbits(h.LR), fbits(h.Beta1), fbits(h.Beta2),
		fbits(h.Eps), fbits(decay), fbits(bc1), fbits(bc2),
		fbits(h.MaxNorm), flag(useNorm), 0, 0,
	}
	b.dispatch(kAdamW, elems(w.Len()), params, w.buf, g.buf, m.buf, v.buf, norm)
}
