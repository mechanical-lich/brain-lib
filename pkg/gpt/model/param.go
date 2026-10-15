package model

import "github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"

// Param is a trainable matrix together with its accumulated gradient.
//
// Backward passes add into Grad; call ZeroGrad (or GPT.ZeroGrads) before each
// new forward/backward pair. Decay marks parameters that weight decay should
// apply to (weight matrices and embeddings, not biases or LayerNorm gains).
type Param struct {
	Name  string
	Value *tensor.Mat
	Grad  *tensor.Mat
	Decay bool
}

func newParam(name string, value *tensor.Mat, decay bool) *Param {
	return &Param{
		Name:  name,
		Value: value,
		Grad:  tensor.New(value.Rows, value.Cols),
		Decay: decay,
	}
}

// ZeroGrad clears the accumulated gradient.
func (p *Param) ZeroGrad() { p.Grad.Zero() }

// Size is the number of scalar values in the parameter.
func (p *Param) Size() int { return len(p.Value.Data) }
