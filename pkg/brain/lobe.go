package brain

import (
	"github.com/mechanical-lich/brain-lib/pkg/ann"
)

type Lobe struct {
	Name       string
	NumInputs  int
	NumOutputs int
	outputs    []float64
	Ann        *ann.ANN
}

func NewLobe(name string, NumInputs int, NumOutputs int, ann *ann.ANN) *Lobe {
	return &Lobe{Name: name, NumInputs: NumInputs, NumOutputs: NumOutputs, outputs: make([]float64, NumOutputs), Ann: ann}
}

func (l *Lobe) Process(inputs []float64) []float64 {
	l.outputs = make([]float64, l.NumOutputs)
	// Passthrough lobe
	if l.Ann == nil {
		for i := range l.outputs {
			l.outputs[i] = inputs[i]
		}
	} else {
		l.outputs = l.Ann.Process(inputs)
	}
	return l.outputs
}

func (l *Lobe) GetOutputs() []float64 {
	return l.outputs
}
