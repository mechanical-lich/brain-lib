package brain

import "github.com/mechanical-lich/brain-lib/pkg/ann"

type Lobe struct {
	Name       string
	numInputs  int
	numOutputs int
	outputs    []float64
	ann        *ann.ANN
}

func NewLobe(name string, numInputs int, numOutputs int, ann *ann.ANN) *Lobe {
	return &Lobe{Name: name, numInputs: numInputs, numOutputs: numOutputs, outputs: make([]float64, numOutputs), ann: ann}
}

func (l *Lobe) Process(inputs []float64) []float64 {
	l.outputs = make([]float64, l.numOutputs)

	// Passthrough lobe
	if l.ann == nil {
		for i := range l.outputs {
			l.outputs[i] = inputs[i]
		}
	}
	return l.outputs
}

func (l *Lobe) GetOutputs() []float64 {
	return l.outputs
}
