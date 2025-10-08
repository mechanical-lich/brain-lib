package brain

import "github.com/mechanical-lich/brain-lib/pkg/ann"

type Lobe struct {
	Name       string    `json:"name"`
	NumInputs  int       `json:"num_inputs"`
	NumOutputs int       `json:"num_outputs"`
	Inputs     []float64 `json:"inputs"`
	Outputs    []float64 `json:"outputs"`
	Ann        *ann.ANN  `json:"ann"`
}

func NewLobe(name string, NumInputs int, NumOutputs int, ann *ann.ANN) *Lobe {
	return &Lobe{Name: name, NumInputs: NumInputs, NumOutputs: NumOutputs, Inputs: make([]float64, NumInputs), Outputs: make([]float64, NumOutputs), Ann: ann}
}

// Set the inputs that'll be used during the processing step
func (l *Lobe) SetInputs(inputs []float64) {
	l.Inputs = inputs
}

func (l *Lobe) Process() {
	l.Outputs = make([]float64, l.NumOutputs)
	// Passthrough lobe
	if l.Ann == nil {
		for i := range l.Outputs {
			l.Outputs[i] = l.Inputs[i]
		}
	} else {
		l.Outputs = l.Ann.Process(l.Inputs)
	}
}

func (l *Lobe) ClearOutputs() {
	l.Outputs = make([]float64, l.NumOutputs)
}

func (l *Lobe) ClearInputs() {
	l.Inputs = make([]float64, l.NumInputs)
}
