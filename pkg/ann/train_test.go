package ann

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTrainXOR(t *testing.T) {
	// XOR problem: 2 inputs, 1 output
	training := []TrainingExample{
		{Inputs: []float64{0, 0}, Outputs: []float64{0}},
		{Inputs: []float64{0, 1}, Outputs: []float64{1}},
		{Inputs: []float64{1, 0}, Outputs: []float64{1}},
		{Inputs: []float64{1, 1}, Outputs: []float64{0}},
	}
	ann := NewANN(2, 1, 1, 10)
	Train(ann, training, 1000, 0.1)

	for _, ex := range training {
		out := ann.Process(ex.Inputs)
		// Output should be close to target (within 0.1)
		assert.InDelta(t, ex.Outputs[0], out[0], 0.1, "XOR output for %v - %v", ex.Inputs, out[0])
	}
}

func TestTrainConverges(t *testing.T) {
	// Simple identity mapping
	training := []TrainingExample{
		{Inputs: []float64{0.0}, Outputs: []float64{0.0}},
		{Inputs: []float64{1.0}, Outputs: []float64{1.0}},
	}
	ann := NewANN(1, 1, 1, 2)
	Train(ann, training, 500, 0.1)

	for _, ex := range training {
		out := ann.Process(ex.Inputs)
		assert.InDelta(t, ex.Outputs[0], out[0], 0.1, "Identity output for %v - %v", ex.Inputs, out[0])
	}
}
