package ann

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewANNConfig(t *testing.T) {
	inputs, outputs, layers, hidden := 3, 2, 2, 4
	ann := NewANN(inputs, outputs, layers, hidden)
	assert.Equal(t, inputs, ann.NumInputs, "numInputs")
	assert.Equal(t, outputs, ann.NumOutputs, "numOutputs")
	assert.Equal(t, layers, ann.NumLayers, "numLayers")
	assert.Equal(t, layers+1, len(ann.Weights), "weights layers")
	assert.Equal(t, layers+1, len(ann.Biases), "biases layers")
}

func TestProcessShape(t *testing.T) {
	inputs, outputs, layers, hidden := 3, 2, 2, 4
	ann := NewANN(inputs, outputs, layers, hidden)
	in := []float64{1.0, 2.0, 3.0}
	out := ann.Process(in)
	assert.Equal(t, outputs, len(out), "output len")
}

func TestProcessDeterministic(t *testing.T) {
	inputs, outputs, layers, hidden := 2, 1, 1, 2
	ann := NewANN(inputs, outputs, layers, hidden)
	// Overwrite weights/biases for deterministic output
	for l := range ann.Weights {
		for n := range ann.Weights[l] {
			for i := range ann.Weights[l][n] {
				ann.Weights[l][n][i] = 1.0
			}
			ann.Biases[l][n] = 0.0
		}
	}
	in := []float64{1, 2}
	out := ann.Process(in)
	assert.Equal(t, outputs, len(out), "output len")
	assert.NotZero(t, out[0], "output is zero, expected nonzero")
}
