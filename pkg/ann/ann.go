package ann

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
)

type ANN struct {
	NumInputs     int           `json:"numInputs"`
	NumOutputs    int           `json:"numOutputs"`
	NumLayers     int           `json:"numLayers"`
	Weights       [][][]float64 `json:"weights"`
	Biases        [][]float64   `json:"biases"`
	CurrentValues [][]float64   `json:"currentValues"`
}

// Save writes the ANN to a file as JSON
func (a *ANN) Save(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	return enc.Encode(a)
}

// LoadANN loads an ANN from a JSON file
func LoadANN(filename string) (*ANN, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var ann ANN
	dec := json.NewDecoder(f)
	if err := dec.Decode(&ann); err != nil {
		return nil, err
	}
	return &ann, nil
}

func NewANN(numInputs int, numOutputs int, numLayers int, hiddenSize int) *ANN {
	ann := &ANN{
		NumInputs:  numInputs,
		NumOutputs: numOutputs,
		NumLayers:  numLayers,
		Weights:    make([][][]float64, numLayers+1), // hidden layers + output layer
		Biases:     make([][]float64, numLayers+1),
	}

	// Input to first hidden
	prevSize := numInputs
	for l := 0; l < numLayers; l++ {
		ann.Weights[l] = make([][]float64, hiddenSize)
		ann.Biases[l] = make([]float64, hiddenSize)
		for n := 0; n < hiddenSize; n++ {
			ann.Weights[l][n] = make([]float64, prevSize)
			for i := 0; i < prevSize; i++ {
				ann.Weights[l][n][i] = rand.NormFloat64() * 0.1
			}
			ann.Biases[l][n] = rand.NormFloat64() * 0.1
		}
		prevSize = hiddenSize
	}
	// Last layer: hidden to output
	ann.Weights[numLayers] = make([][]float64, numOutputs)
	ann.Biases[numLayers] = make([]float64, numOutputs)
	for n := 0; n < numOutputs; n++ {
		ann.Weights[numLayers][n] = make([]float64, prevSize)
		for i := 0; i < prevSize; i++ {
			ann.Weights[numLayers][n][i] = rand.NormFloat64() * 0.1
		}
		ann.Biases[numLayers][n] = rand.NormFloat64() * 0.1
	}
	return ann
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Process runs a forward pass.
//
// A wrong-sized input used to index out of range and panic; it now returns nil,
// which callers can check. Use ProcessChecked for an explicit error.
func (a *ANN) Process(inputs []float64) []float64 {
	out, err := a.ProcessChecked(inputs)
	if err != nil {
		return nil
	}
	return out
}

// ProcessChecked runs a forward pass, reporting a malformed input rather than
// panicking on it.
func (a *ANN) ProcessChecked(inputs []float64) ([]float64, error) {
	if len(inputs) != a.NumInputs {
		return nil, fmt.Errorf("ann: expected %d inputs, got %d", a.NumInputs, len(inputs))
	}
	return a.process(inputs), nil
}

func (a *ANN) process(inputs []float64) []float64 {
	// Reuse the CurrentValues buffers across calls. Process allocated the whole
	// layer stack every time, which matters for anything calling it in a loop.
	if len(a.CurrentValues) != a.NumLayers+2 {
		a.CurrentValues = make([][]float64, a.NumLayers+2) // input + hidden + output
	}
	if len(a.CurrentValues[0]) != len(inputs) {
		a.CurrentValues[0] = make([]float64, len(inputs))
	}
	copy(a.CurrentValues[0], inputs)

	layerInput := inputs
	for l := 0; l < a.NumLayers; l++ {
		next := make([]float64, len(a.Biases[l]))
		for n := range next {
			sum := a.Biases[l][n]
			for i, w := range a.Weights[l][n] {
				sum += w * layerInput[i]
			}
			next[n] = relu(sum)
		}
		a.CurrentValues[l+1] = make([]float64, len(next))
		copy(a.CurrentValues[l+1], next)
		layerInput = next
	}
	// Output layer (no activation)
	out := make([]float64, a.NumOutputs)
	for n := 0; n < a.NumOutputs; n++ {
		sum := a.Biases[a.NumLayers][n]
		for i, w := range a.Weights[a.NumLayers][n] {
			sum += w * layerInput[i]
		}
		out[n] = sum
	}
	a.CurrentValues[a.NumLayers+1] = make([]float64, len(out))
	copy(a.CurrentValues[a.NumLayers+1], out)
	return out
}

// Backpropagate performs one training step. A malformed input or target is
// reported rather than panicking mid-pass.
func (a *ANN) Backpropagate(inputs, targets []float64, learningRate float64) error {
	if len(inputs) != a.NumInputs {
		return fmt.Errorf("ann: expected %d inputs, got %d", a.NumInputs, len(inputs))
	}
	if len(targets) != a.NumOutputs {
		return fmt.Errorf("ann: expected %d targets, got %d", a.NumOutputs, len(targets))
	}
	a.backpropagate(inputs, targets, learningRate)
	return nil
}

func (a *ANN) backpropagate(inputs, targets []float64, learningRate float64) {
	// Forward pass: store activations and weighted sums for each layer
	activations := make([][]float64, a.NumLayers+2) // input + hidden + output
	zs := make([][]float64, a.NumLayers+1)
	activations[0] = inputs
	layerInput := inputs
	for l := 0; l < a.NumLayers; l++ {
		z := make([]float64, len(a.Biases[l]))
		aAct := make([]float64, len(a.Biases[l]))
		for n := range z {
			sum := a.Biases[l][n]
			for i, w := range a.Weights[l][n] {
				sum += w * layerInput[i]
			}
			z[n] = sum
			aAct[n] = relu(sum)
		}
		zs[l] = z
		activations[l+1] = aAct
		layerInput = aAct
	}
	// Output layer
	zOut := make([]float64, a.NumOutputs)
	out := make([]float64, a.NumOutputs)
	for n := 0; n < a.NumOutputs; n++ {
		sum := a.Biases[a.NumLayers][n]
		for i, w := range a.Weights[a.NumLayers][n] {
			sum += w * layerInput[i]
		}
		zOut[n] = sum
		out[n] = sum // linear output
	}
	zs[a.NumLayers] = zOut
	activations[a.NumLayers+1] = out

	// Backward pass
	// Output layer delta
	delta := make([][]float64, a.NumLayers+1)
	delta[a.NumLayers] = make([]float64, a.NumOutputs)
	for n := 0; n < a.NumOutputs; n++ {
		delta[a.NumLayers][n] = out[n] - targets[n] // dLoss/dOut for MSE
	}
	// Hidden layers delta
	for l := a.NumLayers - 1; l >= 0; l-- {
		delta[l] = make([]float64, len(a.Biases[l]))
		for i := range delta[l] {
			var sum float64
			for k := range delta[l+1] {
				sum += a.Weights[l+1][k][i] * delta[l+1][k]
			}
			// relu derivative
			if zs[l][i] > 0 {
				delta[l][i] = sum
			} else {
				delta[l][i] = 0
			}
		}
	}
	// Update weights and biases
	for l := a.NumLayers; l >= 0; l-- {
		for n := range a.Weights[l] {
			for j := range a.Weights[l][n] {
				a.Weights[l][n][j] -= learningRate * delta[l][n] * activations[l][j]
			}
			a.Biases[l][n] -= learningRate * delta[l][n]
		}
	}
}
