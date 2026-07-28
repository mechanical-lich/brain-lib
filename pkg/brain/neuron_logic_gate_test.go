package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type logicGateTestCase struct {
	inputs   []float64
	expected float64
}

func TestNeuronAND(t *testing.T) {
	l := NewLobe("prop", nil)

	// Create neurons A, B, C
	a := &Neuron{ID: "A", Bias: 0.0, Activation: ActLinear}
	b := &Neuron{ID: "B", Bias: 0.0, Activation: ActLinear}
	c := &Neuron{ID: "C", Bias: -1.0, Activation: ActLinear}

	l.AddNeuron(a)
	l.AddNeuron(b)
	l.AddNeuron(c)

	// Connect A->C (1.0), B->C (1.0)
	l.Connect("A", "C", 1.0, false)
	l.Connect("B", "C", 1.0, false)

	for _, tc := range []logicGateTestCase{
		{inputs: []float64{0.0, 0.0}, expected: 0.0},
		{inputs: []float64{0.0, 1.0}, expected: 0.0},
		{inputs: []float64{1.0, 0.0}, expected: 0.0},
		{inputs: []float64{1.0, 1.0}, expected: 1.0},
	} {
		// Set inputs
		a.ExternalInput = tc.inputs[0]
		b.ExternalInput = tc.inputs[1]

		// Run one step
		l.Step(0.1, 0, nil)

		// Check output
		assert.InDelta(t, tc.expected, l.GetNeuron("C").ActivationVal, 1e-6, "inputs: %v, output: %v", tc.inputs, l.GetNeuron("C").ActivationVal)
	}
}

func TestNeuronOR(t *testing.T) {
	l := NewLobe("prop", nil)

	// Create neurons A, B, C
	a := &Neuron{ID: "A", Bias: 0.0, Activation: ActLinear}
	b := &Neuron{ID: "B", Bias: 0.0, Activation: ActLinear}
	c := &Neuron{ID: "C", Bias: -1.0, Activation: ActLinear, Max: 1.0}

	l.AddNeuron(a)
	l.AddNeuron(b)
	l.AddNeuron(c)

	// Connect A->C (2.0), B->C (2.0)
	l.Connect("A", "C", 2.0, false)
	l.Connect("B", "C", 2.0, false)

	for _, tc := range []logicGateTestCase{
		{inputs: []float64{0.0, 0.0}, expected: 0.0},
		{inputs: []float64{0.0, 1.0}, expected: 1.0},
		{inputs: []float64{1.0, 0.0}, expected: 1.0},
		{inputs: []float64{1.0, 1.0}, expected: 1.0},
	} {
		// Set inputs
		a.ExternalInput = tc.inputs[0]
		b.ExternalInput = tc.inputs[1]

		// Run one step
		l.Step(0.1, 0, nil)

		// Check output
		assert.InDelta(t, tc.expected, l.GetNeuron("C").ActivationVal, 1e-6, "inputs: %v, output: %v", tc.inputs, l.GetNeuron("C").ActivationVal)
	}
}

func TestNeuronNOT(t *testing.T) {
	l := NewLobe("prop", nil)

	// Create neurons A, B, C
	a := &Neuron{ID: "A", Bias: 0.0, Activation: ActLinear}

	c := &Neuron{ID: "C", Bias: 1.0, Activation: ActLinear}

	l.AddNeuron(a)

	l.AddNeuron(c)

	// Connect A->C (2.0)
	l.Connect("A", "C", -1.0, false)

	for _, tc := range []logicGateTestCase{
		{inputs: []float64{0.0}, expected: 1.0},
		{inputs: []float64{1.0}, expected: 0.0},
	} {
		// Set inputs
		a.ExternalInput = tc.inputs[0]

		// Run one step
		l.Step(0.1, 0, nil)

		// Check output
		assert.InDelta(t, tc.expected, l.GetNeuron("C").ActivationVal, 1e-6, "inputs: %v, output: %v", tc.inputs, l.GetNeuron("C").ActivationVal)
	}
}

func TestNeuronNOR(t *testing.T) {
	l := NewLobe("prop", nil)

	// Create neurons A, B, C
	a := &Neuron{ID: "A", Bias: 0.0, Activation: ActLinear}
	b := &Neuron{ID: "B", Bias: 0.0, Activation: ActLinear}
	c := &Neuron{ID: "C", Bias: 1.0, Activation: ActLinear}

	l.AddNeuron(a)
	l.AddNeuron(b)
	l.AddNeuron(c)

	// Connect A->C (-1.0), B->C (-1.0)
	l.Connect("A", "C", -1.0, false)
	l.Connect("B", "C", -1.0, false)

	for _, tc := range []logicGateTestCase{
		{inputs: []float64{0.0, 0.0}, expected: 1.0},
		{inputs: []float64{0.0, 1.0}, expected: 0.0},
		{inputs: []float64{1.0, 0.0}, expected: 0.0},
		{inputs: []float64{1.0, 1.0}, expected: 0.0},
	} {
		// Set inputs
		a.ExternalInput = tc.inputs[0]
		b.ExternalInput = tc.inputs[1]

		// Run one step
		l.Step(0.1, 0, nil)

		// Check output
		assert.InDelta(t, tc.expected, l.GetNeuron("C").ActivationVal, 1e-6, "inputs: %v, actual output: %v, expected output: %v", tc.inputs, l.GetNeuron("C").ActivationVal, tc.expected)
	}
}

func TestNeuronNAND(t *testing.T) {
	l := NewLobe("prop", nil)

	// Create neurons A, B, C
	a := &Neuron{ID: "A", Bias: 0.0, Activation: ActLinear}
	b := &Neuron{ID: "B", Bias: 0.0, Activation: ActLinear}
	c := &Neuron{ID: "C", Bias: 2.0, Activation: ActLinear, Max: 1.0}

	l.AddNeuron(a)
	l.AddNeuron(b)
	l.AddNeuron(c)

	// Connect A->C (-1.0), B->C (-1.0)
	l.Connect("A", "C", -1.0, false)
	l.Connect("B", "C", -1.0, false)

	for _, tc := range []logicGateTestCase{
		{inputs: []float64{0.0, 0.0}, expected: 1.0},
		{inputs: []float64{0.0, 1.0}, expected: 1.0},
		{inputs: []float64{1.0, 0.0}, expected: 1.0},
		{inputs: []float64{1.0, 1.0}, expected: 0.0},
	} {
		// Set inputs
		a.ExternalInput = tc.inputs[0]
		b.ExternalInput = tc.inputs[1]

		// Run one step
		l.Step(0.1, 0, nil)

		// Check output
		assert.InDelta(t, tc.expected, l.GetNeuron("C").ActivationVal, 1e-6, "inputs: %v, actual output: %v, expected output: %v", tc.inputs, l.GetNeuron("C").ActivationVal, tc.expected)
	}
}

func TestNeuronXNOR(t *testing.T) {
	l := NewLobe("prop", nil)

	// Create neurons A, B, C
	// XNOR outputs 1 when inputs are the same (both 0 or both 1)
	// XNOR(A,B) = (A AND B) OR (NOT A AND NOT B)
	a := &Neuron{ID: "A", Bias: 0.0, Activation: ActLinear}
	b := &Neuron{ID: "B", Bias: 0.0, Activation: ActLinear}
	notA := &Neuron{ID: "NOT_A", Bias: 1.0, Activation: ActLinear}
	notB := &Neuron{ID: "NOT_B", Bias: 1.0, Activation: ActLinear}
	andAB := &Neuron{ID: "AND_AB", Bias: -1.0, Activation: ActLinear}
	andNotANotB := &Neuron{ID: "AND_NOT_A_NOT_B", Bias: -1.0, Activation: ActLinear}
	c := &Neuron{ID: "C", Bias: 0.0, Activation: ActLinear}

	l.AddNeuron(a)
	l.AddNeuron(b)
	l.AddNeuron(notA)
	l.AddNeuron(notB)
	l.AddNeuron(andAB)
	l.AddNeuron(andNotANotB)
	l.AddNeuron(c)

	// Connect A->NOT_A (-1.0), B->NOT_B (-1.0)
	l.Connect("A", "NOT_A", -1.0, false)
	l.Connect("B", "NOT_B", -1.0, false)

	// Connect A->AND_AB (1.0), B->AND_AB (1.0)
	l.Connect("A", "AND_AB", 1.0, false)
	l.Connect("B", "AND_AB", 1.0, false)

	// Connect NOT_A->AND_NOT_A_NOT_B (1.0), NOT_B->AND_NOT_A_NOT_B (1.0)
	l.Connect("NOT_A", "AND_NOT_A_NOT_B", 1.0, false)
	l.Connect("NOT_B", "AND_NOT_A_NOT_B", 1.0, false)

	// Connect AND_AB->C (1.0), AND_NOT_A_NOT_B->C (1.0)
	l.Connect("AND_AB", "C", 1.0, false)
	l.Connect("AND_NOT_A_NOT_B", "C", 1.0, false)

	for _, tc := range []logicGateTestCase{
		{inputs: []float64{0.0, 0.0}, expected: 1.0},
		{inputs: []float64{0.0, 1.0}, expected: 0.0},
		{inputs: []float64{1.0, 0.0}, expected: 0.0},
		{inputs: []float64{1.0, 1.0}, expected: 1.0},
	} {
		// Set inputs
		a.ExternalInput = tc.inputs[0]
		b.ExternalInput = tc.inputs[1]

		// Run one step
		l.Step(0.1, 0, nil)

		// Check output
		assert.InDelta(t, tc.expected, l.GetNeuron("C").ActivationVal, 1e-6, "inputs: %v, actual output: %v, expected output: %v", tc.inputs, l.GetNeuron("C").ActivationVal, tc.expected)
	}
}
