package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewLobe(t *testing.T) {
	l := NewLobe("foo", 2, 1, nil)
	assert.NotNil(t, l)
	assert.Equal(t, "foo", l.Name)
	assert.Equal(t, 2, l.NumInputs)
	assert.Equal(t, 1, l.NumOutputs)
	assert.NotNil(t, l.outputs)
}

func TestLobeProcessPassthrough(t *testing.T) {
	l := NewLobe("bar", 2, 2, nil)
	inputs := []float64{3.0, 4.0}
	outputs := l.Process(inputs)
	assert.Len(t, outputs, 2)
	for i, out := range outputs {
		assert.Equal(t, inputs[i], out)
	}
}

func TestGetOutputs(t *testing.T) {
	l := NewLobe("baz", 1, 1, nil)
	l.Process([]float64{2.0})
	outs := l.GetOutputs()
	assert.Len(t, outs, 1)
}
