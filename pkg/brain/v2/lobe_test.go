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
	assert.NotNil(t, l.Outputs)
}

func TestLobeProcessPassthrough(t *testing.T) {
	l := NewLobe("bar", 2, 2, nil)
	inputs := []float64{3.0, 4.0}
	l.SetInputs(inputs)
	l.Process()
	assert.Len(t, l.Outputs, 2)
	for i, out := range l.Outputs {
		assert.Equal(t, inputs[i], out)
	}
}
