package brain

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewBrain(t *testing.T) {
	b := NewBrain(2, 1)
	assert.NotNil(t, b)
	assert.Equal(t, 2, b.NumInputs)
	assert.Equal(t, 1, b.NumOutputs)
	assert.NotNil(t, b.Lobes)
	assert.NotNil(t, b.Connections)
}

func TestAddAndGetLobe(t *testing.T) {
	b := NewBrain(2, 1)
	lobe := NewLobe("testLobe", 2, 1, nil)
	b.AddLobe(lobe)
	got := b.GetLobe("testLobe")
	assert.NotNil(t, got)
	assert.Equal(t, "testLobe", got.Name)
}

func TestConnectAndThink(t *testing.T) {
	b := NewBrain(2, 1)

	// Create a test lobe
	lobe := b.NewLobe("testLobe", 2, 1)
	b.AddLobe(lobe)

	slog.Info("Lobes:", "lobes", b.Lobes)
	slog.Info("Connections:", "connections", b.Connections)

	// Test the brain thinking process
	inputs := []float64{1.0, 2.0}
	lobe.SetInputs(inputs)
	b.Think()
	assert.Len(t, lobe.Outputs, 1)

	assert.Equal(t, inputs[0], lobe.Outputs[0])
}

func TestThinkWithMultipleLobes(t *testing.T) {
	b := NewBrain(4, 4)

	// Create lobes
	lobeA := b.NewLobe("lobe_a", 2, 2)
	lobeB := b.NewLobe("lobe_b", 2, 2)
	lobeC := b.NewLobe("lobe_c", 4, 4)

	// Connect A and B to each side of C's inputs
	b.Connect("conn1", "lobe_a", "lobe_c", 0, 0)
	b.Connect("conn2", "lobe_a", "lobe_c", 1, 1)

	b.Connect("conn3", "lobe_b", "lobe_c", 0, 2)
	b.Connect("conn4", "lobe_b", "lobe_c", 1, 3)

	// Set inputs for A and B
	inputsA := []float64{1.0, 2.0}
	inputsB := []float64{3.0, 4.0}
	lobeA.SetInputs(inputsA)
	lobeB.SetInputs(inputsB)

	// Think
	b.Think()

	// Check outputs
	assert.Len(t, lobeA.Outputs, 2)
	assert.Len(t, lobeB.Outputs, 2)
	assert.Len(t, lobeC.Outputs, 4)

	assert.Equal(t, inputsA[0], lobeA.Outputs[0])
	assert.Equal(t, inputsA[1], lobeA.Outputs[1])

	assert.Equal(t, inputsB[0], lobeB.Outputs[0])
	assert.Equal(t, inputsB[1], lobeB.Outputs[1])

	assert.Equal(t, inputsA[0], lobeC.Outputs[0])
	assert.Equal(t, inputsA[1], lobeC.Outputs[1])
	assert.Equal(t, inputsB[0], lobeC.Outputs[2])
	assert.Equal(t, inputsB[1], lobeC.Outputs[3])
}
