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
	b.Connect("conn1", BRAIN_INPUT, "testLobe", 0, 0)
	b.Connect("conn2", "testLobe", BRAIN_OUTPUT, 0, 0)

	slog.Info("Lobes:", "lobes", b.Lobes)
	slog.Info("Connections:", "connections", b.Connections)

	// Test the brain thinking process
	inputs := []float64{1.0, 2.0}
	outputs := b.Think(inputs)
	assert.Len(t, outputs, 1)

	assert.Equal(t, inputs[0], outputs[0])
}

func TestThinkWithMultipleLobes(t *testing.T) {
	b := NewBrain(4, 4)

	// Create lobes
	lobe1 := b.NewLobe("lobe1", 2, 2)
	lobe2 := b.NewLobe("lobe2", 2, 2)
	b.AddLobe(lobe1)
	b.AddLobe(lobe2)

	// Connect brain inputs to lobe1
	b.Connect("conn1", BRAIN_INPUT, "lobe1", 0, 0)
	b.Connect("conn2", BRAIN_INPUT, "lobe1", 1, 1)

	// Connect brain inputs to lobe2
	b.Connect("conn3", BRAIN_INPUT, "lobe2", 2, 0)
	b.Connect("conn4", BRAIN_INPUT, "lobe2", 3, 1)

	// Connect lobe1 outputs to brain outputs
	b.Connect("conn5", "lobe1", BRAIN_OUTPUT, 0, 0)
	b.Connect("conn6", "lobe1", BRAIN_OUTPUT, 1, 1)

	// Connect lobe2 outputs to brain outputs
	b.Connect("conn7", "lobe2", BRAIN_OUTPUT, 0, 2)
	b.Connect("conn8", "lobe2", BRAIN_OUTPUT, 1, 3)

	inputs := []float64{1.0, 2.0, 3.0, 4.0}
	outputs := b.Think(inputs)
	assert.Len(t, outputs, 4)

	// Check outputs from lobe1
	assert.Equal(t, inputs[0], outputs[0])
	assert.Equal(t, inputs[1], outputs[1])

	// Check outputs from lobe2
	assert.Equal(t, inputs[2], outputs[2])
	assert.Equal(t, inputs[3], outputs[3])
}

func TestThinkWithMultipleLobesAndLayers(t *testing.T) {
	b := NewBrain(4, 4)

	// Create lobes
	lobe1 := b.NewLobe("lobe1", 2, 2)
	lobe2 := b.NewLobe("lobe2", 2, 2)
	lobe3 := b.NewLobe("lobe3", 2, 2)
	lobe4 := b.NewLobe("lobe4", 2, 2)

	b.AddLobe(lobe1)
	b.AddLobe(lobe2)
	b.AddLobe(lobe3)
	b.AddLobe(lobe4)

	// Connect brain inputs to lobe1
	b.Connect("conn1", BRAIN_INPUT, "lobe1", 0, 0)
	b.Connect("conn2", BRAIN_INPUT, "lobe1", 1, 1)

	// Connect brain inputs to lobe2
	b.Connect("conn3", BRAIN_INPUT, "lobe2", 2, 0)
	b.Connect("conn4", BRAIN_INPUT, "lobe2", 3, 1)

	// Connect lobe3 outputs to brain outputs
	b.Connect("conn5", "lobe3", BRAIN_OUTPUT, 0, 0)
	b.Connect("conn6", "lobe3", BRAIN_OUTPUT, 1, 1)

	// Connect lobe4 outputs to brain outputs
	b.Connect("conn7", "lobe4", BRAIN_OUTPUT, 0, 2)
	b.Connect("conn8", "lobe4", BRAIN_OUTPUT, 1, 3)

	// Connect Lobe 1 outputs to Lobe 3 inputs
	b.Connect("conn9", "lobe1", "lobe3", 0, 0)
	b.Connect("conn10", "lobe1", "lobe3", 1, 1)

	// Connect Lobe 2 outputs to Lobe 4 inputs
	b.Connect("conn11", "lobe2", "lobe4", 0, 0)
	b.Connect("conn12", "lobe2", "lobe4", 1, 1)

	inputs := []float64{1.0, 2.0, 3.0, 4.0}
	outputs := b.Think(inputs)
	assert.Len(t, outputs, 4)

	// Check outputs from lobe1
	assert.Equal(t, inputs[0], outputs[0])
	assert.Equal(t, inputs[1], outputs[1])

	// Check outputs from lobe2
	assert.Equal(t, inputs[2], outputs[2])
	assert.Equal(t, inputs[3], outputs[3])

	slog.Info("Brain inputs:", "inputs", inputs)
	slog.Info("Brain outputs:", "outputs", outputs)

}

func TestThinkingWithConnectionsPointedAtSameOutput(t *testing.T) {
	b := NewBrain(2, 1)

	// Create a test lobe
	lobe := b.NewLobe("testLobe", 2, 2)
	b.AddLobe(lobe)
	b.Connect("conn1", BRAIN_INPUT, "testLobe", 0, 0)
	b.Connect("conn2", BRAIN_INPUT, "testLobe", 1, 1)

	b.Connect("conn3", "testLobe", BRAIN_OUTPUT, 0, 0)
	b.Connect("conn4", "testLobe", BRAIN_OUTPUT, 1, 0)

	slog.Info("Lobes:", "lobes", b.Lobes)
	slog.Info("Connections:", "connections", b.Connections)

	// Test the brain thinking process
	inputs := []float64{2.0, 2.0}
	outputs := b.Think(inputs)
	assert.Len(t, outputs, 1)

	// With weights of 0.5, the output should be (2*0.5 + 2*0.5) = 2.0
	assert.Equal(t, 4.0, outputs[0])
}

func TestThinkingWithWeightedConnectionsPointedAtSameOutput(t *testing.T) {
	b := NewBrain(2, 1)

	// Create a test lobe
	lobe := b.NewLobe("testLobe", 2, 2)
	b.AddLobe(lobe)
	b.Connect("conn1", BRAIN_INPUT, "testLobe", 0, 0)
	b.Connect("conn2", BRAIN_INPUT, "testLobe", 1, 1)

	b.WeightedConnect("conn3", "testLobe", BRAIN_OUTPUT, 0, 0, 0.5)
	b.WeightedConnect("conn4", "testLobe", BRAIN_OUTPUT, 1, 0, 0.5)

	slog.Info("Lobes:", "lobes", b.Lobes)
	slog.Info("Connections:", "connections", b.Connections)

	// Test the brain thinking process
	inputs := []float64{2.0, 2.0}
	outputs := b.Think(inputs)
	assert.Len(t, outputs, 1)

	// With weights of 0.5, the output should be (2*0.5 + 2*0.5) = 2.0
	assert.Equal(t, 2.0, outputs[0])
}
