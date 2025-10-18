package pnn

import (
	"encoding/json"
	"testing"
)

func makeTestBrain(seed int64) *PNN {
	b := NewPNN(seed)
	s := &Neuron{ID: "sens", Bias: 0, Activation: ActLinear}
	m := &Neuron{ID: "mot", Bias: 0, Activation: ActLinear}
	b.Neurons[s.ID] = s
	b.Neurons[m.ID] = m
	b.Inputs["sens"] = s.ID
	b.Outputs["mot"] = m.ID
	syn := &Synapse{ID: "s1", From: s.ID, To: m.ID, Weight: 0.1, Plastic: true, MinWeight: -2, MaxWeight: 2}
	b.Synapses[syn.ID] = syn
	s.Outgoing = append(s.Outgoing, syn.ID)
	m.Incoming = append(m.Incoming, syn.ID)
	b.Modulators["dopamine"] = &Modulator{Name: "dopamine", Level: 0, DecayRate: 1.0}
	return b
}

func TestDeterminism(t *testing.T) {
	b1 := makeTestBrain(1)
	b2 := makeTestBrain(1)

	var last1, last2 float64
	for i := 0; i < 10; i++ {
		outs1, _ := b1.Step(map[string]float64{"sens": 1.0}, 0.1)
		outs2, _ := b2.Step(map[string]float64{"sens": 1.0}, 0.1)
		if outs1["mot"] != outs2["mot"] {
			t.Fatalf("Non-deterministic outputs: %v vs %v", outs1, outs2)
		}
		last1 = outs1["mot"]
		last2 = outs2["mot"]
	}
	if last1 != last2 {
		t.Fatalf("Last outputs differ: %v vs %v", last1, last2)
	}
}

func TestHebbianIncrease(t *testing.T) {
	b := makeTestBrain(42)
	// pre/post correlated with dopamine pulse at step 2
	b.Step(map[string]float64{"sens": 1.0}, 0.1)
	b.Modulators["dopamine"].Level = 1.0
	b.Step(map[string]float64{"sens": 1.0}, 0.1)
	wAfter := b.Synapses["s1"].Weight
	if wAfter <= 0.1 {
		t.Fatalf("Expected weight to increase above initial 0.1, got %v", wAfter)
	}
}

func TestJSONRoundTrip(t *testing.T) {
	b := makeTestBrain(7)
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	var b2 PNN
	if err := json.Unmarshal(data, &b2); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if len(b2.Neurons) != len(b.Neurons) {
		t.Fatalf("Neuron count mismatch after roundtrip: %d vs %d", len(b2.Neurons), len(b.Neurons))
	}
}
