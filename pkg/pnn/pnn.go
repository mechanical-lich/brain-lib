package pnn

import (
	"encoding/json"
	"math/rand"
)

type PNN struct {
	Neurons    map[string]*Neuron    `json:"neurons"`
	Synapses   map[string]*Synapse   `json:"synapses"`
	Inputs     map[string]string     `json:"inputs"`  // sensorName -> neuronID
	Outputs    map[string]string     `json:"outputs"` // motorName -> neuronID
	Modulators map[string]*Modulator `json:"modulators"`
	Seed       int64                 `json:"seed"`
	rng        *rand.Rand
}

func NewPNN(seed int64) *PNN {
	if seed == 0 {
		seed = 1
	}
	return &PNN{
		Neurons:    make(map[string]*Neuron),
		Synapses:   make(map[string]*Synapse),
		Inputs:     make(map[string]string),
		Outputs:    make(map[string]string),
		Modulators: make(map[string]*Modulator),
		Seed:       seed,
		rng:        rand.New(rand.NewSource(seed)),
	}
}

// InitRNG initializes the RNG after unmarshalling or when needed.
func (b *PNN) InitRNG() {
	if b.Seed == 0 {
		b.Seed = 1
	}
	b.rng = rand.New(rand.NewSource(b.Seed))
}

type Diagnostics struct {
	Weights map[string]float64 `json:"weights"`
	Mods    map[string]float64 `json:"mods"`
}

// Step processes the provided sensor inputs (0..1) and returns motor outputs.
func (b *PNN) Step(inputs map[string]float64, dt float64) (map[string]float64, Diagnostics) {
	// collect sums for each neuron
	sums := make(map[string]float64)
	// apply input mapping
	for name, val := range inputs {
		if nid, ok := b.Inputs[name]; ok {
			sums[nid] += val
		}
	}

	// propagate through synapses using previous activations
	// first compute neuron outputs without lateral propagation (synchronous update)
	outs := make(map[string]float64)
	for id, n := range b.Neurons {
		// sum incoming synapses
		for _, sid := range n.Incoming {
			if s, ok := b.Synapses[sid]; ok {
				if from, ok2 := b.Neurons[s.From]; ok2 {
					sums[id] += from.ActivationVal * s.Weight
				}
			}
		}
		outs[id] = n.Compute(sums[id])
	}

	// update modulators
	for _, m := range b.Modulators {
		m.Step(dt)
	}

	// apply plasticity: simple Hebbian scaled by a modulator (e.g., dopamine)
	modLevel := 1.0
	if d, ok := b.Modulators["dopamine"]; ok {
		modLevel += d.Level
	}

	for sid, s := range b.Synapses {
		if !s.Plastic {
			continue
		}
		pre := 0.0
		post := 0.0
		if from, ok := b.Neurons[s.From]; ok {
			pre = from.ActivationVal
		}
		if to, ok := b.Neurons[s.To]; ok {
			post = to.ActivationVal
		}
		// eligibility trace
		s.Trace = 0.9*s.Trace + pre*post
		delta := s.Trace * (s.Weight) * 0.1 * modLevel
		s.Weight += delta
		s.Clip()
		_ = sid
	}

	// collect outputs
	motorOutputs := make(map[string]float64)
	for name, nid := range b.Outputs {
		if v, ok := outs[nid]; ok {
			motorOutputs[name] = v
		}
	}

	// diagnostics
	diag := Diagnostics{Weights: make(map[string]float64), Mods: make(map[string]float64)}
	for id, s := range b.Synapses {
		diag.Weights[id] = s.Weight
	}
	for id, m := range b.Modulators {
		diag.Mods[id] = m.Level
	}

	return motorOutputs, diag
}

func (b *PNN) ToJSON() (string, error) {
	d, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return "", err
	}
	return string(d), nil
}
