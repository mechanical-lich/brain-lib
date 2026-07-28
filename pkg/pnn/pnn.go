package pnn

import (
	"encoding/json"
	"math"
	"math/rand"
)

type PNN struct {
	Neurons     map[string]*Neuron    `json:"neurons"`
	updateOrder []string              `json:"-"`
	Synapses    map[string]*Synapse   `json:"synapses"`
	Inputs      map[string]string     `json:"inputs"`  // sensorName -> neuronID
	Outputs     map[string]string     `json:"outputs"` // motorName -> neuronID
	Modulators  map[string]*Modulator `json:"modulators"`
	Seed        int64                 `json:"seed"`

	// CollectDiagnostics makes Step return a full weight/modulator snapshot.
	// Off by default: Step used to build one on every call whether or not the
	// caller looked at it.
	CollectDiagnostics bool `json:"-"`

	// rng is the network's own random source. It was initialised and then never
	// used; anything stochastic should draw from it so a seeded PNN is
	// reproducible.
	rng *rand.Rand
}

// Rand returns the network's seeded random source.
func (b *PNN) Rand() *rand.Rand { return b.rng }

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

	// Propagate through synapses using the *previous* activations.
	//
	// This claimed to be a synchronous update but was not: Compute writes
	// n.ActivationVal in place while the loop is still ranging b.Neurons, so a
	// neuron visited later read its predecessors' already-updated values. Since
	// Go randomises map iteration, the same seed produced different results run
	// to run — measured at output 1.0 in 29 of 40 runs and 0.0 in the other 11.
	//
	// Summing against a snapshot, and only committing afterwards, makes the
	// update genuinely synchronous and order-independent.
	prev := make(map[string]float64, len(b.Neurons))
	for id, n := range b.Neurons {
		prev[id] = n.ActivationVal
	}

	for id, n := range b.Neurons {
		// sum incoming synapses
		for _, sid := range n.Incoming {
			if s, ok := b.Synapses[sid]; ok {
				if _, ok2 := b.Neurons[s.From]; ok2 {
					sums[id] += prev[s.From] * s.Weight
				}
			}
		}
	}

	outs := make(map[string]float64, len(b.Neurons))
	for id, n := range b.Neurons {
		outs[id] = n.Compute(sums[id])
	}

	// update modulators
	for _, m := range b.Modulators {
		m.Step(dt)
	}

	// Apply plasticity: simple Hebbian scaled by a modulator (e.g., dopamine).
	//
	// modLevel used to start at 1.0, so learning ran at full speed with no
	// reward signal at all (measured: a weight drifting 0.1 -> 0.573 in 20
	// steps of zero dopamine). It is now the modulator level itself, so no
	// signal means no learning.
	modLevel := 0.0
	if d, ok := b.Modulators["dopamine"]; ok {
		modLevel = d.Level
	}

	for sid, s := range b.Synapses {
		if !s.Plastic {
			continue
		}
		if modLevel == 0 {
			// No reward signal: let the trace fade but do not move the weight.
			s.Trace *= decayFactor(traceDecayPerSecond, dt)
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
		// Eligibility trace as a leaky integrator, scaled by dt so learning is
		// per second rather than per call.
		s.Trace = decayFactor(traceDecayPerSecond, dt)*s.Trace + pre*post*dt

		// Additive, not multiplicative. Scaling the update by s.Weight meant a
		// synapse sitting at exactly 0 could never learn anything, since every
		// delta was multiplied by zero.
		s.Weight += s.Trace * learningRatePerSecond * modLevel * dt
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

	// Diagnostics are built only when asked for. Step used to allocate the full
	// weight and modulator snapshot on every call whether or not the caller
	// looked at it.
	var diag Diagnostics
	if b.CollectDiagnostics {
		diag = Diagnostics{Weights: make(map[string]float64, len(b.Synapses)), Mods: make(map[string]float64, len(b.Modulators))}
		for id, s := range b.Synapses {
			diag.Weights[id] = s.Weight
		}
		for id, m := range b.Modulators {
			diag.Mods[id] = m.Level
		}
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

const (
	// traceDecayPerSecond is the eligibility trace's decay rate.
	traceDecayPerSecond = 1.0
	// learningRatePerSecond scales the Hebbian update.
	learningRatePerSecond = 0.1
)

// decayFactor returns the exact exponential decay multiplier for `rate` per
// second over `dt` seconds.
func decayFactor(rate, dt float64) float64 {
	if rate <= 0 || dt <= 0 {
		return 1.0
	}
	return math.Exp(-rate * dt)
}
