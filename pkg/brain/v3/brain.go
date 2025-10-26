package brain

import (
	"encoding/json"
	"os"
)

type Brain struct {
	Lobes       map[string]*Lobe    `json:"lobes"`
	updateOrder []string            `json:"-"`
	Synapses    map[string]*Synapse `json:"synapses"`
	// Tick is a simple step counter used for structural-plasticity bookkeeping
	Tick int `json:"tick"`
}

func NewBrain() *Brain {
	b := &Brain{
		Lobes:    make(map[string]*Lobe),
		Synapses: make(map[string]*Synapse),
	}

	return b
}

func (b *Brain) ApplyChemistry(chemicals map[string]float64, dt float64) map[string]float64 {
	for _, lobe := range b.Lobes {
		chemicals = lobe.UpdateChemicalReceptors(dt, chemicals)
	}
	return chemicals
}

func (b *Brain) Step(dt float64) {
	// advance global tick once per simulation step
	b.Tick++
	// Single-pass: for each lobe in topological order, inject external inputs from
	// already-stepped source lobes (via cross-lobe synapses) then step the lobe once.
	for _, lobeId := range b.updateOrder {
		lobe, ok := b.Lobes[lobeId]
		if !ok {
			continue
		}

		// Inject cross-lobe inputs from source lobes that should already have been stepped
		for _, syn := range b.Synapses {
			if syn.ToLobe != lobe.Name {
				continue
			}
			// skip if synapse doesn't reference a source lobe (shouldn't happen)
			if syn.FromLobe == "" {
				continue
			}
			fromLobe, ok1 := b.Lobes[syn.FromLobe]
			if !ok1 {
				continue
			}
			fromNeuron, ok2 := fromLobe.Neurons[syn.From]
			toNeuron, ok3 := lobe.Neurons[syn.To]
			if !ok2 || !ok3 {
				continue
			}
			toNeuron.ExternalInput += fromNeuron.ActivationVal * syn.Weight
		}

		// Now step this lobe once so it consumes external inputs and computes activations
		lobe.Step(dt, b.Tick)
	}
}

func (b *Brain) AddLobe(lobe *Lobe) {
	b.Lobes[lobe.Name] = lobe
	b.updateLobeOrder()
}

func (b *Brain) GetLobe(name string) *Lobe {
	if lobe, ok := b.Lobes[name]; ok {
		return lobe
	}
	return nil
}

func (b *Brain) AddSynapse(synapse *Synapse) {
	// If this is an intra-lobe synapse, also add it to the lobe's synapse map
	if synapse.FromLobe != "" && synapse.FromLobe == synapse.ToLobe {
		if l, ok := b.Lobes[synapse.FromLobe]; ok {
			l.AddSynapse(synapse)
		}
	} else {
		b.Synapses[synapse.ID] = synapse
	}
	b.updateLobeOrder()
}

func (b *Brain) GetSynapse(id string) *Synapse {
	if synapse, ok := b.Synapses[id]; ok {
		return synapse
	}
	return nil
}

func (b *Brain) RemoveSynapse(id string) {
	delete(b.Synapses, id)
	// Also remove from lobe if it's an intra-lobe synapse
	for _, lobe := range b.Lobes {
		if _, ok := lobe.Synapses[id]; ok {
			lobe.RemoveSynapse(id)
		}
	}
	b.updateLobeOrder()
}

func (b *Brain) Connect(fromLobe, fromNeuron, toLobe, toNeuron string, weight float64, plastic bool) *Synapse {
	synapse := &Synapse{
		ID:        fromLobe + "_" + fromNeuron + "_to_" + toLobe + "_" + toNeuron,
		From:      fromNeuron,
		To:        toNeuron,
		FromLobe:  fromLobe,
		ToLobe:    toLobe,
		Weight:    weight,
		Plastic:   plastic,
		MinWeight: -1.0,
		MaxWeight: 1.0,
	}
	b.AddSynapse(synapse)

	return synapse
}

func (b *Brain) updateLobeOrder() {
	// simple topological sort based on synapse connections
	visited := make(map[string]bool)
	order := []string{}

	var visit func(string)
	visit = func(lobeName string) {
		if visited[lobeName] {
			return
		}
		visited[lobeName] = true

		// visit all lobes that this lobe connects to
		for _, syn := range b.Synapses {
			if syn.FromLobe == lobeName {
				visit(syn.ToLobe)
			}
		}

		order = append(order, lobeName)
	}

	for lobeName := range b.Lobes {
		visit(lobeName)
	}

	// reverse order so that sources (lobes with no incoming edges) come before targets
	for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}
	b.updateOrder = order
}

func LoadFromJson(data []byte) (*Brain, error) {
	var brain Brain
	if err := json.Unmarshal(data, &brain); err != nil {
		return nil, err
	}

	// Initialize maps if nil
	if brain.Lobes == nil {
		brain.Lobes = make(map[string]*Lobe)
	}
	if brain.Synapses == nil {
		brain.Synapses = make(map[string]*Synapse)
	}

	brain.updateLobeOrder()
	for _, lobe := range brain.Lobes {
		lobe.updateNeuronOrder()
	}

	return &brain, nil
}

func LoadBrain(filePath string) (*Brain, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	return LoadFromJson(data)
}

func (b *Brain) Save(filePath string) error {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0644)
}

func (b *Brain) ToJSON() ([]byte, error) {
	return json.MarshalIndent(b, "", "  ")
}
