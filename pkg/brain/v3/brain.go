package brain

type Brain struct {
	Lobes       map[string]*Lobe    `json:"lobes"`
	updateOrder []string            `json:"-"`
	Synapses    map[string]*Synapse `json:"synapses"`
}

func NewBrain() *Brain {
	b := &Brain{
		Lobes:    make(map[string]*Lobe),
		Synapses: make(map[string]*Synapse),
	}
	return b
}

func (b *Brain) Step(dt float64) {
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
		lobe.Step(dt)
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
	b.Synapses[synapse.ID] = synapse

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
