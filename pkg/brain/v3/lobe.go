package brain

type Lobe struct {
	Neurons     map[string]*Neuron  `json:"neurons"`
	updateOrder []string            `json:"-"`
	Synapses    map[string]*Synapse `json:"synapses"`

	// Lobe level settings
	Name              string                `json:"name"`
	Modulators        map[string]*Modulator `json:"modulators"`
	Persists          bool                  `json:"persists"`   // Neurons retain activation between steps if true
	DecayRate         float64               `json:"decay_rate"` // global decay rate applied to all neurons each step
	ChemicalReceptors []*ChemicalReceptor   `json:"chemical_receptors"`
	// Optional visual position for editor/UI
	X float64 `json:"x,omitempty"`
	Y float64 `json:"y,omitempty"`
}

func NewLobe(name string, modulators map[string]*Modulator) *Lobe {
	if modulators == nil {
		modulators = make(map[string]*Modulator)
	}

	return &Lobe{
		Name:       name,
		Modulators: modulators,
	}
}

func (l *Lobe) AddNeuron(neuron *Neuron) {
	if l.Neurons == nil {
		l.Neurons = make(map[string]*Neuron)
	}
	l.Neurons[neuron.ID] = neuron

	l.updateNeuronOrder()
}

func (l *Lobe) AddSynapse(synapse *Synapse) {
	if l.Synapses == nil {
		l.Synapses = make(map[string]*Synapse)
	}
	l.Synapses[synapse.ID] = synapse

	l.updateNeuronOrder()
}

func (l *Lobe) GetNeuron(id string) *Neuron {
	if neuron, ok := l.Neurons[id]; ok {
		return neuron
	}
	return nil
}

func (l *Lobe) GetSynapse(id string) *Synapse {
	if synapse, ok := l.Synapses[id]; ok {
		return synapse
	}
	return nil
}

func (l *Lobe) RemoveNeuron(id string) {
	delete(l.Neurons, id)

	// also remove any synapses connected to this neuron
	for sid, s := range l.Synapses {
		if s.From == id || s.To == id {
			delete(l.Synapses, sid)
		}
	}

	l.updateNeuronOrder()
}

func (l *Lobe) RemoveSynapse(id string) {
	delete(l.Synapses, id)

	l.updateNeuronOrder()
}

func (l *Lobe) SetNeuronInput(id string, value float64) {
	if neuron, ok := l.Neurons[id]; ok {
		neuron.ExternalInput = value
	}
}

func (l *Lobe) GetNeuronOutput(id string) float64 {
	if neuron, ok := l.Neurons[id]; ok {
		return neuron.ActivationVal
	}
	return 0.0
}

func (l *Lobe) Connect(fromID, toID string, weight float64, plastic bool) *Synapse {
	synapse := &Synapse{
		ID:        fromID + "->" + toID,
		From:      fromID,
		To:        toID,
		Weight:    weight,
		Plastic:   plastic,
		MinWeight: -1.0,
		MaxWeight: 1.0,
	}
	l.AddSynapse(synapse)
	return synapse
}

func (l *Lobe) AddReceptor(chemicalName string, uptakeRate float64, target string, targetType TargetType, receptorType ReceptorType) {
	receptor := &ChemicalReceptor{
		ChemicalName: chemicalName,
		UptakeRate:   uptakeRate,
		Target:       target,
		TargetType:   targetType,
		ReceptorType: receptorType,
	}
	l.ChemicalReceptors = append(l.ChemicalReceptors, receptor)
}

func (l *Lobe) ApplyChemical(chemicalName string, concentration float64) float64 {
	for _, receptor := range l.ChemicalReceptors {
		if receptor.ChemicalName == chemicalName {
			amount := concentration * receptor.UptakeRate
			switch receptor.TargetType {
			case TargetTypeNeuron:
				if neuron, ok := l.Neurons[receptor.Target]; ok {
					if receptor.ReceptorType == ReceptorTypeInhibitory {
						neuron.ExternalInput -= amount
					} else {
						neuron.ExternalInput += amount
					}
					return concentration - amount
				}
			case TargetTypeModulator:
				if modulator, ok := l.Modulators[receptor.Target]; ok {
					if receptor.ReceptorType == ReceptorTypeInhibitory {
						modulator.Level -= amount
					} else {
						modulator.Level += amount
					}
					return concentration - amount
				}
			}
		}
	}
	return concentration
}

// Applies all the chemicals in the map that it can and returns an updated map with the remaining concentrations
func (l *Lobe) UpdateChemicalReceptors(dt float64, chemicals map[string]float64) map[string]float64 {
	// Apply chemical receptors
	for _, receptor := range l.ChemicalReceptors {
		conc, ok := chemicals[receptor.ChemicalName]
		if !ok || conc <= 0.0 {
			continue
		}
		amount := conc * receptor.UptakeRate * dt

		switch receptor.TargetType {
		case TargetTypeNeuron:
			if neuron, ok := l.Neurons[receptor.Target]; ok {
				if receptor.ReceptorType == ReceptorTypeInhibitory {
					neuron.ExternalInput -= amount
				} else {
					neuron.ExternalInput += amount
				}
				chemicals[receptor.ChemicalName] -= amount
			}
		case TargetTypeModulator:
			if modulator, ok := l.Modulators[receptor.Target]; ok {
				if receptor.ReceptorType == ReceptorTypeInhibitory {
					modulator.Level -= amount
				} else {
					modulator.Level += amount
				}
				chemicals[receptor.ChemicalName] -= amount
			}
		}
	}

	return chemicals
}

func (l *Lobe) Step(dt float64) {
	// Update modulators
	for _, m := range l.Modulators {
		m.Step(dt)
	}

	// Update neurons in topological order computed by updateNeuronOrder
	for _, nid := range l.updateOrder {
		neuron, ok := l.Neurons[nid]
		if !ok {
			continue
		}

		// Wipe the neuron if the lobe doesn't persist between steps
		if !l.Persists && !neuron.Persists {
			neuron.ActivationVal = 0.0
		}

		// Sum inputs from all synapses that target this neuron
		sum := 0.0
		for _, s := range l.Synapses {
			if s.To != nid {
				continue
			}
			if pre, ok := l.Neurons[s.From]; ok {
				sum += pre.ActivationVal * s.Weight
			}
		}
		// Compute neuron's new activation value (includes bias and activation func)
		neuron.Compute(sum)

		// Lobe level decay if configured
		neuron.ActivationVal -= neuron.ActivationVal * l.DecayRate
	}

	// apply plasticity: simple Hebbian scaled by a modulator (e.g., dopamine)
	modLevel := 1.0
	if d, ok := l.Modulators[Dopamine]; ok {
		modLevel += d.Level
	}

	for sid, s := range l.Synapses {
		if !s.Plastic {
			continue
		}
		pre := 0.0
		post := 0.0
		if from, ok := l.Neurons[s.From]; ok {
			pre = from.ActivationVal
		}
		if to, ok := l.Neurons[s.To]; ok {
			post = to.ActivationVal
		}
		// eligibility trace
		s.Trace = 0.9*s.Trace + pre*post
		delta := s.Trace * (s.Weight) * 0.1 * modLevel
		s.Weight += delta
		s.Clip()
		_ = sid
	}
}

func (l *Lobe) updateNeuronOrder() {
	l.updateOrder = make([]string, 0, len(l.Neurons))
	visited := make(map[string]bool)
	var visit func(nid string)
	visit = func(nid string) {
		if visited[nid] {
			return
		}
		visited[nid] = true
		// visit all neurons that feed into this one
		for _, s := range l.Synapses {
			if s.To == nid {
				visit(s.From)
			}
		}
		l.updateOrder = append(l.updateOrder, nid)
	}
	for nid := range l.Neurons {
		visit(nid)
	}
}
