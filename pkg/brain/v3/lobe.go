package brain

// PlasticityConfig controls learning parameters for a lobe
type PlasticityConfig struct {
	TraceDecay         float64 `json:"trace_decay"`          // How fast eligibility traces fade (default 0.9)
	RewardLearningRate float64 `json:"reward_learning_rate"` // Learning rate for rewards (default 0.1)
	PunishLearningRate float64 `json:"punish_learning_rate"` // Learning rate for punishment (default 0.15)
	InactivityDecay    float64 `json:"inactivity_decay"`     // Rate weights return to baseline when unused (default 0.00001)
}

// DefaultPlasticityConfig returns sensible defaults for creature learning
func DefaultPlasticityConfig() PlasticityConfig {
	return PlasticityConfig{
		TraceDecay:         0.90,
		RewardLearningRate: 0.10,
		PunishLearningRate: 0.15,
		InactivityDecay:    0.00001,
	}
}

type Lobe struct {
	Neurons     map[string]*Neuron  `json:"neurons"`
	updateOrder []string            `json:"-"`
	Synapses    map[string]*Synapse `json:"synapses"`

	// Lobe level settings
	Name              string                   `json:"name"`
	Modulators        map[string]*Modulator    `json:"modulators"`
	Persists          bool                     `json:"persists"`   // Neurons retain activation between steps if true
	DecayRate         float64                  `json:"decay_rate"` // global decay rate applied to all neurons each step
	ChemicalReceptors []*ChemicalReceptor      `json:"chemical_receptors"`
	Neuroemitters     map[string]*Neuroemitter `json:"neuroemitters"`     // Chemical emitters driven by neural activity
	PlasticityConfig  PlasticityConfig         `json:"plasticity_config"` // Learning parameters
	// Optional visual position for editor/UI
	X float64 `json:"x,omitempty"`
	Y float64 `json:"y,omitempty"`

	WinnerTakesAll bool `json:"winner_takes_all,omitempty"` // If true only the most active neuron retains its activation each step
}

func NewLobe(name string, modulators map[string]*Modulator) *Lobe {
	if modulators == nil {
		modulators = make(map[string]*Modulator)
	}

	return &Lobe{
		Name:             name,
		Modulators:       modulators,
		PlasticityConfig: DefaultPlasticityConfig(),
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
		ID:           fromID + "->" + toID,
		From:         fromID,
		To:           toID,
		Weight:       weight,
		Plastic:      plastic,
		MinWeight:    -2.0,
		MaxWeight:    2.0,
		InitWeight:   weight, // Store baseline weight for decay
		MinDecayFrac: 0.1,    // Kept for backward compatibility
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

// AddEmitter adds a neuroemitter to the lobe
func (l *Lobe) AddEmitter(id, chemicalName string, emissionRate, threshold float64) {
	if l.Neuroemitters == nil {
		l.Neuroemitters = make(map[string]*Neuroemitter)
	}
	emitter := NewNeuroemitter(id, chemicalName, emissionRate, threshold)
	l.Neuroemitters[id] = emitter
}

// RemoveEmitter removes a neuroemitter from the lobe
func (l *Lobe) RemoveEmitter(id string) {
	if l.Neuroemitters != nil {
		delete(l.Neuroemitters, id)
	}
	// Also remove any synapses connected to this emitter
	for sid, s := range l.Synapses {
		if s.To == id {
			delete(l.Synapses, sid)
		}
	}
}

// GetEmitter returns a neuroemitter by ID
func (l *Lobe) GetEmitter(id string) *Neuroemitter {
	if l.Neuroemitters != nil {
		return l.Neuroemitters[id]
	}
	return nil
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
		// GetOutput will return the amount to apply to the target and will
		// subtract the consumed chemical from the chemicals map.
		out, consumed := receptor.GetOutput(chemicals, dt)
		if out == 0.0 {
			continue
		}

		applied := false
		switch receptor.TargetType {
		case TargetTypeNeuron:
			if neuron, ok := l.Neurons[receptor.Target]; ok {
				neuron.ExternalInput += out
				applied = true
			}
		case TargetTypeModulator:
			if modulator, ok := l.Modulators[receptor.Target]; ok {
				modulator.Level += out
				applied = true
			}
		}

		// Only subtract consumed chemical if the receptor was applied to a real target
		if applied {
			chemicals[receptor.ChemicalName] -= consumed
		}
	}

	return chemicals
}

// UpdateNeuroemitters processes all neuroemitters and adds their emissions to the chemistry map
func (l *Lobe) UpdateNeuroemitters(dt float64, chemicals map[string]float64) map[string]float64 {
	if l.Neuroemitters == nil {
		return chemicals
	}

	for _, emitter := range l.Neuroemitters {
		emission := emitter.Step(dt)
		if emission > 0.0 {
			if chemicals == nil {
				chemicals = make(map[string]float64)
			}
			chemicals[emitter.ChemicalName] += emission
		}
	}

	return chemicals
}

func (l *Lobe) Step(dt float64, now int) {
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
				mult := pre.ActivationVal * s.Weight
				sum += mult
				// mark synapse as used if presynaptic activity contributed
				if mult != 0.0 {
					s.LastUsed = now
					s.UsageCount++
				}
			}
		}

		// Compute neuron's new activation value (includes bias and activation func)
		neuron.Compute(sum)

		// Lobe level decay if configured
		neuron.ActivationVal -= neuron.ActivationVal * l.DecayRate

		// Update eligibility traces for plastic synapses targeting this neuron
		// Use presynaptic activation (already updated in topological order) and
		// postsynaptic activation we just computed
		for _, s := range l.Synapses {
			if s.To != nid || !s.Plastic {
				continue
			}
			if pre, ok := l.Neurons[s.From]; ok {
				// Decaying eligibility trace captures pre-post correlation
				s.Trace = l.PlasticityConfig.TraceDecay*s.Trace + pre.ActivationVal*neuron.ActivationVal
			}
		}
	}

	// Send activation from neurons to neuroemitters via synapses
	// This happens after all neurons have been updated
	if l.Neuroemitters != nil {
		for _, s := range l.Synapses {
			if emitter := l.GetEmitter(s.To); emitter != nil {
				if pre, ok := l.Neurons[s.From]; ok {
					emitter.ReceiveInput(pre.ActivationVal * s.Weight)
				}
			}
		}
	}

	// Apply multi-modal reinforcement learning
	l.applyReinforcementLearning()

	// Apply inactivity-based synaptic decay
	l.applyInactivityDecay(dt, now)

	if l.WinnerTakesAll {
		var maxNeuron *Neuron
		for _, neuron := range l.Neurons {
			if maxNeuron == nil || neuron.ActivationVal > maxNeuron.ActivationVal {
				maxNeuron = neuron
			}
		}

		for _, neuron := range l.Neurons {
			if neuron != maxNeuron {
				neuron.ActivationVal = 0.0
			}
		}
	}
}

// applyReinforcementLearning applies reward and punishment signals to plastic synapses
func (l *Lobe) applyReinforcementLearning() {
	// Get current chemical levels
	dopamineLevel := 0.0
	serotoninLevel := 0.0

	if d, ok := l.Modulators[Dopamine]; ok {
		dopamineLevel = d.Level
	}
	if s, ok := l.Modulators[Serotonin]; ok {
		serotoninLevel = s.Level
	}

	// Apply reinforcement to each plastic synapse
	for _, s := range l.Synapses {
		if !s.Plastic {
			continue
		}

		// Reward strengthens active pathways (positive reinforcement)
		rewardDelta := s.Trace * l.PlasticityConfig.RewardLearningRate * dopamineLevel

		// Punishment weakens active pathways (negative reinforcement)
		punishDelta := -s.Trace * l.PlasticityConfig.PunishLearningRate * serotoninLevel

		// Apply combined update
		s.Weight += rewardDelta + punishDelta

		s.Clip()
	}
}

// applyInactivityDecay implements synaptic homeostasis - unused synapses decay toward baseline
func (l *Lobe) applyInactivityDecay(dt float64, now int) {
	for _, s := range l.Synapses {
		if !s.Plastic {
			continue
		}

		inactivitySteps := now - s.LastUsed
		if inactivitySteps > 0 {
			// Decay toward initial weight (baseline), not toward zero
			// This preserves learned behaviors while allowing unused modifications to fade
			decayRate := l.PlasticityConfig.InactivityDecay * dt

			// Move weight toward InitWeight by decay rate
			delta := s.InitWeight - s.Weight
			s.Weight += delta * decayRate

			s.Clip()
		}
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

// Reinforce applies a reinforcement signal to a specified chemical modulator.
// Use this to trigger learning from game events:
//   - Positive amounts for rewards (use Dopamine)
//   - Negative amounts or positive punishment signals (use Serotonin)
//
// Example usage:
//
//	critter.Brain.GetLobe("decision").Reinforce(Dopamine, 1.0)  // Reward
//	critter.Brain.GetLobe("decision").Reinforce(Serotonin, 0.5) // Punishment
func (l *Lobe) Reinforce(chemicalName string, amount float64) {
	if l.Modulators == nil {
		l.Modulators = make(map[string]*Modulator)
	}

	m, ok := l.Modulators[chemicalName]
	if !ok {
		// Create modulator if it doesn't exist with sensible defaults
		decayRate := 0.5 // Medium decay by default
		if chemicalName == Serotonin {
			decayRate = 0.3 // Pain/punishment lingers longer
		} else if chemicalName == Endorphin {
			decayRate = 0.8 // Satisfaction fades quickly
		}

		m = &Modulator{
			Name:      chemicalName,
			Level:     0.0,
			DecayRate: decayRate,
		}
		l.Modulators[chemicalName] = m
	}

	m.Level += amount
}
