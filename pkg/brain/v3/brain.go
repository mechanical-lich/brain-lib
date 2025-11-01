package brain

import (
	"encoding/json"
	"os"
)

type Brain struct {
	Lobes             map[string]*Lobe      `json:"lobes"`
	updateOrder       []string              `json:"-"`
	Synapses          map[string]*Synapse   `json:"synapses"`
	Modulators        map[string]*Modulator `json:"modulators"`         // Brain-level modulators
	ChemicalReceptors []*ChemicalReceptor   `json:"chemical_receptors"` // Brain-level receptors
	PlasticityConfig  PlasticityConfig      `json:"plasticity_config"`  // Default plasticity config for brain
	// Tick is a simple step counter used for structural-plasticity bookkeeping
	Tick int `json:"tick"`
}

func NewBrain() *Brain {
	b := &Brain{
		Lobes:            make(map[string]*Lobe),
		Synapses:         make(map[string]*Synapse),
		Modulators:       make(map[string]*Modulator),
		PlasticityConfig: DefaultPlasticityConfig(),
	}

	return b
}

func (b *Brain) ApplyChemistry(chemicals map[string]float64, dt float64) {
	// Apply brain-level chemical receptors first
	b.UpdateChemicalReceptors(dt, chemicals)

	// Then apply lobe-level receptors
	for _, lobe := range b.Lobes {
		lobe.UpdateChemicalReceptors(dt, chemicals)
	}
}

func (b *Brain) Step(dt float64) {
	// advance global tick once per simulation step
	b.Tick++

	// Update brain-level modulators (decay their levels)
	for _, m := range b.Modulators {
		m.Step(dt)
	}

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

			// Apply synaptic transmission
			contribution := fromNeuron.ActivationVal * syn.Weight
			toNeuron.ExternalInput += contribution

			// Mark synapse as used if it contributed (for plasticity tracking)
			if contribution != 0.0 && syn.Plastic {
				syn.LastUsed = b.Tick
				syn.UsageCount++
			}
		}

		// Now step this lobe once so it consumes external inputs and computes activations
		lobe.Step(dt, b.Tick)

		// Update eligibility traces for plastic cross-lobe synapses targeting this lobe
		// Do this after the lobe has computed its new activations
		for _, syn := range b.Synapses {
			if syn.ToLobe != lobe.Name || !syn.Plastic {
				continue
			}
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

			// Update eligibility trace using target lobe's plasticity config
			// (source lobe has already been stepped, so its activations are current)
			syn.Trace = lobe.PlasticityConfig.TraceDecay*syn.Trace + fromNeuron.ActivationVal*toNeuron.ActivationVal
		}
	}

	// After all lobes have stepped, apply plasticity to inter-lobe synapses
	b.applyInterLobePlasticity(dt)
}

// applyInterLobePlasticity applies reinforcement learning and inactivity decay to inter-lobe synapses
func (b *Brain) applyInterLobePlasticity(dt float64) {
	for _, syn := range b.Synapses {
		if !syn.Plastic {
			continue
		}

		// Get the target lobe's plasticity config and modulators
		toLobe, ok := b.Lobes[syn.ToLobe]
		if !ok {
			continue
		}

		// Get modulator levels - check lobe first, then brain level as fallback
		dopamineLevel := 0.0
		serotoninLevel := 0.0

		// Try lobe-level modulators first
		if d, ok := toLobe.Modulators[Dopamine]; ok {
			dopamineLevel = d.Level
		} else if d, ok := b.Modulators[Dopamine]; ok {
			// Fall back to brain-level modulator
			dopamineLevel = d.Level
		}

		if s, ok := toLobe.Modulators[Serotonin]; ok {
			serotoninLevel = s.Level
		} else if s, ok := b.Modulators[Serotonin]; ok {
			// Fall back to brain-level modulator
			serotoninLevel = s.Level
		}

		// Get plasticity config - use lobe's config, or brain's as fallback
		plasticityConfig := toLobe.PlasticityConfig
		if plasticityConfig.TraceDecay == 0 {
			// Lobe doesn't have config set, use brain's
			plasticityConfig = b.PlasticityConfig
		}

		// Reward strengthens active pathways
		rewardDelta := syn.Trace * plasticityConfig.RewardLearningRate * dopamineLevel

		// Punishment weakens active pathways
		punishDelta := -syn.Trace * plasticityConfig.PunishLearningRate * serotoninLevel

		// Apply combined update
		syn.Weight += rewardDelta + punishDelta

		// Apply inactivity decay toward baseline
		inactivitySteps := b.Tick - syn.LastUsed
		if inactivitySteps > 0 {
			decayRate := plasticityConfig.InactivityDecay * dt
			delta := syn.InitWeight - syn.Weight
			syn.Weight += delta * decayRate
		}

		syn.Clip()
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

// Reinforce applies a reinforcement signal to a brain-level modulator.
// This affects all inter-lobe synapses and any lobes that don't have their own modulators.
// Use this for global learning signals that should affect the entire brain.
//
// Example usage:
//
//	brain.Reinforce(Dopamine, 1.0)  // Global reward signal
//	brain.Reinforce(Serotonin, 0.5) // Global punishment signal
func (b *Brain) Reinforce(chemicalName string, amount float64) {
	if b.Modulators == nil {
		b.Modulators = make(map[string]*Modulator)
	}

	m, ok := b.Modulators[chemicalName]
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
		b.Modulators[chemicalName] = m
	}

	m.Level += amount
}

// AddReceptor adds a chemical receptor at the brain level that can affect any lobe or neuron
func (b *Brain) AddReceptor(chemicalName string, uptakeRate float64, target string, targetType TargetType, receptorType ReceptorType) {
	receptor := &ChemicalReceptor{
		ChemicalName: chemicalName,
		UptakeRate:   uptakeRate,
		Target:       target,
		TargetType:   targetType,
		ReceptorType: receptorType,
	}
	b.ChemicalReceptors = append(b.ChemicalReceptors, receptor)
}

// UpdateChemicalReceptors applies brain-level chemical receptors
// This can be used to have chemicals affect brain-level modulators or specific neurons across lobes
func (b *Brain) UpdateChemicalReceptors(dt float64, chemicals map[string]float64) {
	for _, receptor := range b.ChemicalReceptors {
		conc, ok := chemicals[receptor.ChemicalName]
		if !ok || conc <= 0.0 {
			continue
		}
		amount := conc * receptor.UptakeRate * dt

		switch receptor.TargetType {
		case TargetTypeModulator:
			// Brain-level modulator
			if modulator, ok := b.Modulators[receptor.Target]; ok {
				if receptor.ReceptorType == ReceptorTypeInhibitory {
					modulator.Level -= amount
				} else {
					modulator.Level += amount
				}
				chemicals[receptor.ChemicalName] -= amount
			}
		case TargetTypeNeuron:
			// Could target neurons in specific lobes by parsing target as "lobeName.neuronID"
			// For now, skip neuron targeting at brain level (handled by lobes)
		}
	}

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
		ID:           fromLobe + "_" + fromNeuron + "_to_" + toLobe + "_" + toNeuron,
		From:         fromNeuron,
		To:           toNeuron,
		FromLobe:     fromLobe,
		ToLobe:       toLobe,
		Weight:       weight,
		Plastic:      plastic,
		MinWeight:    -2.0,
		MaxWeight:    2.0,
		InitWeight:   weight, // Store baseline weight for decay
		MinDecayFrac: 0.1,    // Kept for backward compatibility
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
	if brain.Modulators == nil {
		brain.Modulators = make(map[string]*Modulator)
	}

	// Set default plasticity config if not loaded
	if brain.PlasticityConfig.TraceDecay == 0 {
		brain.PlasticityConfig = DefaultPlasticityConfig()
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
