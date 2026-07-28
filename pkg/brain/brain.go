package brain

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/google/uuid"
)

type Brain struct {
	Lobes       map[string]*Lobe `json:"lobes"`
	updateOrder []string         `json:"-"`
	// byTargetLobe indexes inter-lobe synapses by ToLobe, rebuilt by updateLobeOrder.
	byTargetLobe      map[string][]*Synapse `json:"-"`
	Synapses          map[string]*Synapse   `json:"synapses"`
	Modulators        map[string]*Modulator `json:"modulators"`         // Brain-level modulators
	ChemicalReceptors []*ChemicalReceptor   `json:"chemical_receptors"` // Brain-level receptors
	PlasticityConfig  PlasticityConfig      `json:"plasticity_config"`  // Default plasticity config for brain
	// Tick is a simple step counter used for structural-plasticity bookkeeping
	Tick int `json:"tick"`

	// FormatVersion identifies the on-disk schema. Absent or 0 means a file
	// written before versioning existed, which is read with the compatibility
	// rules in LoadFromJson.
	FormatVersion int `json:"format_version"`
}

// CurrentFormatVersion is the schema version Save/ToJSON writes.
//
// Version history:
//
//	0 - unversioned. Rates in PlasticityConfig were per simulation step, decay
//	    used a linear approximation, and chemical receptors serialized in
//	    PascalCase while the rest of the format was snake_case.
//	1 - all rates are per second, decay is exponential, receptors and
//	    neuroemitters use snake_case keys, and a neuroemitter's pending charge
//	    persists.
//
// A version 0 file still loads: PascalCase receptor keys are accepted, and
// missing plasticity configs are defaulted. Its *rate values* are not converted,
// because there is no way to know the frame rate they were tuned at — a v0 file
// loaded as v1 will learn far more slowly than it used to. Re-tune, or migrate
// deliberately.
const CurrentFormatVersion = 1

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

	// Then apply lobe-level receptors, in the stable update order.
	//
	// Receptors *consume* from a shared pool, so whoever runs first gets the
	// chemical. Ranging b.Lobes made that a coin flip: with the shipped brain's
	// dopamine receptors in both attention and decision at a high uptake rate,
	// one lobe took nearly the whole pool and the other took the remainder —
	// and which one varied run to run. Stepping is deterministic but chemistry
	// was not, so the two disagreed.
	for _, name := range b.updateOrder {
		if lobe, ok := b.Lobes[name]; ok {
			lobe.UpdateChemicalReceptors(dt, chemicals)
		}
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

		// Inject cross-lobe inputs from source lobes that should already have
		// been stepped. The index is ordered by synapse ID, so the float
		// accumulation into ExternalInput is reproducible — ranging the map
		// directly made the same brain and inputs produce bit-different
		// activations run to run.
		for _, syn := range b.byTargetLobe[lobe.Name] {
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

			// Usage is recorded for every synapse whose source was active, not
			// only plastic ones and not only when the product was non-zero.
			// Lobe.Step already recorded all synapses; the two disagreed, and
			// testing the product meant a zero-weight synapse never counted as
			// used and decayed forever.
			if fromNeuron.ActivationVal != 0.0 {
				syn.LastUsed = b.Tick
				syn.UsageCount++
			}
		}

		// Now step this lobe once so it consumes external inputs and computes activations
		lobe.Step(dt, b.Tick, b.Modulators)

		// Update eligibility traces for plastic cross-lobe synapses targeting this lobe
		// Do this after the lobe has computed its new activations
		for _, syn := range b.byTargetLobe[lobe.Name] {
			if !syn.Plastic {
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
			// (source lobe has already been stepped, so its activations are current).
			// Leaky integrator, matching Lobe.Step: TraceDecay is a per-second
			// decay rate and the input term is scaled by dt.
			decay := decayFactor(lobe.PlasticityConfig.TraceDecay, dt)
			syn.Trace = decay*syn.Trace + fromNeuron.ActivationVal*toNeuron.ActivationVal*dt
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

		// Apply combined update, scaled by dt so learning rates are per second.
		syn.Weight += (rewardDelta + punishDelta) * dt

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
	reinforce(b.Modulators, chemicalName, amount)
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
		out, consumed := receptor.GetOutput(chemicals, dt)
		if out == 0.0 {
			continue
		}

		applied := false
		switch receptor.TargetType {
		case TargetTypeModulator:
			// Brain-level modulator
			if modulator, ok := b.Modulators[receptor.Target]; ok {
				modulator.Add(out)
				applied = true
			}
		case TargetTypeNeuron:
			// Target a neuron anywhere in the brain, addressed as
			// "lobeName.neuronID". This used to be an explicit empty case, so a
			// brain-level neuron receptor succeeded and did nothing forever.
			if lobeName, neuronID, ok := splitQualifiedNeuron(receptor.Target); ok {
				if lobe, found := b.Lobes[lobeName]; found {
					if neuron, found := lobe.Neurons[neuronID]; found {
						neuron.ExternalInput += out
						applied = true
					}
				}
			} else if name, found := b.lobeOwningNeuron(receptor.Target); found {
				// Unqualified but unambiguous.
				b.Lobes[name].Neurons[receptor.Target].ExternalInput += out
				applied = true
			}
		}

		if applied {
			chemicals[receptor.ChemicalName] -= consumed
		}
	}

}

func (b *Brain) AddSynapse(synapse *Synapse) {
	// Fill in missing lobe attribution before routing.
	//
	// A synapse with both lobe fields empty used to fall straight through to
	// b.Synapses, where Step skips it on FromLobe == "" — so it existed,
	// transmitted nothing, and reported no error. Locating each endpoint's
	// owning lobe recovers the intended wiring instead.
	if synapse.FromLobe == "" {
		if name, ok := b.lobeOwningNeuron(synapse.From); ok {
			synapse.FromLobe = name
		}
	}
	if synapse.ToLobe == "" {
		if name, ok := b.lobeOwningNeuron(synapse.To); ok {
			synapse.ToLobe = name
		}
	}

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

// lobeOwningNeuron returns the name of the single lobe containing neuronID.
//
// Reports false when no lobe has the neuron, or when more than one does: with
// an ambiguous name there is no safe inference, and guessing would silently
// wire the synapse to the wrong lobe. Validation reports those instead.
func (b *Brain) lobeOwningNeuron(neuronID string) (string, bool) {
	if neuronID == "" {
		return "", false
	}
	found := ""
	for name, lobe := range b.Lobes {
		if lobe == nil {
			continue
		}
		if _, ok := lobe.Neurons[neuronID]; !ok {
			continue
		}
		if found != "" {
			return "", false // ambiguous
		}
		found = name
	}
	return found, found != ""
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
		ID:           uuid.New().String(),
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

// updateLobeOrder recomputes the lobe update order and the inter-lobe synapse
// index. Both the seed set and the edge walk are sorted, so the order is
// reproducible — ranging the maps directly made it depend on Go's randomised
// iteration.
func (b *Brain) updateLobeOrder() {
	b.byTargetLobe = buildIncomingIndex(b.Synapses, func(s *Synapse) string { return s.ToLobe })

	outgoing := buildIncomingIndex(b.Synapses, func(s *Synapse) string { return s.FromLobe })

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
		for _, syn := range outgoing[lobeName] {
			visit(syn.ToLobe)
		}

		if _, ok := b.Lobes[lobeName]; ok {
			order = append(order, lobeName)
		}
	}

	lobeNames := make([]string, 0, len(b.Lobes))
	for name := range b.Lobes {
		lobeNames = append(lobeNames, name)
	}
	sort.Strings(lobeNames)

	for _, lobeName := range lobeNames {
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

	if brain.FormatVersion > CurrentFormatVersion {
		return nil, fmt.Errorf("brain: file format version %d is newer than this build supports (%d)",
			brain.FormatVersion, CurrentFormatVersion)
	}

	// Set default plasticity config if not loaded
	if brain.PlasticityConfig.TraceDecay == 0 {
		brain.PlasticityConfig = DefaultPlasticityConfig()
	}

	for key, lobe := range brain.Lobes {
		if lobe == nil {
			continue
		}

		// A lobe's name is the map key. The two used to be able to disagree,
		// and Step read them inconsistently — `ToLobe` was matched against
		// lobe.Name (brain.go:61) while `FromLobe` was matched against the map
		// key (:68). A file whose key and name differed gave a lobe that worked
		// as a synapse source but was silently inert as a target.
		if lobe.Name != key {
			lobe.Name = key
		}

		// Default the lobe's plasticity config too, not just the brain's.
		// Without this a lobe whose JSON omits plasticity_config loads with an
		// all-zero config, so its intra-lobe learning is silently off while
		// inter-lobe learning still runs via the brain-level fallback.
		if lobe.PlasticityConfig.TraceDecay == 0 {
			lobe.PlasticityConfig = brain.PlasticityConfig
		}

		for _, syn := range lobe.Synapses {
			normalizeSynapse(syn)
		}
	}

	for _, syn := range brain.Synapses {
		normalizeSynapse(syn)
	}

	brain.updateLobeOrder()
	for _, lobe := range brain.Lobes {
		if lobe == nil {
			continue // a JSON null lobe would panic here
		}
		lobe.updateNeuronOrder()
	}

	// Anything we have just normalized is, by definition, current.
	brain.FormatVersion = CurrentFormatVersion

	return &brain, nil
}

// normalizeSynapse repairs a synapse whose baseline sits outside its own bounds.
//
// applyInactivityDecay moves an unused weight toward InitWeight. If InitWeight
// is outside [MinWeight, MaxWeight] that pull points away from the legal range,
// so idling *strengthens* the synapse until it pins against the clip —
// InitWeight 50 with a [-2,2] clip drove an idle weight to +2. Lobe.Connect
// produces exactly this whenever it is handed a weight beyond the default
// bounds, since it copies weight into InitWeight unchecked.
func normalizeSynapse(syn *Synapse) {
	if syn == nil || syn.MinWeight >= syn.MaxWeight {
		return // bounds unset or inverted; nothing meaningful to clamp against
	}
	syn.InitWeight = clampFloat(syn.InitWeight, syn.MinWeight, syn.MaxWeight)
	syn.Weight = clampFloat(syn.Weight, syn.MinWeight, syn.MaxWeight)
}

func clampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func LoadBrain(filePath string) (*Brain, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	return LoadFromJson(data)
}

func (b *Brain) Save(filePath string) error {
	b.FormatVersion = CurrentFormatVersion

	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0644)
}

func (b *Brain) ToJSON() ([]byte, error) {
	b.FormatVersion = CurrentFormatVersion

	return json.MarshalIndent(b, "", "  ")
}

func (b *Brain) RefreshUpdateOrder() {
	b.updateLobeOrder()
	for _, lobe := range b.Lobes {
		lobe.updateNeuronOrder()
	}
}

// splitQualifiedNeuron splits a "lobeName.neuronID" target.
func splitQualifiedNeuron(target string) (lobe, neuron string, ok bool) {
	i := strings.IndexByte(target, '.')
	if i <= 0 || i == len(target)-1 {
		return "", "", false
	}
	return target[:i], target[i+1:], true
}
