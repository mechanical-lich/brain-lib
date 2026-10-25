package validation

import (
	"math"
	"sort"
	"strings"

	"github.com/mechanical-lich/brain-lib/pkg/brain"
)

type Complaint struct {
	ComplaintType ComplaintType
	ID            string // ID of the synapse, neuron, or lobe involved
}

// ValidateBrain reports every problem it can find, in a stable order.
//
// The result used to depend on Go's map iteration, so two runs over the same
// brain listed the same complaints in different orders — awkward to diff, to
// snapshot-test, or to show a user.
func ValidateBrain(b *brain.Brain) []Complaint {
	var complaints []Complaint

	complaints = append(complaints, validateStructure(b)...)

	// Validate synapses at the brain level
	for _, synapse := range b.Synapses {
		// A brain-level synapse naming no lobes at all cannot be routed: Step
		// skips it on FromLobe == "", so it transmits nothing. That is a
		// distinct problem from an inter-lobe synapse pointing at its own lobe,
		// and it must not be reported as one — the Inter_Synapse_To_Same_Lobe
		// repair path deletes a synapse it cannot attribute.
		if synapse.FromLobe == "" || synapse.ToLobe == "" {
			complaints = append(complaints, Complaint{
				ComplaintType: Unattributable_Synapse,
				ID:            synapse.ID,
			})
			continue
		}

		// Validate if this is actually an intralobe synapse
		if synapse.FromLobe == synapse.ToLobe {
			complaints = append(complaints, Complaint{
				ComplaintType: Inter_Synapse_To_Same_Lobe,
				ID:            synapse.ID,
			})
			continue
		}

		// Validate From exists
		if lobe, ok := b.Lobes[synapse.FromLobe]; ok {
			if _, ok := lobe.Neurons[synapse.From]; !ok {
				complaints = append(complaints, Complaint{
					ComplaintType: Missing_Neuron_From,
					ID:            synapse.ID,
				})
			}
		} else {
			complaints = append(complaints, Complaint{
				ComplaintType: Missing_From_Lobe,
				ID:            synapse.ID,
			})
		}

		// Validate To exists
		if lobe, ok := b.Lobes[synapse.ToLobe]; ok {
			if _, ok := lobe.Neurons[synapse.To]; !ok {
				complaints = append(complaints, Complaint{
					ComplaintType: Missing_Neuron_To,
					ID:            synapse.ID,
				})
			}
		} else {
			complaints = append(complaints, Complaint{
				ComplaintType: Missing_To_Lobe,
				ID:            synapse.ID,
			})
		}
	}

	// Validate Lobes
	for _, lobe := range b.Lobes {
		for _, synapse := range lobe.Synapses {
			// A synapse sitting in a lobe's own map with no lobe attribution
			// belongs to that lobe by construction — that is what the map
			// means. Reporting it as Wrong_Lobe_Synapse is what made repair
			// delete every synapse Lobe.Connect had ever created.
			if synapse.FromLobe == "" && synapse.ToLobe == "" {
				complaints = append(complaints, Complaint{
					ComplaintType: Unset_Lobe_Fields,
					ID:            synapse.ID,
				})
				continue
			}

			// Validate if this is actually an intralobe synapse
			if synapse.FromLobe != synapse.ToLobe {
				complaints = append(complaints, Complaint{
					ComplaintType: Intra_Synapse_To_Different_Lobe,
					ID:            synapse.ID,
				})
				continue
			}

			// Validate that this synapse belongs to this lobe
			if synapse.FromLobe != lobe.Name && synapse.ToLobe != lobe.Name {
				complaints = append(complaints, Complaint{
					ComplaintType: Wrong_Lobe_Synapse,
					ID:            synapse.ID,
				})
			}

			// Validate the endpoints against the lobe the synapse *names*, not
			// merely the lobe whose map it happens to sit in. A misfiled
			// synapse would otherwise also report missing neurons, and the
			// repair for that deletes — so moving it to the correct lobe was
			// immediately undone by a stale complaint from the same pass.
			fromLobe, toLobe := lobe, lobe
			if named, ok := b.Lobes[synapse.FromLobe]; ok {
				fromLobe = named
			}
			if named, ok := b.Lobes[synapse.ToLobe]; ok {
				toLobe = named
			}

			// Validate From exists
			if _, ok := fromLobe.Neurons[synapse.From]; !ok {
				complaints = append(complaints, Complaint{
					ComplaintType: Missing_Neuron_From,
					ID:            synapse.ID,
				})
			}

			// Validate To exists. A synapse may legitimately target a
			// neuroemitter, which does not live in lobe.Neurons — reporting
			// those as Missing_Neuron_To made repair delete every valid
			// neuron->emitter connection, the exact data-loss class this
			// validator was fixed to stop.
			_, toIsNeuron := toLobe.Neurons[synapse.To]
			_, toIsEmitter := toLobe.Neuroemitters[synapse.To]
			if !toIsNeuron && !toIsEmitter {
				complaints = append(complaints, Complaint{
					ComplaintType: Missing_Neuron_To,
					ID:            synapse.ID,
				})
			}

		}

	}

	sort.Slice(complaints, func(i, j int) bool {
		if complaints[i].ComplaintType != complaints[j].ComplaintType {
			return complaints[i].ComplaintType < complaints[j].ComplaintType
		}
		return complaints[i].ID < complaints[j].ID
	})
	return complaints
}

// validateStructure adds the checks that have nothing to do with synapse
// routing: receptors and emitters pointing at things that do not exist, IDs
// colliding across namespaces, and numerically invalid weights.
//
// None of these were checked before, so a receptor naming a missing modulator
// or a weight of NaN passed validation cleanly.
func validateStructure(b *brain.Brain) []Complaint {
	var complaints []Complaint
	add := func(t ComplaintType, id string) {
		complaints = append(complaints, Complaint{ComplaintType: t, ID: id})
	}

	for _, r := range b.ChemicalReceptors {
		if !brainLevelReceptorResolves(b, r) {
			add(Dangling_Receptor_Target, r.ChemicalName+"->"+r.Target)
		}
	}

	for key, lobe := range b.Lobes {
		if lobe == nil {
			continue
		}
		if lobe.Name != key {
			add(Lobe_Name_Key_Mismatch, key)
		}

		// A neuron and an emitter sharing an ID silently misroutes: Step
		// resolves synapse targets against neurons first, so the emitter never
		// receives anything.
		for id := range lobe.Neuroemitters {
			if _, clash := lobe.Neurons[id]; clash {
				add(Duplicate_Node_ID, key+"."+id)
			}
		}

		for _, r := range lobe.ChemicalReceptors {
			ok := false
			switch r.TargetType {
			case brain.TargetTypeNeuron:
				_, ok = lobe.Neurons[r.Target]
			case brain.TargetTypeModulator:
				_, ok = lobe.Modulators[r.Target]
			}
			if !ok {
				add(Dangling_Receptor_Target, key+"."+r.ChemicalName+"->"+r.Target)
			}
		}

		// A persisting neuron relaxes toward its input with its decay rate as
		// the time constant, so a rate of 0 is an infinite time constant: it
		// holds whatever value it has forever and is deaf to everything feeding
		// it. Coherent, but almost never what an author means.
		//
		// Resolved against the lobe, which supplies the default for any neuron
		// that sets no rate of its own. Checking the neuron's raw DecayRate
		// reported every neuron in a lobe that had a perfectly good rate — the
		// shipped critter brain's four emotion neurons among them.
		for id, n := range lobe.Neurons {
			if (lobe.Persists || n.Persists) && !n.Capacitor && n.EffectiveDecayRate(lobe.DecayRate) <= 0 {
				add(Frozen_Persisting_Neuron, key+"."+id)
			}
		}

		for id, syn := range lobe.Synapses {
			_, isNeuron := lobe.Neurons[syn.To]
			_, isEmitter := lobe.Neuroemitters[syn.To]
			if !isNeuron && !isEmitter {
				add(Dangling_Emitter_Synapse, id)
			}
			complaints = append(complaints, weightComplaints(syn)...)
		}
	}

	for _, syn := range b.Synapses {
		complaints = append(complaints, weightComplaints(syn)...)
	}

	return complaints
}

func brainLevelReceptorResolves(b *brain.Brain, r *brain.ChemicalReceptor) bool {
	switch r.TargetType {
	case brain.TargetTypeModulator:
		_, ok := b.Modulators[r.Target]
		return ok
	case brain.TargetTypeNeuron:
		// Either "lobe.neuron" or an unqualified, unambiguous neuron name.
		if lobeName, neuronID, ok := splitTarget(r.Target); ok {
			if lobe, found := b.Lobes[lobeName]; found {
				_, found = lobe.Neurons[neuronID]
				return found
			}
			return false
		}
		matches := 0
		for _, lobe := range b.Lobes {
			if _, ok := lobe.Neurons[r.Target]; ok {
				matches++
			}
		}
		return matches == 1
	}
	return false
}

func splitTarget(target string) (lobe, neuron string, ok bool) {
	i := strings.IndexByte(target, '.')
	if i <= 0 || i == len(target)-1 {
		return "", "", false
	}
	return target[:i], target[i+1:], true
}

func weightComplaints(syn *brain.Synapse) []Complaint {
	var out []Complaint
	for _, v := range []float64{syn.Weight, syn.InitWeight, syn.Trace} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			out = append(out, Complaint{ComplaintType: Non_Finite_Weight, ID: syn.ID})
			break
		}
	}
	// A negative per-synapse TraceDecay is silently treated as "use the
	// lobe's", which is almost certainly not what the author meant.
	if syn.TraceDecay < 0 || math.IsNaN(syn.TraceDecay) || math.IsInf(syn.TraceDecay, 0) {
		out = append(out, Complaint{ComplaintType: Invalid_Trace_Decay, ID: syn.ID})
	}
	// Only meaningful when bounds are actually set.
	if syn.MinWeight < syn.MaxWeight {
		if syn.InitWeight < syn.MinWeight || syn.InitWeight > syn.MaxWeight {
			out = append(out, Complaint{ComplaintType: Init_Weight_Out_Of_Range, ID: syn.ID})
		}
	}
	return out
}
