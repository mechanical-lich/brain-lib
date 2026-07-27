package validation

import (
	"github.com/mechanical-lich/brain-lib/pkg/brain"
)

type Complaint struct {
	ComplaintType ComplaintType
	ID            string // ID of the synapse, neuron, or lobe involved
}

func ValidateBrain(b *brain.Brain) []Complaint {
	var complaints []Complaint

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

			// Validate To exists
			if _, ok := toLobe.Neurons[synapse.To]; !ok {
				complaints = append(complaints, Complaint{
					ComplaintType: Missing_Neuron_To,
					ID:            synapse.ID,
				})
			}

		}

	}
	return complaints
}
