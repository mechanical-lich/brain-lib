package validation

import (
	"github.com/mechanical-lich/brain-lib/pkg/brain/v3"
)

type Complaint struct {
	ComplaintType ComplaintType
	ID            string // ID of the synapse, neuron, or lobe involved
}

func ValidateBrain(b *brain.Brain) []Complaint {
	var complaints []Complaint

	// Validate synapses at the brain level
	for _, synapse := range b.Synapses {
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

			// Validate From exists
			if _, ok := lobe.Neurons[synapse.From]; !ok {
				complaints = append(complaints, Complaint{
					ComplaintType: Missing_Neuron_From,
					ID:            synapse.ID,
				})
			}

			// Validate To exists
			if _, ok := lobe.Neurons[synapse.To]; !ok {
				complaints = append(complaints, Complaint{
					ComplaintType: Missing_Neuron_To,
					ID:            synapse.ID,
				})
			}

		}

	}
	return complaints
}
