package validation

import (
	"fmt"

	"github.com/mechanical-lich/brain-lib/pkg/brain"
)

func FixSynapses(b *brain.Brain, complaints []Complaint) error {
	if b.Synapses == nil {
		b.Synapses = make(map[string]*brain.Synapse)
	}

	for _, comp := range complaints {
		id := comp.ID
		switch comp.ComplaintType {
		case Inter_Synapse_To_Same_Lobe:
			if syn, ok := b.Synapses[id]; ok {
				if syn.FromLobe == "" {
					fmt.Printf("Deleted synapse %s: missing FromLobe for intra-lobe move\n", id)
					delete(b.Synapses, id)
					continue
				}
				l, ok := b.Lobes[syn.FromLobe]
				if !ok || l.Neurons == nil {
					fmt.Printf("Deleted synapse %s: lobes missing (%s)\n", id, syn.FromLobe)
					delete(b.Synapses, id)
					continue
				}
				if _, ok := l.Neurons[syn.From]; !ok {
					fmt.Printf("Deleted synapse %s: missing neuron %s in lobe %s\n", id, syn.From, syn.FromLobe)
					delete(b.Synapses, id)
					continue
				}
				if _, ok := l.Neurons[syn.To]; !ok {
					fmt.Printf("Deleted synapse %s: missing neuron %s in lobe %s\n", id, syn.To, syn.ToLobe)
					delete(b.Synapses, id)
					continue
				}
				l.AddSynapse(syn)
				delete(b.Synapses, id)
				fmt.Printf("Moved synapse %s into lobe %s (intra-lobe)\n", id, syn.FromLobe)
			}

		case Missing_From_Lobe, Missing_To_Lobe, Missing_Neuron_From, Missing_Neuron_To:
			if _, ok := b.Synapses[id]; ok {
				fmt.Printf("Deleted synapse %s: %s\n", id, comp.ComplaintType)
				delete(b.Synapses, id)
				continue
			}
			for _, l := range b.Lobes {
				if _, ok := l.Synapses[id]; ok {
					fmt.Printf("Deleted synapse %s in lobe %s: %s\n", id, l.Name, comp.ComplaintType)
					delete(l.Synapses, id)
					break
				}
			}

		case Intra_Synapse_To_Different_Lobe:
			for _, l := range b.Lobes {
				if syn, ok := l.Synapses[id]; ok {
					if syn.FromLobe == "" {
						syn.FromLobe = l.Name
						fmt.Printf("Set FromLobe for synapse %s to %s\n", id, l.Name)
					}
					if syn.ToLobe == "" {
						found := ""
						for lname, ol := range b.Lobes {
							if ol.Neurons != nil {
								if _, ok := ol.Neurons[syn.To]; ok {
									found = lname
									break
								}
							}
						}
						if found == "" {
							fmt.Printf("Deleted synapse %s in lobe %s: could not infer ToLobe for neuron %s\n", id, l.Name, syn.To)
							delete(l.Synapses, id)
							break
						}
						syn.ToLobe = found
						fmt.Printf("Inferred ToLobe for synapse %s: to=%s\n", id, found)
					}

					fl, ok1 := b.Lobes[syn.FromLobe]
					tl, ok2 := b.Lobes[syn.ToLobe]
					if !ok1 || !ok2 || fl.Neurons == nil || tl.Neurons == nil {
						fmt.Printf("Deleted synapse %s in lobe %s: missing lobe or neurons (%s->%s)\n", id, l.Name, syn.FromLobe, syn.ToLobe)
						delete(l.Synapses, id)
						break
					}
					if _, ok := fl.Neurons[syn.From]; !ok {
						fmt.Printf("Deleted synapse %s in lobe %s: missing neuron %s in lobe %s\n", id, l.Name, syn.From, syn.FromLobe)
						delete(l.Synapses, id)
						break
					}
					if _, ok := tl.Neurons[syn.To]; !ok {
						fmt.Printf("Deleted synapse %s in lobe %s: missing neuron %s in lobe %s\n", id, l.Name, syn.To, syn.ToLobe)
						delete(l.Synapses, id)
						break
					}

					if b.Synapses == nil {
						b.Synapses = make(map[string]*brain.Synapse)
					}
					b.Synapses[id] = syn
					delete(l.Synapses, id)
					fmt.Printf("Moved synapse %s from lobe %s to brain-level (inter-lobe)\n", id, l.Name)
					break
				}
			}

		case Unset_Lobe_Fields:
			// Repair, don't delete: a synapse in a lobe's map belongs to that
			// lobe, so record what was already true.
			for _, l := range b.Lobes {
				if syn, ok := l.Synapses[id]; ok {
					syn.FromLobe = l.Name
					syn.ToLobe = l.Name
					fmt.Printf("Attributed synapse %s to lobe %s\n", id, l.Name)
					break
				}
			}

		case Wrong_Lobe_Synapse:
			// Move it to the lobe it names rather than destroying it. Deleting
			// was unconditional, so a recoverable filing error cost the
			// connection and its learned weight.
			for _, l := range b.Lobes {
				syn, ok := l.Synapses[id]
				if !ok {
					continue
				}

				target, exists := b.Lobes[syn.FromLobe]
				if !exists {
					fmt.Printf("Deleted synapse %s in lobe %s: names unknown lobe %s\n", id, l.Name, syn.FromLobe)
					delete(l.Synapses, id)
					break
				}
				if _, ok := target.Neurons[syn.From]; !ok {
					fmt.Printf("Deleted synapse %s: missing neuron %s in lobe %s\n", id, syn.From, syn.FromLobe)
					delete(l.Synapses, id)
					break
				}
				if _, ok := target.Neurons[syn.To]; !ok {
					fmt.Printf("Deleted synapse %s: missing neuron %s in lobe %s\n", id, syn.To, syn.ToLobe)
					delete(l.Synapses, id)
					break
				}

				delete(l.Synapses, id)
				target.AddSynapse(syn)
				fmt.Printf("Moved synapse %s from lobe %s to lobe %s\n", id, l.Name, target.Name)
				break
			}

		case Unattributable_Synapse:
			// Try to recover the lobes from neuron membership before giving up.
			if syn, ok := b.Synapses[id]; ok {
				b.AddSynapse(syn) // re-routes and fills in what it can
				if syn.FromLobe != "" && syn.ToLobe != "" {
					if syn.FromLobe == syn.ToLobe {
						delete(b.Synapses, id)
					}
					fmt.Printf("Attributed brain-level synapse %s to %s -> %s\n", id, syn.FromLobe, syn.ToLobe)
					continue
				}
				fmt.Printf("Deleted synapse %s: no lobe owns neurons %s / %s\n", id, syn.From, syn.To)
				delete(b.Synapses, id)
			}

		default:
			fmt.Printf("Unhandled complaint type %v for synapse %s\n", comp.ComplaintType, id)
		}
	}

	// Refresh derived state
	b.RefreshUpdateOrder()

	return nil
}
