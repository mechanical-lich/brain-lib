package validation

import (
	"fmt"
	"sort"
	"strings"

	"github.com/mechanical-lich/brain-lib/pkg/brain"
)

// Action describes what a repair did to a synapse.
type Action string

const (
	Deleted    Action = "deleted"
	Moved      Action = "moved"
	Attributed Action = "attributed"
	Skipped    Action = "skipped"
)

// Repair records one change FixSynapses made.
type Repair struct {
	SynapseID string
	Action    Action
	Reason    string
}

// Report is the full set of repairs from one FixSynapses call.
//
// FixSynapses used to print straight to stdout, which made it untestable and
// unusable from a library or a server — the lab could repair a brain but had no
// way to tell the user what had changed, and a caller could not tell a
// destructive repair from a harmless one.
type Report struct {
	Repairs []Repair
}

// Deletions returns just the destructive repairs, which are the ones a caller
// most likely wants to confirm before saving over the original.
func (r Report) Deletions() []Repair {
	var out []Repair
	for _, rep := range r.Repairs {
		if rep.Action == Deleted {
			out = append(out, rep)
		}
	}
	return out
}

func (r Report) String() string {
	var b strings.Builder
	for _, rep := range r.Repairs {
		fmt.Fprintf(&b, "%s synapse %s: %s\n", rep.Action, rep.SynapseID, rep.Reason)
	}
	return b.String()
}

func sortedLobeNames(b *brain.Brain) []string {
	names := make([]string, 0, len(b.Lobes))
	for name := range b.Lobes {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// FixSynapses repairs the problems in complaints and reports what it did.
//
// The error return is retained for API compatibility and is always nil; the
// Report carries the outcome.
func FixSynapses(b *brain.Brain, complaints []Complaint) (Report, error) {
	var report Report
	record := func(action Action, id, reason string) {
		report.Repairs = append(report.Repairs, Repair{SynapseID: id, Action: action, Reason: reason})
	}

	if b.Synapses == nil {
		b.Synapses = make(map[string]*brain.Synapse)
	}

	for _, comp := range complaints {
		id := comp.ID
		switch comp.ComplaintType {
		case Inter_Synapse_To_Same_Lobe:
			syn, ok := b.Synapses[id]
			if !ok {
				continue
			}
			if syn.FromLobe == "" {
				record(Deleted, id, "missing FromLobe for intra-lobe move")
				delete(b.Synapses, id)
				continue
			}
			l, ok := b.Lobes[syn.FromLobe]
			if !ok || l.Neurons == nil {
				record(Deleted, id, fmt.Sprintf("lobe %s missing", syn.FromLobe))
				delete(b.Synapses, id)
				continue
			}
			if _, ok := l.Neurons[syn.From]; !ok {
				record(Deleted, id, fmt.Sprintf("missing neuron %s in lobe %s", syn.From, syn.FromLobe))
				delete(b.Synapses, id)
				continue
			}
			if _, ok := l.Neurons[syn.To]; !ok {
				record(Deleted, id, fmt.Sprintf("missing neuron %s in lobe %s", syn.To, syn.ToLobe))
				delete(b.Synapses, id)
				continue
			}
			l.AddSynapse(syn)
			delete(b.Synapses, id)
			record(Moved, id, fmt.Sprintf("into lobe %s (intra-lobe)", syn.FromLobe))

		case Missing_From_Lobe, Missing_To_Lobe, Missing_Neuron_From, Missing_Neuron_To:
			if _, ok := b.Synapses[id]; ok {
				record(Deleted, id, string(comp.ComplaintType))
				delete(b.Synapses, id)
				continue
			}
			for _, name := range sortedLobeNames(b) {
				l := b.Lobes[name]
				if _, ok := l.Synapses[id]; ok {
					record(Deleted, id, fmt.Sprintf("%s (in lobe %s)", comp.ComplaintType, l.Name))
					delete(l.Synapses, id)
					break
				}
			}

		case Intra_Synapse_To_Different_Lobe:
			for _, name := range sortedLobeNames(b) {
				l := b.Lobes[name]
				syn, ok := l.Synapses[id]
				if !ok {
					continue
				}

				if syn.FromLobe == "" {
					syn.FromLobe = l.Name
					record(Attributed, id, fmt.Sprintf("FromLobe set to %s", l.Name))
				}
				if syn.ToLobe == "" {
					found := ""
					for _, lname := range sortedLobeNames(b) {
						if ol := b.Lobes[lname]; ol.Neurons != nil {
							if _, ok := ol.Neurons[syn.To]; ok {
								found = lname
								break
							}
						}
					}
					if found == "" {
						record(Deleted, id, fmt.Sprintf("could not infer ToLobe for neuron %s", syn.To))
						delete(l.Synapses, id)
						break
					}
					syn.ToLobe = found
					record(Attributed, id, fmt.Sprintf("ToLobe inferred as %s", found))
				}

				fl, ok1 := b.Lobes[syn.FromLobe]
				tl, ok2 := b.Lobes[syn.ToLobe]
				if !ok1 || !ok2 || fl.Neurons == nil || tl.Neurons == nil {
					record(Deleted, id, fmt.Sprintf("missing lobe or neurons (%s->%s)", syn.FromLobe, syn.ToLobe))
					delete(l.Synapses, id)
					break
				}
				if _, ok := fl.Neurons[syn.From]; !ok {
					record(Deleted, id, fmt.Sprintf("missing neuron %s in lobe %s", syn.From, syn.FromLobe))
					delete(l.Synapses, id)
					break
				}
				if _, ok := tl.Neurons[syn.To]; !ok {
					record(Deleted, id, fmt.Sprintf("missing neuron %s in lobe %s", syn.To, syn.ToLobe))
					delete(l.Synapses, id)
					break
				}

				b.Synapses[id] = syn
				delete(l.Synapses, id)
				record(Moved, id, fmt.Sprintf("from lobe %s to brain-level (inter-lobe)", l.Name))
				break
			}

		case Unset_Lobe_Fields:
			// Repair, don't delete: a synapse in a lobe's map belongs to that
			// lobe, so record what was already true.
			for _, name := range sortedLobeNames(b) {
				l := b.Lobes[name]
				if syn, ok := l.Synapses[id]; ok {
					syn.FromLobe = l.Name
					syn.ToLobe = l.Name
					record(Attributed, id, fmt.Sprintf("attributed to lobe %s", l.Name))
					break
				}
			}

		case Wrong_Lobe_Synapse:
			// Move it to the lobe it names rather than destroying it. Deleting
			// was unconditional, so a recoverable filing error cost the
			// connection and its learned weight.
			for _, name := range sortedLobeNames(b) {
				l := b.Lobes[name]
				syn, ok := l.Synapses[id]
				if !ok {
					continue
				}

				target, exists := b.Lobes[syn.FromLobe]
				if !exists {
					record(Deleted, id, fmt.Sprintf("names unknown lobe %s", syn.FromLobe))
					delete(l.Synapses, id)
					break
				}
				if _, ok := target.Neurons[syn.From]; !ok {
					record(Deleted, id, fmt.Sprintf("missing neuron %s in lobe %s", syn.From, syn.FromLobe))
					delete(l.Synapses, id)
					break
				}
				if _, ok := target.Neurons[syn.To]; !ok {
					record(Deleted, id, fmt.Sprintf("missing neuron %s in lobe %s", syn.To, syn.ToLobe))
					delete(l.Synapses, id)
					break
				}

				delete(l.Synapses, id)
				target.AddSynapse(syn)
				record(Moved, id, fmt.Sprintf("from lobe %s to lobe %s", l.Name, target.Name))
				break
			}

		case Unattributable_Synapse:
			// Try to recover the lobes from neuron membership before giving up.
			syn, ok := b.Synapses[id]
			if !ok {
				continue
			}
			b.AddSynapse(syn) // re-routes and fills in what it can
			if syn.FromLobe != "" && syn.ToLobe != "" {
				if syn.FromLobe == syn.ToLobe {
					delete(b.Synapses, id)
				}
				record(Attributed, id, fmt.Sprintf("%s -> %s", syn.FromLobe, syn.ToLobe))
				continue
			}
			record(Deleted, id, fmt.Sprintf("no lobe owns neurons %s / %s", syn.From, syn.To))
			delete(b.Synapses, id)

		default:
			record(Skipped, id, fmt.Sprintf("unhandled complaint type %v", comp.ComplaintType))
		}
	}

	// Refresh derived state
	b.RefreshUpdateOrder()

	return report, nil
}
