package brain

type Synapse struct {
	ID        string  `json:"id"`
	From      string  `json:"from"`
	To        string  `json:"to"`
	FromLobe  string  `json:"from_lobe"` // only needed for cross-lobe synapses
	ToLobe    string  `json:"to_lobe"`   // only needed for cross-lobe synapses
	Weight    float64 `json:"weight"`
	Plastic   bool    `json:"plastic"`
	Trace     float64 `json:"trace"`
	MinWeight float64 `json:"min_weight"`
	MaxWeight float64 `json:"max_weight"`

	// Plasticity control
	InitWeight float64 `json:"init_weight"` // Initial weight for decay bounds

	// TraceDecay overrides the eligibility-trace decay rate (per second) for
	// this synapse alone. 0 means "use the lobe's": an intra-lobe synapse
	// falls back to its lobe's PlasticityConfig.TraceDecay, an inter-lobe one
	// to its target lobe's. It lets a few synapses carry a wider reinforcement
	// window than the rest of the lobe without widening it for every plastic
	// synapse that shares the lobe. Additive and omitted when zero, so files
	// without it load unchanged and CurrentFormatVersion stays 1.
	TraceDecay float64 `json:"trace_decay,omitempty"`
	// MinDecayFrac is serialized but never read. Inactivity decay moves a weight
	// toward InitWeight without a floor, so this has no effect — the plasticity
	// guide documented it as active, which it never was. Kept so existing files
	// round-trip without losing data.
	MinDecayFrac float64 `json:"min_decay_frac"`

	// Structural/usage stats
	LastUsed   int `json:"last_used"`
	UsageCount int `json:"usage_count"`
}

// traceDecayFor resolves the trace decay rate for syn: its own TraceDecay when
// set, otherwise the rate from lobeCfg (the owning lobe's config for an
// intra-lobe synapse, the target lobe's for an inter-lobe one).
func traceDecayFor(syn *Synapse, lobeCfg PlasticityConfig) float64 {
	if syn.TraceDecay > 0 {
		return syn.TraceDecay
	}
	return lobeCfg.TraceDecay
}

func (s *Synapse) Clip() {
	if s.Weight < s.MinWeight {
		s.Weight = s.MinWeight
	}
	if s.Weight > s.MaxWeight {
		s.Weight = s.MaxWeight
	}
}
