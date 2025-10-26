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
	InitWeight   float64 `json:"init_weight"`    // Initial weight for decay bounds
	MinDecayFrac float64 `json:"min_decay_frac"` // Minimum fraction of init weight (e.g., 0.1 = can decay to 10%)

	// Structural/usage stats
	LastUsed   int `json:"last_used"`
	UsageCount int `json:"usage_count"`
}

func (s *Synapse) Clip() {
	if s.Weight < s.MinWeight {
		s.Weight = s.MinWeight
	}
	if s.Weight > s.MaxWeight {
		s.Weight = s.MaxWeight
	}
}
