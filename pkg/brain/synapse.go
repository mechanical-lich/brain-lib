package brain

import "math"

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
	// Hebbian makes the synapse learn from co-activity alone (two-factor),
	// with no modulator involved, by the instar rule in applyHebbian. The rates
	// come from the owning lobe's PlasticityConfig (the target lobe's for an
	// inter-lobe synapse). Independent of Plastic: a synapse with both learns
	// from co-activity always and from reward/punishment when a modulator is
	// present. The Hebbian rule reads current activations, not the eligibility
	// trace, so a Hebbian-only synapse keeps no trace. Additive and omitted
	// when false, so CurrentFormatVersion stays 1.
	Hebbian bool `json:"hebbian,omitempty"`
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

// applyHebbian advances a Hebbian synapse by dt seconds under the instar
// (competitive-learning) rule plus passive forgetting:
//
//	dW/dt = HebbianRate·post·(pre − W) + HebbianDecay·(InitWeight − W)
//
// pre and post are this step's activations (post-WTA). While the target fires,
// W moves toward the source's activation: up from a present input, down from
// an absent one. That depression is what lets a winner-takes-all lobe recruit
// a fresh slot for a new feature bundle — a slot trained on one bundle loses
// its weight from features it never saw, so on a novel bundle an untrained
// slot still holding its small InitWeight responds more. With the target
// silent only the forgetting term runs, whether the synapse is in use or not
// (unlike inactivity decay, which runs only while a synapse is unused).
//
// It does not use the eligibility trace: post·(pre − W) is not a function of
// the integrated pre·post. With activations held over the step the equation
// is linear in W, so it is integrated exactly — W relaxes toward
// (rate·post·pre + decay·InitWeight)/(rate·post + decay) at rate
// rate·post + decay — and the result does not depend on frame rate, the same
// reason decayFactor replaced the Euler form. It does not clip; callers clip
// once after every rule for the step has run.
func (s *Synapse) applyHebbian(cfg PlasticityConfig, pre, post, dt float64) {
	k := cfg.HebbianRate*post + cfg.HebbianDecay
	if k <= 0 {
		// Only reachable with a signed (negative) post activation outweighing
		// the decay: the equation then grows rather than relaxes, so there is
		// no fixed point to integrate toward. Take a plain Euler step.
		s.Weight += (cfg.HebbianRate*post*(pre-s.Weight) + cfg.HebbianDecay*(s.InitWeight-s.Weight)) * dt
		return
	}
	target := (cfg.HebbianRate*post*pre + cfg.HebbianDecay*s.InitWeight) / k
	s.Weight = target + (s.Weight-target)*math.Exp(-k*dt)
}

func (s *Synapse) Clip() {
	if s.Weight < s.MinWeight {
		s.Weight = s.MinWeight
	}
	if s.Weight > s.MaxWeight {
		s.Weight = s.MaxWeight
	}
}
