package brain

const (
	Dopamine      = "dopamine"      // Reward/pleasure signal
	Serotonin     = "serotonin"     // Punishment/pain signal
	Endorphin     = "endorphin"     // Wellbeing/satisfaction
	Acetylcholine = "acetylcholine" // Attention/focus
)

type Modulator struct {
	Name      string  `json:"name"`
	Level     float64 `json:"level"`
	DecayRate float64 `json:"decay_rate"`
}

// Add changes the modulator's level, clamping at zero.
//
// Levels must never go negative: plasticity multiplies them by a learning rate
// and applies the result with a fixed sign (reward positive, punishment
// negative). A negative level would invert that sign, turning punishment into
// reward. Inhibitory receptors targeting a modulator make this reachable, so
// the floor is enforced here rather than at each call site.
func (m *Modulator) Add(delta float64) {
	m.Level += delta
	if m.Level < 0 {
		m.Level = 0
	}
}

func (m *Modulator) Step(dt float64) {
	// simple exponential decay towards 0
	m.Level -= m.Level * m.DecayRate * dt

	// A brain loaded from an older file may carry a negative level. Decay would
	// creep it back toward zero over many ticks; clamp so it can't invert
	// plasticity in the meantime.
	if m.Level < 0 {
		m.Level = 0
	}
}
