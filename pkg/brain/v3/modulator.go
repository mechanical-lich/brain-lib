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

func (m *Modulator) Step(dt float64) {
	// simple exponential decay towards 0
	m.Level -= m.Level * m.DecayRate * dt
}
