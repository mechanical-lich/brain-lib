package brain

const Dopamine = "dopamine"

type Modulator struct {
	Name      string  `json:"name"`
	Level     float64 `json:"level"`
	DecayRate float64 `json:"decay_rate"`
}

func (m *Modulator) Step(dt float64) {
	// simple exponential decay towards 0
	m.Level -= m.Level * m.DecayRate * dt
}
