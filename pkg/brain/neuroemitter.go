package brain

// Neuroemitter represents a structure in a lobe that releases chemicals based on neural activity.
// Unlike receptors which consume chemicals to affect neurons, emitters produce chemicals when
// neurons reach certain activation thresholds.
type Neuroemitter struct {
	ID           string  `json:"id"`            // Unique identifier for this emitter
	ChemicalName string  `json:"chemical_name"` // Name of chemical this emitter releases
	EmissionRate float64 `json:"emission_rate"` // Rate at which the chemical is emitted when activated
	Threshold    float64 `json:"threshold"`     // Activation threshold that must be met before emitting

	// AccumulatedActivation is input received from synapses since the last
	// emission. It is persisted: it was previously unexported, so a brain saved
	// between receiving input and emitting silently dropped the pending charge.
	AccumulatedActivation float64 `json:"accumulated_activation,omitempty"`
}

// NewNeuroemitter creates a new neuroemitter
func NewNeuroemitter(id, chemicalName string, emissionRate, threshold float64) *Neuroemitter {
	return &Neuroemitter{
		ID:           id,
		ChemicalName: chemicalName,
		EmissionRate: emissionRate,
		Threshold:    threshold,
	}
}

// ReceiveInput accumulates input from a synapse (called during neuron activation propagation)
func (ne *Neuroemitter) ReceiveInput(input float64) {
	ne.AccumulatedActivation += input
}

// Step processes the neuroemitter for a time step, emitting chemicals if threshold is met
// and returns the amount of chemical to add to the chemistry map
func (ne *Neuroemitter) Step(dt float64) float64 {
	emission := 0.0

	// Only emit if accumulated activation meets or exceeds threshold
	if ne.AccumulatedActivation >= ne.Threshold {
		emission = ne.AccumulatedActivation * ne.EmissionRate * dt
	}

	// Reset accumulated activation for next step
	ne.AccumulatedActivation = 0.0

	return emission
}

// Reset clears the accumulated activation
func (ne *Neuroemitter) Reset() {
	ne.AccumulatedActivation = 0.0
}
