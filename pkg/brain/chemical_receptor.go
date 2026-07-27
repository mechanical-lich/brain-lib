package brain

type ReceptorType string

const (
	ReceptorTypeExcitatory ReceptorType = "excitatory"
	ReceptorTypeInhibitory ReceptorType = "inhibitory"
)

type TargetType string

const (
	TargetTypeNeuron    TargetType = "neuron"
	TargetTypeModulator TargetType = "modulator"
)

// ChemicalReceptor represents a receptor in a lobe that responds to a specific chemical and applies it to a target neuron.
type ChemicalReceptor struct {
	ChemicalName           string       // Name of chemical receptor uses
	UptakeRate             float64      // Rate at which the chemical is taken up
	ReceptorType           ReceptorType // Type of receptor (excitatory/inhibitory)
	Target                 string       // Target to stimulate based on chemical concentration
	TargetType             TargetType   // Type of target (neuron/modulator)
	Threshold              float64      // Threshold concentration for activation
	OutputNormalizationMin float64      // Minimum output value - When both min and max are 0, output is forwarded as is.
	OutputNormalizationMax float64      // Maximum output value - When min and max are not both 0, output is normalized between 0 and 1 using this range.
}

// GetOutput computes how much of the chemical this receptor would consume and
// the resulting output to apply to the receptor's target. It does NOT mutate
// the chemistry map.
func (cr *ChemicalReceptor) GetOutput(chemistry map[string]float64, dt float64) (float64, float64) {
	concentration, ok := chemistry[cr.ChemicalName]
	if !ok || concentration < cr.Threshold || dt <= 0.0 {
		return 0.0, 0.0
	}

	// Uptake scales with dt, and cannot exceed what is actually in the pool.
	//
	// The clamp used to be applied only to `consumed`, not to `output`, so a
	// receptor with UptakeRate 5.0 against a pool of 1.0 delivered 5.0 to its
	// target while removing 1.0 — matter out of nothing, and a reward source
	// that scaled with the rate rather than the supply.
	uptake := concentration * cr.UptakeRate * dt
	if uptake > concentration {
		uptake = concentration
	}

	// Normalize output if min and max are set (based on concentration)
	output := uptake
	if cr.OutputNormalizationMin != 0 || cr.OutputNormalizationMax != 0 {
		if concentration <= cr.OutputNormalizationMin {
			output = 0.0
		} else if concentration >= cr.OutputNormalizationMax {
			output = 1.0 * dt
		} else {
			// normalize concentration into [0,1] then scale by dt
			output = ((concentration - cr.OutputNormalizationMin) / (cr.OutputNormalizationMax - cr.OutputNormalizationMin)) * dt
		}
	}

	// Apply receptor type (inhibitory receptors produce negative output)
	if cr.ReceptorType == ReceptorTypeInhibitory {
		output = -output
	}

	// consumed amount is the actual uptake (always positive, already clamped)
	consumed := uptake

	return output, consumed
}
