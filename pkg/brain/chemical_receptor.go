package brain

import "encoding/json"

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
	ChemicalName           string       `json:"chemical_name"`            // Name of chemical receptor uses
	UptakeRate             float64      `json:"uptake_rate"`              // Rate at which the chemical is taken up
	ReceptorType           ReceptorType `json:"receptor_type"`            // Type of receptor (excitatory/inhibitory)
	Target                 string       `json:"target"`                   // Target to stimulate based on chemical concentration
	TargetType             TargetType   `json:"target_type"`              // Type of target (neuron/modulator)
	Threshold              float64      `json:"threshold"`                // Threshold concentration for activation
	OutputNormalizationMin float64      `json:"output_normalization_min"` // Minimum output value - When both min and max are 0, output is forwarded as is.
	OutputNormalizationMax float64      `json:"output_normalization_max"` // Maximum output value - When min and max are not both 0, output is normalized between 0 and 1 using this range.
}

// legacyChemicalReceptor is the PascalCase shape receptors serialized as before
// they had JSON tags — everything else in the format is snake_case, so they were
// the odd ones out. Existing brain and lab snapshot files still use it.
type legacyChemicalReceptor struct {
	ChemicalName           *string       `json:"ChemicalName"`
	UptakeRate             *float64      `json:"UptakeRate"`
	ReceptorType           *ReceptorType `json:"ReceptorType"`
	Target                 *string       `json:"Target"`
	TargetType             *TargetType   `json:"TargetType"`
	Threshold              *float64      `json:"Threshold"`
	OutputNormalizationMin *float64      `json:"OutputNormalizationMin"`
	OutputNormalizationMax *float64      `json:"OutputNormalizationMax"`
}

// UnmarshalJSON reads both the snake_case form and the legacy PascalCase one,
// so files written by any version keep loading.
func (cr *ChemicalReceptor) UnmarshalJSON(data []byte) error {
	type alias ChemicalReceptor // avoids recursing into this method
	var current alias
	if err := json.Unmarshal(data, &current); err != nil {
		return err
	}
	*cr = ChemicalReceptor(current)

	var legacy legacyChemicalReceptor
	if err := json.Unmarshal(data, &legacy); err != nil {
		return err
	}
	if legacy.ChemicalName != nil {
		cr.ChemicalName = *legacy.ChemicalName
	}
	if legacy.UptakeRate != nil {
		cr.UptakeRate = *legacy.UptakeRate
	}
	if legacy.ReceptorType != nil {
		cr.ReceptorType = *legacy.ReceptorType
	}
	if legacy.Target != nil {
		cr.Target = *legacy.Target
	}
	if legacy.TargetType != nil {
		cr.TargetType = *legacy.TargetType
	}
	if legacy.Threshold != nil {
		cr.Threshold = *legacy.Threshold
	}
	if legacy.OutputNormalizationMin != nil {
		cr.OutputNormalizationMin = *legacy.OutputNormalizationMin
	}
	if legacy.OutputNormalizationMax != nil {
		cr.OutputNormalizationMax = *legacy.OutputNormalizationMax
	}
	return nil
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
