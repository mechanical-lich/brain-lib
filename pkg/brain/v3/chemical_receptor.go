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
	ChemicalName string       // Name of chemical receptor uses
	UptakeRate   float64      // Rate at which the chemical is taken up
	ReceptorType ReceptorType // Type of receptor (excitatory/inhibitory)
	Target       string       // Target to stimulate based on chemical concentration
	TargetType   TargetType   // Type of target (neuron/modulator)
}
