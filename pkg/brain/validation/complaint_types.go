package validation

type ComplaintType string

const (
	Inter_Synapse_To_Same_Lobe      ComplaintType = "Inter_Synapse_To_Same_Lobe"      // Inter-lobe synapse points to same lobe
	Intra_Synapse_To_Different_Lobe ComplaintType = "Intra_Synapse_To_Different_Lobe" // Intra-lobe synapse points to different lobe
	Missing_Neuron_From             ComplaintType = "Missing_Neuron_From"             // Missing 'From' neuron
	Missing_Neuron_To               ComplaintType = "Missing_Neuron_To"               // Missing 'To' neuron
	Missing_To_Lobe                 ComplaintType = "Missing_To_Lobe"                 // Missing 'To' lobe
	Missing_From_Lobe               ComplaintType = "Missing_From_Lobe"               // Missing 'From' lobe
	Wrong_Lobe_Synapse              ComplaintType = "Wrong_Lobe_Synapse"              // Synapse does not belong to lobe
	Unset_Lobe_Fields               ComplaintType = "Unset_Lobe_Fields"               // Synapse in a lobe's map with empty FromLobe/ToLobe - repairable
	Unattributable_Synapse          ComplaintType = "Unattributable_Synapse"          // Brain-level synapse whose lobes cannot be determined
)

// Additional checks. ValidateBrain previously looked only at synapse wiring, so
// a receptor or emitter pointing at nothing, a weight of NaN, or an ID shared
// between a neuron and an emitter all passed silently.
const (
	Dangling_Receptor_Target ComplaintType = "Dangling_Receptor_Target" // Receptor targets a neuron/modulator that does not exist
	Dangling_Emitter_Synapse ComplaintType = "Dangling_Emitter_Synapse" // Synapse targets neither a neuron nor an emitter
	Duplicate_Node_ID        ComplaintType = "Duplicate_Node_ID"        // A neuron and an emitter share an ID in one lobe
	Init_Weight_Out_Of_Range ComplaintType = "Init_Weight_Out_Of_Range" // InitWeight sits outside [MinWeight, MaxWeight]
	Non_Finite_Weight        ComplaintType = "Non_Finite_Weight"        // Weight, InitWeight or Trace is NaN or Inf
	Lobe_Name_Key_Mismatch   ComplaintType = "Lobe_Name_Key_Mismatch"   // lobe.Name disagrees with its map key
	Frozen_Persisting_Neuron ComplaintType = "Frozen_Persisting_Neuron" // Persisting neuron with DecayRate 0 can never change value
	Invalid_Trace_Decay      ComplaintType = "Invalid_Trace_Decay"      // Synapse TraceDecay is negative, NaN or Inf (0 means "use the lobe's")
)
