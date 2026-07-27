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
