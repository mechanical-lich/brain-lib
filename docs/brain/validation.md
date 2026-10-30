---
title: Validation
parent: Brain
nav_order: 6
permalink: /brain/validation/
---

# Validation and repair (`pkg/brain/validation`)
{: .no_toc }

1. TOC
{:toc}

`validation.ValidateBrain(b)` returns every structural problem it can find,
as a list of `Complaint{ComplaintType, ID}`, in a stable order.
`validation.FixSynapses(b, complaints)` repairs the ones that have a
mechanical fix and returns a `Report` of what it did. `cmd/validator` wraps
both; see the [command-line tools](../../cli/#validator) page.

Neither function prints anything. The report is for the caller to show, and
`Report.Deletions()` picks out the destructive repairs a UI should confirm
before overwriting the original.

```go
complaints := validation.ValidateBrain(b)
if len(complaints) > 0 {
    report, _ := validation.FixSynapses(b, complaints) // error is always nil
    fmt.Print(report)
    if dels := report.Deletions(); len(dels) > 0 {
        // ask before saving over the original
    }
}
```

## Wiring complaints

These concern [synapse ownership](../lobes-and-synapses#synapse-ownership) and
dangling references. Most are repairable.

| Complaint | Meaning | Repair |
|---|---|---|
| `Inter_Synapse_To_Same_Lobe` | A brain-level synapse whose `from_lobe` equals its `to_lobe`. | Moved into that lobe. |
| `Intra_Synapse_To_Different_Lobe` | A lobe-level synapse whose lobes differ. | Moved to the brain. |
| `Wrong_Lobe_Synapse` | A synapse filed in a lobe that is not its own. | Moved to the right owner. |
| `Unset_Lobe_Fields` | A lobe-level synapse with empty `from_lobe` or `to_lobe`. | Attributed from the lobe it sits in. |
| `Unattributable_Synapse` | A brain-level synapse whose lobes cannot be determined from neuron membership. | Deleted. |
| `Missing_Neuron_From`, `Missing_Neuron_To` | The named neuron does not exist. | Deleted. |
| `Missing_From_Lobe`, `Missing_To_Lobe` | The named lobe does not exist. | Deleted. |

A repair is recorded as one of `moved`, `attributed`, `deleted` or
`skipped`, with a reason.

## Structural complaints

Added after the wiring checks, when a receptor pointing at nothing, a weight
of NaN or an ID shared between a neuron and an emitter were all passing
silently.

| Complaint | Meaning |
|---|---|
| `Dangling_Receptor_Target` | A receptor targets a neuron or modulator that does not exist. |
| `Dangling_Emitter_Synapse` | A lobe synapse targets neither a neuron nor an emitter. |
| `Duplicate_Node_ID` | A neuron and an emitter share an ID in one lobe. |
| `Init_Weight_Out_Of_Range` | `init_weight` sits outside `[min_weight, max_weight]`, so decay pulls toward a value the clip forbids. |
| `Non_Finite_Weight` | `weight`, `init_weight` or `trace` is NaN or infinite. |
| `Lobe_Name_Key_Mismatch` | A lobe's `name` disagrees with its key in `lobes`. |
| `Frozen_Persisting_Neuron` | A persisting neuron whose resolved decay rate is 0: it can rise but never fall. |
| `Invalid_Trace_Decay` | A synapse `trace_decay` that is negative, NaN or infinite (0 means "use the lobe's"). |

## Plasticity warnings

Warnings only. `FixSynapses` skips them, because the right rate is a design
decision, not a repair.

| Complaint | Meaning |
|---|---|
| `Hebbian_Without_Decay` | A Hebbian synapse whose lobe config has `hebbian_decay` 0. A binding is then never forgotten, so a recognition slot is never freed for reuse. |
| `Invalid_Hebbian_Rate` | A `plasticity_config` whose `hebbian_rate` or `hebbian_decay` is negative, NaN or infinite. |

## Using it from the lab

The validator is a library first. The critter laboratory calls
`ValidateBrain` on load, shows the complaints, and offers `FixSynapses` with
the report's deletions highlighted. The tests in `validation_test.go`,
`fix_test.go` and `repair_test.go` construct each complaint in isolation and
are the clearest statement of what each one means.
