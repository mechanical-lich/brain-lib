---
title: File format
parent: Brain
nav_order: 5
permalink: /brain/file-format/
---

# Brain file format
{: .no_toc }

1. TOC
{:toc}

A brain is saved as JSON with snake_case keys. `Brain.Save(path)` and
`Brain.ToJSON()` write it; `brain.LoadBrain(path)` and
`brain.LoadFromJson(bytes)` read it. Everything the simulation needs is in the
file, including live state (activations, traces, modulator levels, the tick),
so a loaded brain resumes exactly where it was saved.

## Shape

```json
{
  "format_version": 1,
  "tick": 48213,
  "plasticity_config": {
    "trace_decay": 1.0,
    "reward_learning_rate": 10.0,
    "punish_learning_rate": 15.0,
    "inactivity_decay": 0.00001
  },
  "modulators": {
    "dopamine": {"name": "dopamine", "level": 0.0, "decay_rate": 0.5}
  },
  "chemical_receptors": [],
  "lobes": {
    "senses": {
      "name": "senses",
      "persists": false,
      "decay_rate": 0,
      "neurons": {
        "see_food": {
          "id": "see_food", "bias": 0, "activation": 0, "activation_val": 0,
          "external_lobe_input": 0, "learning_rate": 0, "decay_rate": 0,
          "max_value": 0, "min_value": 0, "threshold": 0,
          "x": 30, "y": 100
        }
      },
      "synapses": {},
      "modulators": {},
      "chemical_receptors": [
        {
          "chemical_name": "food_smell", "uptake_rate": 1.0,
          "receptor_type": "excitatory", "target": "see_food",
          "target_type": "neuron", "threshold": 0,
          "output_normalization_min": 0, "output_normalization_max": 1
        }
      ],
      "neuroemitters": {},
      "plasticity_config": {
        "trace_decay": 1.0, "reward_learning_rate": 10.0,
        "punish_learning_rate": 15.0, "inactivity_decay": 0.00001
      }
    },
    "decision": {
      "name": "decision",
      "winner_takes_all": true,
      "neurons": {
        "approach": {"id": "approach", "bias": 0.1, "activation": 0, "activation_val": 0,
                     "external_lobe_input": 0, "learning_rate": 0, "decay_rate": 0,
                     "max_value": 0, "min_value": 0, "threshold": 0},
        "flee":     {"id": "flee", "bias": 0.1, "activation": 0, "activation_val": 0,
                     "external_lobe_input": 0, "learning_rate": 0, "decay_rate": 0,
                     "max_value": 0, "min_value": 0, "threshold": 0}
      },
      "synapses": {
        "approach->flee": {
          "id": "approach->flee", "from": "approach", "to": "flee",
          "from_lobe": "decision", "to_lobe": "decision",
          "weight": -0.5, "plastic": false, "trace": 0,
          "min_weight": -2, "max_weight": 2, "init_weight": -0.5,
          "min_decay_frac": 0.1, "last_used": 0, "usage_count": 0
        }
      },
      "modulators": {},
      "chemical_receptors": [],
      "neuroemitters": {},
      "plasticity_config": {
        "trace_decay": 1.0, "reward_learning_rate": 10.0,
        "punish_learning_rate": 15.0, "inactivity_decay": 0.00001
      }
    }
  },
  "synapses": {
    "7f3c1e2a-...": {
      "id": "7f3c1e2a-...", "from": "see_food", "to": "approach",
      "from_lobe": "senses", "to_lobe": "decision",
      "weight": 0.1, "plastic": true, "trace": 0,
      "min_weight": -2, "max_weight": 2, "init_weight": 0.1,
      "min_decay_frac": 0.1, "last_used": 0, "usage_count": 0
    }
  }
}
```

The top-level `synapses` map holds only **inter-lobe** synapses; each lobe's
`synapses` holds its **intra-lobe** ones. See
[synapse ownership](../lobes-and-synapses#synapse-ownership). The validator
repairs files that break this rule.

## Field reference

The per-type tables on the [neurons](../neurons),
[lobes and synapses](../lobes-and-synapses) and [chemistry](../chemistry) pages
give every key. Keys with `omitempty` in the Go struct are absent when zero
or false: `persists`, `capacitor`, `hidden`, `x`, `y`, `group_id` on
neurons; `winner_takes_all` and the editor rectangle on lobes; `trace_decay`
and `hebbian` on synapses; `hebbian_rate` and `hebbian_decay` in plasticity
configs; `accumulated_activation` on emitters. A file that never uses those
features looks exactly as it did before they existed.

Two keys are written and read but never used by the simulation:
`learning_rate` on a neuron and `min_decay_frac` on a synapse. They are kept
so existing files round-trip without losing data.

## Versions

`format_version` is the schema version. `CurrentFormatVersion` in
`pkg/brain/brain.go` is what `Save` writes; today it is `1`.

| Version | Meaning |
|---|---|
| `0` (or absent) | Unversioned. Rates in `plasticity_config` were per simulation step, decay used a linear approximation, and chemical receptors serialized in PascalCase (`ChemicalName`, `UptakeRate`, ...) while the rest of the file was snake_case. |
| `1` | All rates are per second, decay is exponential, receptors and neuroemitters use snake_case keys, and a neuroemitter's pending charge is persisted. |

A version 0 file still loads: PascalCase receptor keys are accepted and a
missing plasticity config is defaulted. Its **rate values are not converted**,
because there is no way to know the frame rate they were tuned at. A version
0 file loaded as version 1 will learn far more slowly than it used to.
Re-tune it, or migrate it deliberately, and save it to stamp the new version.

A file whose `format_version` is **greater** than the library's refuses to
load with an error, rather than guessing.

Features added since version 1 (per-synapse `trace_decay`, Hebbian learning)
are additive and omitted when unset, so they did not bump the version.

## Example files

The repository's `local_data/` is git-ignored, so there are no checked-in
brains. `cmd/validator` writes a repaired copy next to the original as
`<name>.json.fixed`.
