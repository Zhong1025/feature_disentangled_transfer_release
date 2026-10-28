# Public examples and input formats

This directory separates representative inputs transcribed from the manuscript's Supplementary Information from independently generated synthetic inputs. Representative rows must not be mixed into training or evaluation. Synthetic inputs are marked `synthetic / demonstration only`; their numbers and labels are artificial and do not establish real synthesis recipes, historical LLM outputs or manuscript performance.

## Files and commands

| File, relative to this directory | Rows/records | Purpose |
|---|---:|---|
| `cnt/representative_inputs.csv` | 3 | SI Table S2: one input each for Dataset A, B and C |
| `mos2/representative_mapped_inputs.csv` | 2 | SI Table S7: the first two already-mapped inputs |
| `representative_provenance.json` | — | Table/page, field mapping, units and transcription provenance |
| `cnt/synthetic/source_a.csv` | 12 | Artificial source-A input descriptors |
| `cnt/synthetic/source_b.csv` | 12 | Artificial source-B input descriptors |
| `cnt/synthetic/target_c.csv` | 9 | Artificial target inputs, three per class |
| `cnt/synthetic/new1.json` | 3 | Artificial candidates, one per class |
| `cnt/synthetic/new2.json` | 3 | Artificial candidates, one per class |
| `mos2/synthetic/raw_inputs.csv` | 12 | Artificial raw-process inputs, four per layer class |

Run these commands from the repository root:

```bash
python scripts/prepare_examples.py
python scripts/prepare_examples.py --check
```

The first command writes the six synthetic files using standard-library Python and fixed seed 42. It does not read or modify the representative files. `--check` reads and verifies those six files, including their deterministic content, field order, counts, labels and declared missing values; it writes no files. It does not verify the representative table transcription or run a model. `--output PATH` selects another examples root for either command.

Training and evaluation entry points are not included in this core release. Generation/checking is the supported public demonstration; it is not an end-to-end training or paper-reproduction command. See [reproducibility limits](../REPRODUCIBILITY.md) and [data access](../DATA_AVAILABILITY.md).

## Shared 15-column descriptor order

CNT model input selects these columns by their exact names, in this order. The representative MoS₂ table also uses these column names because it is already mapped into the CNT-compatible interface.

| Order | Exact column name | Unit stated in SI Tables S2/S7 |
|---:|---|---|
| 1 | `Process Temp(oC)` | °C |
| 2 | `Total flow (sccm)` | sccm |
| 3 | `H2 flow (sccm)` | sccm |
| 4 | `C-flow (sccm)` | sccm |
| 5 | `H/C` | Not stated |
| 6 | `Carbon_Gibbs` | Not stated |
| 7 | `Carbon_Enthalpy` | Not stated |
| 8 | `Cat_MeltingPoint` | Not stated |
| 9 | `Predicted Formation Energy` | Not stated |
| 10 | `Cat_LatticeConstant` | Not stated |
| 11 | `Etch_level(ppm)` | ppm |
| 12 | `Thermal Conductivity` | Not stated |
| 13 | `Support_latticeConstant` | Not stated |
| 14 | `Support_AnnealingTemp` | Not stated |
| 15 | `Support_AnnealingTime` | Not stated |

“Not stated” means the supplied table does not declare a unit; it does not mean dimensionless. No energy, lattice, thermal-conductivity, annealing or target unit is inferred from a field name. Synthetic CNT values use this same interface without asserting missing physical units. The artificial CNT `H/C` values are constructed as `H2 flow / C-flow`; this generator rule does not alter rounded values transcribed from the SI.

For mapped MoS₂ inputs, inherited column names and printed units do not guarantee equivalent physical meaning. For example, the current mapping uses H₂S flow or a sulfur-source-temperature proxy for `C-flow`, promoter presence for `Etch_level`, and precursor ratio for `Cat_LatticeConstant`. These proxies must not be interpreted as measured CNT descriptors or used to reconstruct undisclosed raw conditions.

## CNT representative and synthetic schemas

The 27 columns of `cnt/representative_inputs.csv` are ordered as:

1. Metadata: `sample_id`, `data_origin`, `source_table`, `dataset`.
2. The 15 descriptors listed above.
3. Table targets: `Height`, `G/D`, `ln_density`, `Chiral Index`, `Chiral angle`, `Chirality Classification`, `label_0based`, `chirality_group`.

The three sample IDs are `si_s2_a`, `si_s2_b` and `si_s2_c`; `data_origin` is `supplied_supplementary_information`. Dataset A has the table's Height and G/D targets, B has ln_density, and C has chirality descriptors and labels. Other target cells are empty because they are not applicable to that dataset or not disclosed in the table. Units for those targets are not specified in Table S2. The C row preserves source code 3, report code 2 and the group near-Armchair. Metadata and target columns are not part of the 15 model inputs.

Synthetic source-A/source-B CSV order is `sample_id`, `data_origin`, followed by the 15 descriptors; no regression or chirality target is supplied for these source-pretraining inputs. Synthetic target-C has the same order plus `Chirality Classification` as its final column.

Each candidate JSON file is an array of three objects. Each object has `sample_id`, `data_origin`, then `data`; the `data` object contains the 15 descriptors in the order above followed by `Chirality Classification`. The IDs start with `synthetic_`. These records contain no historical explanation, probability distribution, model/provider metadata or LLM-generation claim.

| CNT CSV/JSON code (`Chirality Classification`) | Internal zero-based class | Class name |
|---:|---:|---|
| 1 | 0 | near-Zigzag |
| 2 | 1 | Chiral |
| 3 | 2 | near-Armchair |

Synthetic target labels and candidate labels are assigned artificially; class balance is a format demonstration, not a measured distribution. CNT synthetic numeric inputs are finite and contain no missing cells. Source A/B contain 12 rows each, target C contains three rows per class, and each candidate file contains one record per class (six candidates in total).

## MoS₂ representative mapped schema

The 22 columns of `mos2/representative_mapped_inputs.csv` are ordered as `sample_id`, `data_origin`, `source_table`, `row_id`, `doi`, `layer_group`, `layer_class_3`, then the 15 descriptors. The two records are `si_s7_01` and `si_s7_02`, both monolayer with `layer_class_3=0`. Row IDs and DOIs are the identifiers printed in SI Table S7; sample IDs are local example identifiers.

This file is **already mapped**. It lacks the raw process columns expected by `normalize_mos_layer_data` and `map_mos_to_cnt15`, and must not be passed to those functions as raw input. Numeric tokens retain the table's printed precision. The four literal `NA` values preserve missing entries: the first sample's `Support_AnnealingTemp` and `Support_AnnealingTime`; the second sample's `C-flow (sccm)` and `Support_AnnealingTemp`. They are not zeros and have not been imputed.

The `Thermal Conductivity` value for `si_s7_02` is corrected from 149 to 1.4 after the substrate mapper's `si` substring precedence bug was fixed. The revised SI Table S7 also corrects the same SiO2/Si mapping for records #3 and #4. This example remains for inspection only and is not an evaluation input.

## MoS₂ synthetic raw schema

`mos2/synthetic/raw_inputs.csv` has 30 columns, in the following exact order:

```text
row_id
data_origin
growth_temperature_C
mo_source_temperature_C
s_source_temperature_C
substrate_temperature_C
total_flow_sccm
ar_flow_sccm
n2_flow_sccm
h2_flow_sccm
h2s_flow_sccm
o2_flow_sccm
growth_pressure_Torr
holding_time_min
ramp_rate_C_min
mo_precursor_mass_mg
mo_precursor_thickness_nm
s_precursor_mass_mg
precursor_ratio
promoter_used
oxygen_used
nacl_mass_mg
growth_method
mo_precursor
s_precursor
substrate
substrate_class
pressure_type
promoter_type
layer_class_3
```

Raw numeric units are expressed in the field names: temperature `_C` means °C; `_sccm` means sccm; pressure uses Torr; holding time uses minutes; ramp rate uses °C/min; masses use mg; and precursor thickness uses nm. `precursor_ratio` is a numeric ratio whose specific molar/mass basis is not specified by this interface. `promoter_used` and `oxygen_used` are artificial 0/1 absence/presence flags. The seven categorical fields from `growth_method` to `promoter_type` are text. The literal promoter category `none` means no promoter, not missing text.

`layer_class_3` uses 0 = monolayer, 1 = bilayer and 2 = multilayer; each occurs four times. Do not subtract one from these MoS₂ labels. `row_id` and `data_origin` are metadata. Derived fields such as `promoter_group` and `nacl_mass_mg_num` are produced by preprocessing and are not raw CSV columns. No DOI or source-publication identity is invented for synthetic records.

The deliberate empty numeric cells are:

| `row_id` | Empty field |
|---|---|
| `synthetic_mos2_01` | `h2s_flow_sccm` |
| `synthetic_mos2_02` | `total_flow_sccm` |
| `synthetic_mos2_05` | `nacl_mass_mg` |
| `synthetic_mos2_06` | `total_flow_sccm` |
| `synthetic_mos2_07` | `h2s_flow_sccm` |
| `synthetic_mos2_09` | `mo_source_temperature_C` |

An empty cell represents unavailable information; an explicit `0` is a numeric value. The existing mapper can substitute the sum of available component flows for missing total flow, and use a sulfur-source-temperature proxy when H₂S flow is missing. These are code behaviors, not measurements recovered from the examples. Other imputation depends on the selected feature path and training/reference statistics; no missing value should be filled silently in the representative table itself.
