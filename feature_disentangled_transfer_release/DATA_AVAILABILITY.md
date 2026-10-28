# Data availability

## Materials in this repository

The repository contains five representative rows transcribed from Supplementary Information tables: three CNT rows from Table S2 and two already-mapped MoS2 rows from Table S7. Their provenance and field interpretation are documented in `examples/representative_provenance.json` and `examples/README.md`. These excerpts are provided for inspection only and must not be treated as a training or evaluation dataset.

The remaining example files are synthetic. A deterministic standard-library script creates them without reading experimental records or contacting an external model service. Their artificial values and labels do not represent synthesis recipes, historical candidate records, or evidence for the manuscript's reported performance.

The complete experimental datasets are not included because they contain sensitive technical information. The data supporting the study may be requested from the corresponding authors, subject to their data-sharing conditions:

- pengyijie@gsm.pku.edu.cn
- yuehu@wzu.edu.cn
- dewulin@pku.edu.cn

Historical candidate records, selected encoder weights, and row-level predictions are also excluded. Access to the experimental data alone is not sufficient to recreate all historical preprocessing and model-selection decisions; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

No software or data license has been selected. Public visibility does not grant reuse rights. The authors should add the applicable license and publication details before release.
