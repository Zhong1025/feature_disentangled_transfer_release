# Feature-disentangled prior transfer for small-sample chirality prediction in carbon nanotube synthesis

This repository contains the core representation-learning and classifier components described in the manuscript, the audited MoS2 descriptor mapper, and a small set of inspectable examples. It does not contain the complete experimental datasets, candidate records used in the historical evaluation, trained encoder weights, row-level predictions, or a runnable reproduction of the reported metrics.

## Model components

- `feature_transfer/networks.py`: shared and dataset-specific encoders, reconstruction network, and gradient-reversal domain discriminator.
- `feature_transfer/pretraining.py`: reconstruction, domain-adversarial, and orthogonality objectives for source-domain representation learning.
- `feature_transfer/features.py`: the 15-descriptor input contract, fold-local feature processing, and shared-embedding helpers. Historical data-loading functions refer to inputs that are not included here.
- `feature_transfer/cascade.py` and `feature_transfer/priors.py`: the rare-class cascade and candidate-handling code retained for implementation transparency. Historical candidate records are not included.
- `feature_transfer/mos2.py`: raw MoS2 input normalization, substrate-property mapping, and transfer-feature construction. Compound substrate names are resolved before elemental silicon tokens.

The source representation is learned first. The shared encoder is then used as a frozen feature extractor for downstream classifiers. The implementation does not jointly optimize the source representation and CNT classifier, and it does not implement a soft LLM-prior loss or a trainable feature-fusion coefficient.

## Examples and checks

`examples/` contains five representative rows transcribed from the manuscript's Supplementary Information and independently generated synthetic inputs. Representative rows are included for inspection only; synthetic values and labels are artificial. Neither set is a training or evaluation dataset.

Generate or verify the deterministic synthetic examples from the repository root:

```bash
python scripts/prepare_examples.py
python scripts/prepare_examples.py --check
```

Run the available code checks:

```bash
python -m unittest discover -s tests -v
```

The tests cover the corrected MoS2 substrate mapping and architecture-level tensor/gradient contracts. They do not train a model or validate manuscript metrics. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for implementation-to-manuscript differences and limits.

## Data and scope

The complete CNT and MoS2 experimental records are not included because they contain sensitive technical information. The historical training scripts, selected model weights, and row-level reference predictions are also excluded. The remaining architecture code can be inspected and imported, but reproducing the historical evaluation requires the unavailable inputs and additional records described in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

No software or data license has been selected. Public visibility does not grant reuse rights. Update [CITATION.cff](CITATION.cff) with publication details when available.
