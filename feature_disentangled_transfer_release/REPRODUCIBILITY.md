# Reproducibility and manuscript correspondence

## Public release boundary

This release contains core model and preprocessing modules, five representative inputs transcribed from supplementary tables, synthetic examples, and focused tests. It excludes full experimental datasets, the 24 historical CNT candidate records, selected encoder weights, per-sample predictions, and result tables. The synthetic examples are not substitutes for those records. `python scripts/prepare_examples.py --check` validates the generated synthetic files only; it does not run training or reproduce a reported result.

The historical training and evaluation entry points are intentionally not included. Several retained helper functions still refer to the original private input layout, so the package is for architecture inspection and selected preprocessing checks, not a standalone reproduction pipeline.

## Representation learning

The source representation module has two dataset-specific encoders and a shared encoder. Source-only pretraining combines reconstruction loss, gradient-reversal domain discrimination, and normalized orthogonality loss. The shared encoder is subsequently frozen for target classification. The target classifiers are separate scikit-learn models; target classification loss is not jointly optimized with the representation network.

No soft LLM-prior loss or trainable feature-fusion coefficient is present in the reported classifier path. The paper and SI should describe the source-pretraining objective and downstream classifier fitting as separate stages.

## CNT classifier and candidate-handling discrepancy

The retained historical implementation treats candidate class labels as hard labels. The ExtraTrees gate uses candidates repaired and selected within each training fold, with a maximum of eight per class. The SVC head receives all 24 candidate records and applies random class oversampling.

This differs from the current SI text, which states that candidates are not appended to the SVC training set. The manuscript's Methods text also says the held-out sample was excluded from prior generation, but the available records establish only fold-local candidate repair and selection; they do not establish that historical candidate generation was independent of all held-out records. These statements require correction or confirmation against the original experiment records before submission. This release preserves the historical implementation and does not change its classifier behavior.

The reported 35-fold evaluation leaves out one real record per fold. Model and encoder selection were not nested inside this evaluation. Two Dataset C records have identical 15-dimensional inputs, and folds are by record rather than unique condition. The reported score should not be presented as an unbiased independent generalization estimate.

## MoS2 mapping

The revised mapper was audited across the 77 evaluated records and 23 substrate strings. It handles SiO2, silicon dioxide, quartz, silica and glass compounds before checking for elemental silicon. Standalone silicon and Si tokens retain the elemental-silicon value; SiC is handled explicitly, while Si3N4 remains unresolved. Ambiguous values remain missing rather than being guessed. Regression tests cover these precedence rules and mapped values.

The corrected historical HistGB comparison reports 51.67% accuracy for mapped-only inputs and 66.83% for mapped inputs plus shared CNT features and MoS2 advanced features. Those results pool 600 predictions from 30 stratified row splits. Predictions from repeated splits are not independent observations, and records are not grouped by DOI. The release omits the input rows, encoder weights, and row-level predictions required to regenerate these results.

## Remaining evidence limits

The complete experimental data are available from the corresponding authors upon reasonable request, subject to their data-sharing conditions. Data access alone does not recover the historical candidate-generation context, exact model-selection history, or all preprocessing state. The public tests verify software contracts only; they do not independently validate the scientific conclusions or reproduce the paper's performance.
