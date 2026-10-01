# Results provenance

## Audit date: 2026-10-01 (UTC)

Source inspected: `notebooks/credit_risk_classification.ipynb`, as tracked on main at commit `6d156f2`. The full dataset and training run were not available in the audit environment. The table below reports **saved notebook output**, not newly reproduced results.

| Metric | Neural network, threshold 0.50 | Pruned tree, threshold 0.70 |
| --- | ---: | ---: |
| Accuracy | 0.716912 | 0.821189 |
| Precision | 0.169242 | 0.203849 |
| Recall | 0.641246 | 0.418099 |
| F1 | 0.267803 | 0.274071 |
| AUC | 0.749549 | 0.702620 |

Evidence: final comparison code cell at zero-based index 101; pruning selection at index 84; threshold selection at index 85. The saved test split contains 46,127 rows. These results are internal test estimates and do not establish external generalization.

## Conflicting artifacts

| Item | Saved notebook/code | Separate report or older documentation |
| --- | --- | --- |
| Tree threshold | 0.70 | 0.75 in `relatorio_final.md` |
| Tree AUC | 0.702620 | 0.6658 in report |
| Tree F1 | 0.274071 | 0.2601 in report |
| Pruning selection | One validation split; 30,000-row stratified sample per candidate | Five-fold CV; 60,000 rows per training fold claimed in correction/spec documents |
| Selected alpha | 6.6067937e-05 in saved output | 3.6207798e-05 in correction document |
| Highest F1 | Tree at the saved operating threshold | Report claims neural network has higher F1 in its different experiment |

No evidence establishes that the separate report and model files came from the currently tracked notebook execution. Preserve them as historical artifacts until a versioned rerun links source, dataset, preprocessing, parameters and metrics.

## Interpretation

The saved neural network has higher AUC and recall; the tree has higher F1, precision and accuracy at its selected threshold. The choice requires a predeclared objective and error costs. AUC alone does not establish calibrated probabilities or fairness. BCE for the network and classification error for the tree are not interchangeable losses.

## Required verification for a canonical run

1. Record dataset fingerprint, package versions, source commit and split identifiers.
2. Fit all preprocessing only on training data and record fitted transformations.
3. Select pruning parameters and thresholds using training/validation or a specified CV procedure.
4. Complete every model-selection decision before the final held-out test evaluation.
5. Export machine-readable metrics, matched model/preprocessing artifacts and the report from the same run.
6. Retain historical results under distinct run IDs rather than silently overwriting their provenance.

This document corrects the public interpretation of existing artifacts; it does not claim that those training-protocol changes have already been implemented or validated.
