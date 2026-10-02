# Frozen raw-feature MLP run and error analysis

Executed on 2026-10-01 using the same dataset and exact train/validation/test row positions as tree-v1 (verified by comparing all split arrays). The protocol was committed as `45d666590aa1d1bea7243ac288b86de7df163a93` before training. This is a separate MLP, not a reproduction of the historical TensorFlow notebook.

Validation F1 selected architecture **(64,32)** and threshold **0.70** from the two predeclared architectures and nine thresholds. Training used 30 epochs and balanced sample weights; no refit or test-based search followed. Fixed-budget convergence warnings are recorded in the [unaltered manifest](results/neural-v1.json).

| Test metric | Tree-v1 | MLP-v1 | Prior dummy |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.722700 | 0.744282 | 0.500000 |
| Average precision | 0.197235 | 0.212916 | 0.080734 |
| F1 at selected 0.70 | 0.269388 | 0.288744 | 0.000000 (0.50) |
| Precision | 0.238095 | 0.248930 | Undefined, no positives predicted |
| Recall | 0.310150 | 0.343716 | 0.000000 |
| Brier score, lower is better | 0.209063 | 0.207539 | 0.074216 |

The MLP improves discrimination and F1 on this reused split. Both balanced models have substantially **worse probability calibration** than the prior baseline. Their scores must not be interpreted as calibrated default probabilities or deployed approval/rejection rules.

## Error counts

| Model | True negatives | False positives | False negatives | True positives |
| --- | ---: | ---: | ---: | ---: |
| Tree | 38,707 | 3,696 | 2,569 | 1,155 |
| MLP | 38,541 | 3,862 | 2,444 | 1,280 |

The MLP detects 125 more positive labels while generating 166 more false positives at these selected operating points. About three quarters of its positive predictions are false positives. Business cost and acceptable operating thresholds have not been established.

Calibration-bin counts and mean scores versus observed positive rates are preserved for both models. For example, MLP scores averaging 0.746 in its 0.7–0.8 bin correspond to only 0.213 observed positives. Balanced training reweights the classes and these scores are not calibrated to the original prevalence.

The [tree analysis](results/tree-error-analysis-v1.json) is post-hoc description of the unchanged frozen tree artifact; it did not alter selection or fit. No calibration correction was fitted using this test set. A future calibration study needs training/validation partitions declared beforehand and a protected evaluation set.

## Limits and reproduction

Follow [the fixed neural protocol](NEURAL_PROTOCOL.md), using `requirements-tree.txt`. The source, script/data hashes, package versions, all validation candidates and training loss are recorded. Applicant-level data/artifacts remain local and ignored.

Although preprocessing and selection use only training/validation in these scripts, this dataset and potentially these row positions have prior historical test exposure. The comparison is useful internal evidence, not independent external validation, statistical significance, fairness or clinical/financial deployment readiness. No additional architectures or epochs were selected after inspecting these test results.

The runner later gained working-directory-independent source-commit lookup. The original manifest keeps the fingerprint of the training version at `45d6665`; this metadata fix does not change preprocessing, selection or training.
