# Reproducible tree experiment

This is a **new, separate experiment**, not a reproduction of the historical notebook or its neural network. A real-data run is now recorded in [Home Credit results](HOMECREDIT_RESULTS.md), including aggregate metrics and historical test-exposure limitations.

## Protocol fixed before evaluation

- Stratified 70/15/15 train/validation/test split with a recorded seed.
- Exclude TARGET and applicant identifier SK_ID_CURR from predictors; reject duplicate applicant identifiers.
- Fit median imputation and categorical encoding on training rows only. Unknown categories are supported.
- Select pruning alpha and classification threshold by validation F1 using the fixed grids in the script. Ties use grid order.
- Preserve the selected training-fitted pipeline; do not refit using validation rows.
- Evaluate the frozen selection once on test rows. Report ROC AUC, average precision, and threshold-dependent F1, alongside a training-prior dummy baseline with fixed threshold 0.5.

Changing the experiment after inspecting test results invalidates its status as a final holdout. A fresh output directory prevents overwriting, but cannot prevent repeated human-driven test-set tuning.

## Run

Install `requirements-tree.txt` for this separate tree experiment, then from the repository root:

```bash
python scripts/train_tree.py --data data/raw/application_train.csv --output runs/tree-v1
python -m unittest discover -s tests -v
```

The output directory must not exist. Dataset access remains subject to the competition conditions. Do not commit applicant-level data, split indices, or fitted artifacts to this public repository. Only load joblib artifacts from trusted sources.

Outputs: `manifest.json` (dataset fingerprint, package versions, source commit, script fingerprint, selection, metrics), `split_indices.npz` (row positions tied to the dataset fingerprint), and `tree.joblib` (fitted preprocessing, estimator, threshold, feature names).

## Validation and limits

Synthetic tests check split separation, repeatability, train-only imputation, unseen categories, persistence, and duplicate rejection. They do not establish performance, calibration, fairness, or external generalization. This protocol makes no claim to reproduce historical feature engineering. The eight tree environment packages are pinned to the recorded run; a clean installation and protocol tests were verified. Package hashes and platform constraints are not locked. The neural-network protocol is still pending.
