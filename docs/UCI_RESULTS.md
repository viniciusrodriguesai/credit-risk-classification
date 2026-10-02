# UCI credit replication: recorded results v1

A new-cohort replication on Yeh (2009), [UCI Default of Credit Card Clients](https://doi.org/10.24432/C55S3H), CC BY 4.0. Different features and a different default horizon: these are **newly trained UCI models**, not validation of the existing Home Credit models or a direct performance comparison with them. Historical Home Credit test exposure remains.

## Prospective evidence

- [Protocol](UCI_PROTOCOL.md) published in commit `24cc1b08f45adb36f0f062a3313d03b294d90917` before the dataset download.
- Runner published in `1bc3f62748d9d6558a7f4c1a1601c20f9cd86165` before training. The local working tree used its exact script bytes while HEAD still referenced the Home Credit base commit; the runtime manifest records that base and the script hash separately. It does not claim a clean checkout at training time.
- [Frozen validation selection](results/uci-v1-frozen.json) published in `51eec1b788ff6b335350588ee9e6cbe228eef599` before test access; its SHA-256 is `863e340648274815e418cdadd726831781f7dfc55278b9456e895557f296bb5f`.
- 30,000 unique client IDs, with hash-based ID-only partition: 21,123 train / 4,500 validation / 4,377 test. Numeric scaling, imputation and categorical encoding fit train only. Training does not read the test file. Final evaluator verifies model/manifest/test/script/protocol hashes and uses an exclusive test-access marker.
- One completed final evaluation. Seeds 42 (training) and 20261002 (bootstrap), one numerical thread. Exact seven-package numerical environment in `requirements-uci.txt`; NumPy 2.2.6.

## Selected before test

Per-family validation-F1 selection: logistic C=0.1, threshold 0.6; tree depth=5 / minimum leaf=50, threshold 0.5; MLP (32,), 50 epochs, threshold 0.6. MLP was the overall validation-selected family. No post-validation refit, test-based threshold search or calibration.

## Held-out metrics

Test positive prevalence: 21.34%. AP is prevalence-dependent; do not compare directly with Home Credit AP. Brier is lower-is-better; other columns are higher-is-better. The dummy uses the **training** prevalence and an all-negative decision at 0.5.

| Model | ROC AUC | AP | F1 | Precision | Recall | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.5000 | 0.2134 | 0.0000 | 0.0000 | 0.0000 | 0.1679 |
| logistic | 0.7174 | 0.4952 | 0.5058 | 0.5534 | 0.4657 | 0.2070 |
| tree | 0.7581 | 0.5066 | 0.5185 | 0.4700 | 0.5782 | 0.1862 |
| mlp | 0.7757 | 0.5241 | 0.5334 | 0.5243 | 0.5428 | 0.1874 |

MLP confusion matrix (actual rows 0/1, predicted columns 0/1): `[[2983, 460], [427, 507]]`. Almost 48% of its positive predictions are false positives; 427 of 934 defaults are missed. Its Brier score is worse than the train-prior dummy: balanced training scores must not be represented as calibrated default probabilities.

## Uncertainty and error analysis

500 paired row-bootstrap replicates, percentile 95% intervals. MLP minus logistic: ROC AUC +0.0583 [0.0459, 0.0724], AP +0.0289 [0.0108, 0.0454], F1 +0.0276 [0.0090, 0.0440]. Tree-minus-logistic AP and F1 intervals include zero. These are conditional intervals for fixed fitted models and this test sample; no training/selection uncertainty, repeated seeds or distribution-shift inference is included.

[Machine-readable test report](results/uci-v1.json) includes every family's confusion matrix, calibration bins, metric intervals and paired differences. The frozen manifest includes all 81 validation candidate/threshold combinations, warnings, data/split hashes, model hashes and versions. Fixed-budget MLP convergence warnings are retained rather than hidden.

## Reproduce

Use Python 3.12 and a separate environment from the historical TensorFlow notebook:

```bash
python -m venv .venv-uci
# Activate your virtual environment, then:
python -m pip install -r requirements-uci.txt
```

Download the official [UCI ZIP](https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip), unzip locally, and pass the XLS file. No client rows or trained models are committed.

```bash
python scripts/evaluate_uci.py prepare --data "data/raw/default of credit card clients.xls" --workspace runs/uci-v1
python scripts/evaluate_uci.py train --workspace runs/uci-v1
```

Review and archive `runs/uci-v1/frozen.json` before accessing test outcomes. Calculate its SHA-256, then pass that exact digest (replace the placeholder with your own run's digest):

```bash
python scripts/evaluate_uci.py evaluate --workspace runs/uci-v1 --frozen-sha <your-frozen-manifest-sha256>
python -m unittest discover -s tests -v
```

## Limits

The historical Taiwan 2005 cohort is a public dataset, not a present-day deployment population. Separation is procedural in one execution environment, without independent custodians. This is a held-out replication using third-party outcome labels, not an audit executed by independent evaluators. Demographic predictors were retained for academic comparability; fairness and temporal/country transfer are untested. No raw data were published. No credit-decision deployment claim is supported.
