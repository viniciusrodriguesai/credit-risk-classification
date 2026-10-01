# Recorded Home Credit tree experiment

Executed on 2026-10-01 with the user-supplied `application_train.csv`: 307,511 rows, stratified 215,257 / 46,127 / 46,127 train/validation/test split, seed 42. The predictor pipeline excludes TARGET and SK_ID_CURR and fits preprocessing on training only.

The fixed validation search selected pruning alpha **0.0001** and threshold **0.70**. The selected fitted model was frozen before the final test prediction; no new test-based search or refit was performed in this run.

| Final test metric | Selected tree | Prior dummy (threshold 0.50) |
| --- | ---: | ---: |
| ROC AUC | 0.722700 | 0.500000 |
| Average precision | 0.197235 | 0.080734 |
| F1 | 0.269388 | 0.000000 |

The 46,127 test rows contain 3,724 positive labels (8.0734%). Average precision exceeds prevalence, while F1 remains modest. The dummy predicts no positive labels at 0.50, so its zero F1 is expected and is not a competitive tuned classifier. These results do not establish deployability, calibration, fairness, or business value.

## Reproduce

```bash
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements-tree.txt
python scripts/train_tree.py --data data/raw/application_train.csv --output runs/tree-v1
python -m unittest discover -s tests -v
```

Python 3.12.14 and all eight packages in the tree dependency environment are recorded in [the supplemental environment record](results/tree-environment.json) and pinned in `requirements-tree.txt`. A clean installation and the two protocol tests were verified separately; the full real-data training was performed in the recorded original environment. Package hashes and platform constraints are not locked. The historical notebook requires a different, larger environment.

The [unaltered run manifest](results/homecredit-tree-v1.json) contains all 45 validation candidates, actual package versions, dataset and script SHA-256 fingerprints and aggregate test metrics. Raw rows, split indices and fitted applicant-level artifacts are not published. Obtain the source data under the competition's access conditions.

## Provenance and limitations

The execution occurred in a local worktree based on `33def9411cee898c5ea6e484094ddb8474b021e4` with the new script present as an uncommitted file. That is why the manifest records that base commit. Its script SHA-256 was verified against the exact tracked script in PR #3 commit `263f450f3eff6160732ecbc52048b90c5ffd3493`; the original manifest has not been rewritten.

This dataset was already used in the historical academic study, including intermediate test inspection. The current run follows a frozen selection procedure, but its test set must not be marketed as an entirely unseen external or prospective validation set. Same-seed splits may overlap earlier experiments. Future quality claims require a separately protected evaluation set and cannot be obtained by tuning after inspecting these results.

This separate experiment uses the raw application table; it does not reproduce historical feature engineering or the neural network. Historical and new scores are not controlled head-to-head comparisons. The neural-network protocol and independent evaluation remain pending.
