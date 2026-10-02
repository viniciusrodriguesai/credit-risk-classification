# UCI credit replication protocol v1

Recorded before downloading the data or observing this experiment's outcomes.
This is a new-cohort replication with newly trained models, **not external validation of the existing Home Credit model**. Different features and default horizons prevent direct transfer. Historical Home Credit test exposure remains disclosed.

## Source and partition

Yeh (2009), Default of Credit Card Clients, UCI dataset 350, DOI https://doi.org/10.24432/C55S3H (CC BY 4.0). All 30,000 records; no row filtering. Exclude ID and target from predictors. SEX, EDUCATION and MARRIAGE are categorical; other predictors numeric. This historical Taiwan cohort is not representative of current international credit applicants. Demographic variables are included for academic comparability; no fairness or production readiness claim.

Partition by ID alone: SHA-256 of `uci-credit-v1:42:<integer ID>`; first 8 hexadecimal digits modulo 100: 0–69 train, 70–84 validation, 85–99 test. No target-dependent splitting or repeated seed search. Prepare separate files; training must read only train/validation. Test is accessed only after selection and model hashes are frozen. This is procedural separation in one execution environment, not third-party custody or a permanently unseen public benchmark.

## Fixed budget

Train-only median numeric imputation, most-frequent categorical imputation, one-hot encoding (unknown ignored), StandardScaler for logistic and MLP numeric features.

- Dummy: train prevalence, with all-negative labels at 0.5.
- Logistic: C = 0.1, 1, 10; balanced weights; lbfgs, max_iter 2000.
- Tree: max_depth = 3, 5, 8, 12; min_samples_leaf 50; balanced weights.
- MLP: (32,) or (64,32), Adam, alpha 0.0001, batch 512, learning rate 0.001, 50 epochs, tol 0, n_iter_no_change 51, no early stopping, balanced sample weights.

Seed 42, one numerical thread. Per family, choose the candidate and threshold with highest validation F1 from 0.1,...,0.9; strict improvement only, preserving listed order on ties. Do not refit after validation. Freeze all three selected families and a validation-selected overall family before final testing. No calibration fitted on test.

## Final report

One test pass: ROC AUC, AP, F1, precision, recall, confusion matrix, Brier, calibration bins; train-prevalence dummy. 500 paired row-bootstrap replicates, seed 20261002, percentile 95% intervals for AUC/AP/F1 and differences against logistic. These intervals do not include model-selection uncertainty, repeat-training variation or distribution shift. Record data/script/protocol SHA-256, source commit, versions, warnings, split hashes and frozen manifest. Publish aggregate results only. Report unfavorable findings; no post-test tuning under v1.
