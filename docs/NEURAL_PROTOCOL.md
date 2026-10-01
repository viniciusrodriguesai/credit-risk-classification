# Separate neural-network protocol

This new scikit-learn MLP experiment is not a reproduction of the historical TensorFlow model. Its choices are fixed before its test prediction, but the dataset already has historical test exposure; no independent external evaluation is claimed.

- Same stratified 70/15/15 split and seed 42 as tree-v1.
- Exclude TARGET and SK_ID_CURR; reject duplicate identifiers.
- Fit median/category imputation, one-hot encoding and feature scaling on training only.
- Two predeclared architectures: (32,) and (64,32), Adam learning rate 0.001, L2 alpha 0.0001, batch 512, 30 epochs, no validation-based early stopping or refit.
- Balanced sample weights derived from training labels. Scores are not assumed calibrated probabilities.
- Select architecture and threshold (0.1 through 0.9) by validation F1, using grid order for ties.
- Predict test once for the frozen selection. Report AUC/AP/F1, precision/recall, confusion counts, Brier score, calibration bins, validation candidates and loss curve.
- Preserve matched preprocessing/model, split indices, source fingerprints and package versions. Publish only aggregates.

```bash
python -m pip install -r requirements-tree.txt
python scripts/train_neural.py --data data/raw/application_train.csv --output runs/neural-v1
python -m unittest discover -s tests -v
```

The output must not exist. Runs are private and ignored. Use only trusted joblib artifacts. Fixed-budget convergence warnings are recorded rather than hidden. There is no tuning after inspecting this run's test results. Package and Python/platform dependence may affect exact training reproducibility.

[MLP implementation reference](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
