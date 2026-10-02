# Credit Risk Classification

An academic comparison of a neural network and a regularized decision tree on the Home Credit Default Risk dataset. Developed by **Gustavo Pereira and Vinicius Mangueira** for a Machine Learning course.

## Project scope

The notebook covers exploratory analysis, missing values, feature engineering, stratified train/validation/test splits, model-specific preprocessing, class imbalance, neural-network training and decision-tree pruning.

## Results and provenance

The saved notebook and separate report describe different tree experiments. The current saved notebook records a tree threshold of 0.70, AUC 0.702620 and F1 0.274071. The separate report uses threshold 0.75, AUC 0.6658 and F1 0.2601. These cannot be presented as one verified run.

See [results provenance](docs/RESULTS_PROVENANCE.md) before citing any metric. Saved outputs are historical evidence; later separately documented tree, MLP and UCI runs do not reproduce that historical notebook comparison. The neural network has higher saved AUC and recall, while the tree has higher saved F1 at its selected operating threshold. No single model dominates every metric.

## Reproduce the notebook

Obtain `application_train.csv` from the [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) under its access conditions and place it in `data/raw/`. The raw dataset is not bundled.

Create and activate a Python virtual environment, then:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab notebooks/credit_risk_classification.ipynb
```

Run the notebook from the repository root or the notebooks directory. Training can be computationally expensive. Dependency versions are currently unpinned; an exact validated training environment is still needed.

## New neural experiment and error analysis

A [fixed-budget MLP run](docs/NEURAL_RESULTS.md) uses the same split as tree-v1 and records AUC **0.7443**, AP **0.2129**, and F1 **0.2887**. Both models have poor probability calibration; confusion counts and calibration bins are published. This is a new separate experiment with historical test-exposure limits, not the old TensorFlow model.

## New-cohort replication with a protected final split

A [preregistered UCI replication](docs/UCI_RESULTS.md) uses 30,000 independently sourced credit-client outcomes, train-only preprocessing, validation-frozen logistic/tree/MLP selection and one held-out test pass. The validation-selected MLP scores AUC **0.7757**, AP **0.5241**, F1 **0.5334** on 4,377 reserved records; paired bootstrap intervals and poor calibration are documented. This supplies new-cohort evidence with new models; it does not erase old Home Credit test exposure or validate those fixed models across domains.

## Current methodological limitations

A separate [tree experiment protocol](docs/TREE_PROTOCOL.md) now provides train-only preprocessing, validation-based selection, a dummy baseline and run metadata. Its protocol checks pass and a [recorded Home Credit run](docs/HOMECREDIT_RESULTS.md) achieves ROC AUC **0.7227**, average precision **0.1972**, and F1 **0.2694**. This is a separate raw-feature tree experiment, with prior dataset/test exposure disclosed; it does not reproduce the historical neural-network comparison.

- The tracked pruning code uses one validation split and 30,000 training rows per candidate, rather than the cross-validation implementation claimed in older documentation.
- Test data are accessed during intermediate diagnostics. The new tree script freezes selection before final evaluation, but the dataset already has historical test exposure; independent validation remains pending.
- Neural-network BCE and tree classification error are different quantities; their generalization gaps should not be directly ranked.
- Fairness, external generalization and probability calibration have not been established.
- Saved model files should not be treated as deployable artifacts without matching preprocessing and run metadata.

This is an academic study and should not be used for automatic credit approval or rejection.

## Documentation

- [Results provenance](docs/RESULTS_PROVENANCE.md): reconciles conflicting artifacts.
- [Portuguese overview](README.pt-BR.md): historical course overview.
- [Historical report](relatorio_final.md): retained with an explicit provenance warning.

## License

[MIT](LICENSE) for the repository's source code; dataset access conditions are separate.
