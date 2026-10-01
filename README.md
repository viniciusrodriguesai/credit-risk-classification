# Credit Risk Classification

An academic comparison of a neural network and a regularized decision tree on the Home Credit Default Risk dataset. Developed by **Gustavo Pereira and Vinicius Mangueira** for a Machine Learning course.

## Project scope

The notebook covers exploratory analysis, missing values, feature engineering, stratified train/validation/test splits, model-specific preprocessing, class imbalance, neural-network training and decision-tree pruning.

## Results and provenance

The saved notebook and separate report describe different tree experiments. The current saved notebook records a tree threshold of 0.70, AUC 0.702620 and F1 0.274071. The separate report uses threshold 0.75, AUC 0.6658 and F1 0.2601. These cannot be presented as one verified run.

See [results provenance](docs/RESULTS_PROVENANCE.md) before citing any metric. Saved outputs are historical evidence; the full training was not rerun during the portfolio audit. The neural network has higher saved AUC and recall, while the tree has higher saved F1 at its selected operating threshold. No single model dominates every metric.

## Reproduce the notebook

Obtain `application_train.csv` from the [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) under its access conditions and place it in `data/raw/`. The raw dataset is not bundled.

Create and activate a Python virtual environment, then:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab notebooks/credit_risk_classification.ipynb
```

Run the notebook from the repository root or the notebooks directory. Training can be computationally expensive. Dependency versions are currently unpinned; an exact validated training environment is still needed.

## Current methodological limitations

- The tracked pruning code uses one validation split and 30,000 training rows per candidate, rather than the cross-validation implementation claimed in older documentation.
- Test data are accessed during intermediate diagnostics. A strict, locked final-evaluation protocol remains to be implemented and rerun.
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
