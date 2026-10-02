"""Fixed-budget MLP protocol, separate from the historical TensorFlow notebook."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from train_tree import make_pipeline, metrics


def error_analysis(y, probability, threshold):
    prediction = probability >= threshold
    bins = []
    for lower in np.arange(0, 1, .1):
        upper = lower + .1
        mask = (probability >= lower) & ((probability < upper) if upper < .999 else (probability <= 1))
        bins.append({"lower": round(float(lower), 1), "upper": round(float(upper), 1),
                     "rows": int(mask.sum()),
                     "mean_probability": float(probability[mask].mean()) if mask.any() else None,
                     "positive_rate": float(np.asarray(y)[mask].mean()) if mask.any() else None})
    return {"confusion_matrix_labels": [0, 1],
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "brier_score": float(brier_score_loss(y, probability)),
            "calibration_bins": bins}


def run(data, output, seed=42, *, architectures=((32,), (64, 32)), epochs=30):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(data)
    if "TARGET" not in frame or frame.TARGET.isna().any() or set(frame.TARGET.unique()) != {0, 1}:
        raise ValueError("TARGET must contain both binary classes without missing values")
    if "SK_ID_CURR" in frame and frame.SK_ID_CURR.duplicated().any():
        raise ValueError("Duplicate applicant identifiers")
    y = frame.TARGET.astype(int)
    x = frame.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    train, rest = train_test_split(np.arange(len(y)), test_size=.30, stratify=y, random_state=seed)
    validation, test = train_test_split(rest, test_size=.5, stratify=y.iloc[rest], random_state=seed)
    preprocessing = make_pipeline(x.iloc[train], 0, seed).named_steps["preprocess"]
    # Scaling and imputation fit training only, before any candidate selection.
    preprocessing = Pipeline([("columns", preprocessing), ("scale", StandardScaler(with_mean=False))])
    train_x = preprocessing.fit_transform(x.iloc[train])
    validation_x = preprocessing.transform(x.iloc[validation])
    candidates = []
    best = None
    with threadpool_limits(limits=1):
        for architecture in architectures:
            model = MLPClassifier(hidden_layer_sizes=architecture, alpha=.0001,
                                  batch_size=512, learning_rate_init=.001,
                                  max_iter=epochs, tol=0.0, n_iter_no_change=epochs + 1,
                                  early_stopping=False, random_state=seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train_x, y.iloc[train], sample_weight=compute_sample_weight("balanced", y.iloc[train]))
            probability = model.predict_proba(validation_x)[:, 1]
            for threshold in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
                result = metrics(y.iloc[validation], probability, threshold)
                candidates.append({"architecture": list(architecture), "threshold": threshold,
                                   "validation": result, "iterations": model.n_iter_,
                                   "warnings": sorted({str(w.message) for w in caught})})
                if best is None or result["f1"] > best[0]:
                    best = (result["f1"], architecture, threshold, model)
            print(f"Finished architecture {architecture}", flush=True)
        _, architecture, threshold, selected = best
        probability = selected.predict_proba(preprocessing.transform(x.iloc[test]))[:, 1]
    report = {"protocol": "raw-feature-balanced-mlp-v1", "seed": seed,
              "python": platform.python_version(), "epochs": epochs,
              "dataset_sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "tree_script_sha256": hashlib.sha256(Path(__file__).with_name("train_tree.py").read_bytes()).hexdigest(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
              "versions": {p: importlib.metadata.version(p) for p in
                           ("numpy", "pandas", "scikit-learn", "joblib", "scipy", "threadpoolctl")},
              "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
              "selected": {"architecture": list(architecture), "threshold": threshold},
              "validation_candidates": candidates,
              "test": metrics(y.iloc[test], probability, threshold),
              "error_analysis": error_analysis(y.iloc[test], probability, threshold),
              "loss_curve": selected.loss_curve_}
    np.savez_compressed(output / "split_indices.npz", train=train, validation=validation, test=test)
    joblib.dump({"pipeline": Pipeline([("preprocess", preprocessing), ("model", selected)]),
                 "threshold": threshold, "feature_columns": x.columns.tolist()}, output / "neural.joblib")
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
