"""Separate, reproducible tree experiment; historical notebook results stay intact."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def make_pipeline(frame, alpha, seed):
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", DecisionTreeClassifier(ccp_alpha=alpha, class_weight="balanced",
                                        random_state=seed)),
    ])


def metrics(y, probability, threshold):
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, probability >= threshold, zero_division=0)),
        "threshold": float(threshold),
        "rows": len(y),
        "positive_rows": int(np.sum(y)),
    }


def run(csv_path, output, seed=42):
    output = Path(output)
    # A unique directory prevents accidental overwriting of an earlier run.
    output.mkdir(parents=True, exist_ok=False)
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    if "TARGET" not in frame or frame.TARGET.isna().any():
        raise ValueError("TARGET must be present and contain no missing values")
    if set(frame.TARGET.unique()) != {0, 1}:
        raise ValueError("TARGET must contain both binary classes 0 and 1")
    if "SK_ID_CURR" in frame and frame.SK_ID_CURR.duplicated().any():
        raise ValueError("Duplicate SK_ID_CURR values could leak across splits")
    y = frame.TARGET.astype(int)
    x = frame.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    if x.empty:
        raise ValueError("At least one feature is required")
    train, remainder = train_test_split(np.arange(len(frame)), test_size=.30,
                                       stratify=y, random_state=seed)
    validation, test = train_test_split(remainder, test_size=.50,
                                       stratify=y.iloc[remainder], random_state=seed)
    candidates = []
    best = None
    # Fixed candidate grid and validation F1 objective declared before test access.
    for alpha in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
        model = make_pipeline(x.iloc[train], alpha, seed)
        model.fit(x.iloc[train], y.iloc[train])
        probability = model.predict_proba(x.iloc[validation])[:, 1]
        for threshold in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
            score = float(f1_score(y.iloc[validation], probability >= threshold,
                                   zero_division=0))
            candidates.append({"ccp_alpha": alpha, "threshold": threshold,
                               "validation_f1": score})
            # Stable ties prefer the first alpha/threshold in the declared grid.
            if best is None or score > best[0]:
                best = (score, alpha, threshold, model)
    _, alpha, threshold, selected = best
    # No refit: preprocessing and estimator remain fitted on training only.
    baseline = DummyClassifier(strategy="prior").fit(
        np.zeros((len(train), 1)), y.iloc[train])
    final_probability = selected.predict_proba(x.iloc[test])[:, 1]
    baseline_probability = baseline.predict_proba(np.zeros((len(test), 1)))[:, 1]
    baseline_threshold = .5  # Predeclared; not tuned on the test set.
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    digest = hashlib.sha256()
    with csv_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "experiment": "tree-validation-f1-v1", "seed": seed,
        "dataset_sha256": digest.hexdigest(), "source_commit": commit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "joblib")},
        "split_sizes": {"train": len(train), "validation": len(validation),
                        "test": len(test)},
        "selected": {"ccp_alpha": alpha, "threshold": threshold},
        "validation_candidates": candidates,
        "test": metrics(y.iloc[test], final_probability, threshold),
        "baseline_test": metrics(y.iloc[test], baseline_probability, baseline_threshold),
    }
    np.savez_compressed(output / "split_indices.npz", train=train,
                        validation=validation, test=test)
    joblib.dump({"pipeline": selected, "threshold": threshold,
                 "feature_columns": x.columns.tolist()}, output / "tree.joblib")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data, args.output, args.seed)
