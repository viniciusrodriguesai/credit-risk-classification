"""Three-stage UCI replication: prepare, train/freeze, then evaluate once."""
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

TARGET = "default payment next month"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
THRESHOLDS = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def partition(identifier):
    bucket = int(hashlib.sha256(f"uci-credit-v1:42:{int(identifier)}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def prepare(data, workspace):
    workspace.mkdir(parents=True, exist_ok=False)
    frame = pd.read_excel(data, header=1)
    if len(frame) != 30000 or frame.ID.duplicated().any():
        raise ValueError("Expected 30,000 unique IDs")
    if set(frame[TARGET].unique()) != {0, 1} or frame[TARGET].isna().any():
        raise ValueError("Invalid binary target")
    if len(frame.columns) != 25 or not set(CATEGORICAL).issubset(frame.columns):
        raise ValueError("Unexpected UCI schema")
    assignment = frame.ID.map(partition)
    manifest = {"dataset_sha256": sha(data), "split_files": {}, "split_sizes": {}}
    for name in ["train", "validation", "test"]:
        selected = frame.loc[assignment == name]
        file = workspace / f"{name}.csv"
        selected.to_csv(file, index=False)
        manifest["split_files"][name] = sha(file)
        manifest["split_sizes"][name] = len(selected)
    dump(workspace / "data-manifest.json", manifest)
    return manifest


def features(frame):
    return frame.drop(columns=["ID", TARGET]), frame[TARGET].to_numpy(dtype=int)


def preprocessing(x, scaled):
    numeric = [c for c in x.columns if c not in CATEGORICAL]
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scaled:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL),
    ])


def metrics(y, probability, threshold):
    predicted = probability >= threshold
    return {"roc_auc": float(roc_auc_score(y, probability)),
            "average_precision": float(average_precision_score(y, probability)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "brier": float(brier_score_loss(y, probability)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def candidates():
    for c in [.1, 1., 10.]:
        yield "logistic", {"C": c}, LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42)
    for depth in [3, 5, 8, 12]:
        yield "tree", {"max_depth": depth, "min_samples_leaf": 50}, DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=50, class_weight="balanced", random_state=42)
    for architecture in [(32,), (64, 32)]:
        yield "mlp", {"architecture": list(architecture), "epochs": 50}, MLPClassifier(
            hidden_layer_sizes=architecture, alpha=.0001, batch_size=512, learning_rate_init=.001,
            max_iter=50, tol=0., n_iter_no_change=51, early_stopping=False, random_state=42)


def train(workspace):
    if (workspace / "frozen.json").exists():
        raise FileExistsError("Selection is already frozen; no overwrite")
    manifest = json.loads((workspace / "data-manifest.json").read_text())
    # Deliberately no test.csv access in this stage.
    for name in ["train", "validation"]:
        if sha(workspace / f"{name}.csv") != manifest["split_files"][name]:
            raise ValueError("Changed split")
    train_x, train_y = features(pd.read_csv(workspace / "train.csv"))
    val_x, val_y = features(pd.read_csv(workspace / "validation.csv"))
    best = {}
    records = []
    with threadpool_limits(limits=1):
        for family, params, model in candidates():
            pipeline = Pipeline([("preprocess", preprocessing(train_x, family != "tree")), ("model", model)])
            kwargs = {"model__sample_weight": compute_sample_weight("balanced", train_y)} if family == "mlp" else {}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(train_x, train_y, **kwargs)
            probability = pipeline.predict_proba(val_x)[:, 1]
            for threshold in THRESHOLDS:
                result = metrics(val_y, probability, threshold)
                records.append({"family": family, "parameters": params, "threshold": threshold,
                                "validation": result, "warnings": sorted({str(w.message) for w in caught})})
                if family not in best or result["f1"] > best[family]["validation"]["f1"]:
                    best[family] = {"parameters": params, "threshold": threshold, "validation": result}
                    joblib.dump(pipeline, workspace / f"{family}.joblib")
            print(f"Completed {family} {params}", flush=True)
    for family in best:
        best[family]["model_sha256"] = sha(workspace / f"{family}.joblib")
    overall = max(best, key=lambda key: best[key]["validation"]["f1"])
    frozen = {"protocol": "uci-credit-v1", "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "docs/UCI_PROTOCOL.md"),
        "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in
        ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl", "xlrd"]},
        "data_manifest": manifest, "train_prevalence": float(train_y.mean()),
        "selected": best, "overall_selected_family": overall, "validation_candidates": records}
    dump(workspace / "frozen.json", frozen)
    return frozen


def verify_frozen(workspace, expected_sha):
    if sha(workspace / "frozen.json") != expected_sha:
        raise ValueError("Frozen manifest does not match preregistered hash")
    frozen = json.loads((workspace / "frozen.json").read_text())
    if frozen["script_sha256"] != sha(__file__) or frozen["protocol_sha256"] != sha(ROOT / "docs/UCI_PROTOCOL.md"):
        raise ValueError("Protocol or evaluator changed after selection")
    for name, entry in frozen["selected"].items():
        if sha(workspace / f"{name}.joblib") != entry["model_sha256"]:
            raise ValueError("Frozen model was modified")
    if sha(workspace / "test.csv") != frozen["data_manifest"]["split_files"]["test"]:
        raise ValueError("Test file changed")
    return frozen


def evaluate(workspace, expected_sha):
    # Consume the single-run marker before reading outcomes; even a failed final pass is logged.
    frozen = verify_frozen(workspace, expected_sha)
    with (workspace / "test-accessed.txt").open("x") as handle:
        handle.write("Final evaluation started; do not reuse as unseen data.\n")
    test_x, test_y = features(pd.read_csv(workspace / "test.csv"))
    predictions = {"dummy": np.full(len(test_y), frozen["train_prevalence"])}
    thresholds = {"dummy": .5}
    with threadpool_limits(limits=1):
        for family, entry in frozen["selected"].items():
            predictions[family] = joblib.load(workspace / f"{family}.joblib").predict_proba(test_x)[:, 1]
            thresholds[family] = entry["threshold"]
    result = {name: metrics(test_y, prob, thresholds[name]) for name, prob in predictions.items()}
    for name, probability in predictions.items():
        bins = []
        for low in np.arange(0, 1, .1):
            mask = (probability >= low) & ((probability < low + .1) if low < .85 else (probability <= 1))
            bins.append({"lower": round(float(low), 1), "rows": int(mask.sum()),
                         "mean_probability": float(probability[mask].mean()) if mask.any() else None,
                         "positive_rate": float(test_y[mask].mean()) if mask.any() else None})
        result[name]["calibration_bins"] = bins
    rng = np.random.default_rng(20261002)
    samples = {name: {metric: [] for metric in ["roc_auc", "average_precision", "f1"]} for name in predictions}
    for _ in range(500):
        indices = rng.integers(0, len(test_y), size=len(test_y))
        if len(np.unique(test_y[indices])) < 2:
            continue
        for name, probability in predictions.items():
            values = metrics(test_y[indices], probability[indices], thresholds[name])
            for metric in samples[name]:
                samples[name][metric].append(values[metric])
    intervals = {name: {metric: np.quantile(values, [.025, .975]).tolist() for metric, values in series.items()}
                 for name, series in samples.items()}
    differences = {name: {metric: {"point": result[name][metric] - result["logistic"][metric],
        "ci95": np.quantile(np.array(values) - np.array(samples["logistic"][metric]), [.025, .975]).tolist()}
        for metric, values in series.items()} for name, series in samples.items() if name != "logistic"}
    report = {"protocol": "uci-credit-v1", "frozen_manifest_sha256": expected_sha,
              "test_rows": len(test_y), "test_prevalence": float(test_y.mean()),
              "overall_selected_family": frozen["overall_selected_family"],
              "metrics": result, "bootstrap_replicates": len(samples["logistic"]["f1"]),
              "bootstrap_ci95": intervals, "paired_differences_vs_logistic": differences}
    dump(workspace / "test-results.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--frozen-sha")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.data, args.workspace)
    elif args.stage == "train":
        train(args.workspace)
    elif not args.frozen_sha:
        parser.error("Evaluation requires --frozen-sha from the registered frozen manifest")
    else:
        evaluate(args.workspace, args.frozen_sha)
