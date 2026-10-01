"""Synthetic checks of the protocol, not evidence of Home Credit performance."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "train_tree", Path(__file__).resolve().parents[1] / "scripts/train_tree.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProtocolTests(unittest.TestCase):
    def test_repeatability_splits_and_serialized_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({
                "SK_ID_CURR": np.arange(200), "TARGET": [0, 1] * 100,
                "amount": np.arange(200, dtype=float), "category": ["a", "b"] * 100,
            })
            frame.loc[::7, "amount"] = np.nan
            frame.to_csv(root / "data.csv", index=False)
            first = module.run(root / "data.csv", root / "first")
            second = module.run(root / "data.csv", root / "second")
            self.assertEqual(first, second)
            split = np.load(root / "first/split_indices.npz")
            self.assertEqual(len(set(np.concatenate(list(split.values())))), len(frame))
            for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
                self.assertFalse(set(split[a]) & set(split[b]))
            artifact = joblib.load(root / "first/tree.joblib")
            self.assertNotIn("SK_ID_CURR", artifact["feature_columns"])
            imputer = artifact["pipeline"].named_steps["preprocess"].named_transformers_["numeric"]
            self.assertAlmostEqual(imputer.statistics_[0], frame.iloc[split["train"]].amount.median())
            unseen = pd.DataFrame({"amount": [np.nan], "category": ["unseen"]})
            self.assertEqual(artifact["pipeline"].predict_proba(unseen).shape, (1, 2))
            with self.assertRaises(FileExistsError):
                module.run(root / "data.csv", root / "first")

    def test_duplicate_applicants_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame({"SK_ID_CURR": [1, 1], "TARGET": [0, 1], "x": [1, 2]}).to_csv(root / "data.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                module.run(root / "data.csv", root / "output")


if __name__ == "__main__":
    unittest.main()
