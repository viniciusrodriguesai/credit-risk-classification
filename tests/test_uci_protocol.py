import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import pandas as pd

spec = importlib.util.spec_from_file_location("evaluate_uci", Path(__file__).resolve().parents[1] / "scripts/evaluate_uci.py")
uci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = uci
spec.loader.exec_module(uci)


class UCIProtocolTests(unittest.TestCase):
    def test_partitions_are_id_only_and_cover_all_buckets(self):
        labels = [uci.partition(i) for i in range(1, 30001)]
        self.assertEqual(set(labels), {"train", "validation", "test"})
        self.assertEqual(labels, [uci.partition(i) for i in range(1, 30001)])
        self.assertTrue(all(labels.count(name) > 4000 for name in set(labels)))

    def test_preprocessing_does_not_learn_validation_outlier_or_unknown_category(self):
        train = pd.DataFrame({"SEX": [1, 2], "EDUCATION": [1, 1], "MARRIAGE": [1, 1], "AGE": [20., 30.]})
        validation = train.iloc[:1].copy()
        validation["AGE"] = 10000.
        validation["SEX"] = 99
        transform = uci.preprocessing(train, True)
        transform.fit(train)
        self.assertEqual(transform.named_transformers_["numeric"].named_steps["scale"].mean_[0], 25.)
        transform.transform(validation)
        self.assertNotIn(99, transform.named_transformers_["categorical"].named_steps["onehot"].categories_[0])

    def test_features_remove_identity_and_outcome(self):
        frame = pd.DataFrame({"ID": [1], uci.TARGET: [1], "AGE": [30]})
        x, y = uci.features(frame)
        self.assertEqual(list(x), ["AGE"])
        self.assertEqual(y.tolist(), [1])

    def test_changed_model_or_manifest_cannot_be_evaluated(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder)
            joblib.dump({"fixed": 1}, workspace / "tree.joblib")
            (workspace / "test.csv").write_text("unopened outcome\n")
            frozen = {"script_sha256": uci.sha(uci.__file__),
                      "protocol_sha256": uci.sha(uci.ROOT / "docs/UCI_PROTOCOL.md"),
                      "selected": {"tree": {"model_sha256": uci.sha(workspace / "tree.joblib")}},
                      "data_manifest": {"split_files": {"test": uci.sha(workspace / "test.csv")}}}
            uci.dump(workspace / "frozen.json", frozen)
            seal = uci.sha(workspace / "frozen.json")
            uci.verify_frozen(workspace, seal)
            joblib.dump({"fixed": 2}, workspace / "tree.joblib")
            with self.assertRaisesRegex(ValueError, "model was modified"):
                uci.verify_frozen(workspace, seal)
            with self.assertRaisesRegex(ValueError, "manifest"):
                uci.verify_frozen(workspace, "wrong hash")


if __name__ == "__main__":
    unittest.main()
