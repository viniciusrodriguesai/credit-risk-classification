from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_neural import run


class NeuralProtocolTests(unittest.TestCase):
    def test_frozen_pipeline_split_and_error_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            frame = pd.DataFrame({"SK_ID_CURR": range(100), "TARGET": [0, 1] * 50,
                                  "numeric": np.arange(100), "category": ["A", "B"] * 50})
            frame.to_csv(source, index=False)
            report = run(source, root / "out", architectures=((4,),), epochs=2)
            indices = np.load(root / "out/split_indices.npz")
            self.assertFalse(set(indices['train']) & set(indices['test']))
            self.assertFalse(set(indices['validation']) & set(indices['test']))
            self.assertEqual(sum(map(sum, report['error_analysis']['confusion_matrix'])), 15)
            self.assertEqual(sum(b['rows'] for b in report['error_analysis']['calibration_bins']), 15)
            artifact = joblib.load(root / "out/neural.joblib")
            scaler = artifact['pipeline'].named_steps['preprocess'].named_steps['scale']
            self.assertEqual(scaler.n_samples_seen_, 70)
            sample = frame.drop(columns=['TARGET', 'SK_ID_CURR']).iloc[:1].copy()
            sample['category'] = 'UNSEEN'
            probability = artifact['pipeline'].predict_proba(sample)
            self.assertTrue(np.isfinite(probability).all())
            with self.assertRaises(FileExistsError):
                run(source, root / "out", architectures=((4,),), epochs=2)


if __name__ == '__main__':
    unittest.main()
