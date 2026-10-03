"""Unit tests for reusable preprocessing utilities."""
import unittest
import pandas as pd
from src.data_preprocessing import split_data
from src.train_model import build_models, regression_metrics


class PreprocessingTests(unittest.TestCase):
    def test_split_sizes_and_reproducibility(self):
        X = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        y = pd.Series(range(100))
        split1 = split_data(X, y, test_size=0.2, random_state=42)
        split2 = split_data(X, y, test_size=0.2, random_state=42)
        self.assertEqual(len(split1[0]), 80)
        self.assertEqual(len(split1[1]), 20)
        self.assertTrue(split1[0].equals(split2[0]))
        self.assertTrue(split1[1].equals(split2[1]))

    def test_models_and_metrics(self):
        models = build_models()
        self.assertEqual(set(models), {"Linear Regression", "Decision Tree", "Random Forest"})
        metrics = regression_metrics([1, 2, 3], [1, 2, 2])
        self.assertAlmostEqual(metrics["MAE"], 1 / 3)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2", metrics)


if __name__ == "__main__":
    unittest.main()
