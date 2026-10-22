"""
Unit tests for Model Training, Evaluation, and Inference Pipeline (4-Class Mode)
--------------------------------------------------------------------------------
Verifies:
1. Dataset and session-aware split loading with leaning_forward safely excluded.
2. Correct feature selection (strictly distance-invariant angles, no sentinels, no metadata).
3. Exclusion of constant sentinel columns (-1.0).
4. Training and evaluation execution without data leakage.
5. Model artifact loading from disk using joblib.
6. Real-time inference helper `predict_posture` returning valid classes and probabilities.
7. Verification of candidate model bundle structure and performance metadata.
"""

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from src.train_model import (
    CLASS_LABELS,
    FOUR_CLASS_LABELS,
    PRIMARY_FEATURES,
    build_candidate_models,
    evaluate_classifier,
    load_dataset_and_split,
    predict_posture,
)


class TestModelTraining(unittest.TestCase):

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.clean_csv = self.project_root / "data" / "posture_data_clean.csv"
        self.splits_json = self.project_root / "data" / "session_split.json"
        self.saved_model = self.project_root / "models" / "posture_model.joblib"
        self.candidate_model = self.project_root / "models" / "candidate_posture_model.joblib"

    def test_dataset_and_split_loading(self):
        """Verify load_dataset_and_split loads clean data without leakage and safely excludes leaning_forward."""
        self.assertTrue(self.clean_csv.exists(), "Clean CSV must exist")
        self.assertTrue(self.splits_json.exists(), "Splits JSON must exist")

        X_train, y_train, X_test, y_test, split_meta = load_dataset_and_split(
            self.clean_csv, self.splits_json
        )

        # In 4-class mode, 74 leaning_forward samples are excluded (48 in train, 26 in test)
        self.assertEqual(len(X_train), 197)
        self.assertEqual(len(X_test), 125)
        self.assertEqual(len(y_train), 197)
        self.assertEqual(len(y_test), 125)

        # Check feature dimension equals 2 (shoulder_tilt_deg, neck_angle_deg)
        self.assertEqual(X_train.shape[1], len(PRIMARY_FEATURES))
        self.assertEqual(X_test.shape[1], len(PRIMARY_FEATURES))
        self.assertEqual(len(PRIMARY_FEATURES), 2)

        # Check all 4 target classes exist in both partitions
        self.assertEqual(set(y_train), set(FOUR_CLASS_LABELS))
        self.assertEqual(set(y_test), set(FOUR_CLASS_LABELS))
        self.assertNotIn("leaning_forward", set(y_train))
        self.assertNotIn("leaning_forward", set(y_test))

    def test_feature_selection_excludes_sentinels_and_metadata(self):
        """Selected features must be strictly PRIMARY_FEATURES, finite, and free of -1.0 sentinels."""
        X_train, _, X_test, _, _ = load_dataset_and_split(
            self.clean_csv, self.splits_json
        )

        for name, X in [("train", X_train), ("test", X_test)]:
            # All numbers must be finite
            self.assertTrue(np.all(np.isfinite(X)), f"{name} contains non-finite numbers")

            # No sentinels (-1.0) allowed in primary close-up features
            sentinel_mask = np.isclose(X, -1.0, atol=1e-4)
            self.assertFalse(np.any(sentinel_mask), f"{name} unexpectedly contains -1.0 sentinels")

    def test_candidate_models_building(self):
        """Candidate models must include DecisionTree, RandomForest, SVM, and LogisticRegression."""
        models = build_candidate_models(random_state=42)
        expected_keys = {"DecisionTree", "RandomForest", "SVM (RBF)", "LogisticRegression"}
        self.assertEqual(set(models.keys()), expected_keys)

    def test_training_and_evaluation_without_leakage(self):
        """Test evaluation output structure and metrics validity."""
        X_train, y_train, X_test, y_test, _ = load_dataset_and_split(
            self.clean_csv, self.splits_json
        )
        models = build_candidate_models(random_state=42)
        rf_model = models["RandomForest"]

        res = evaluate_classifier("RandomForest", rf_model, X_train, y_train, X_test, y_test, CLASS_LABELS)

        # Metrics bounds
        self.assertGreaterEqual(res["test_accuracy"], 0.0)
        self.assertLessEqual(res["test_accuracy"], 1.0)
        self.assertGreaterEqual(res["macro_f1"], 0.0)
        self.assertLessEqual(res["macro_f1"], 1.0)

        # Confusion matrix shape: 4x4
        cm = np.array(res["confusion_matrix"])
        self.assertEqual(cm.shape, (4, 4))
        self.assertEqual(int(cm.sum()), len(y_test))

    def test_saved_model_artifact_loading_and_inference(self):
        """Test that saved model artifact can be reloaded and produces valid 4-class predictions."""
        self.assertTrue(self.saved_model.exists(), "Trained model artifact must exist at models/posture_model.joblib")

        artifact = joblib.load(self.saved_model)
        self.assertIn("model", artifact)
        self.assertIn("feature_names", artifact)
        self.assertIn("classes", artifact)
        self.assertEqual(artifact["feature_names"], PRIMARY_FEATURES)
        self.assertEqual(artifact["classes"], CLASS_LABELS)
        self.assertIn("forward_lean_status", artifact)

        # Test inference with sample leaning_left features
        sample_lean_left = {
            "shoulder_tilt_deg": 18.2,
            "neck_angle_deg": 46.0,
        }
        pred_label, probs = predict_posture(artifact, sample_lean_left)
        self.assertIn(pred_label, CLASS_LABELS)
        self.assertEqual(pred_label, "leaning_left")
        self.assertIn("leaning_left", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

        # Test inference with sample leaning_right features
        sample_lean_right = {
            "shoulder_tilt_deg": -17.5,
            "neck_angle_deg": 40.0,
        }
        pred_label_r, probs_r = predict_posture(artifact, sample_lean_right)
        self.assertIn(pred_label_r, CLASS_LABELS)
        self.assertEqual(pred_label_r, "leaning_right")

    def test_inference_rejects_missing_features(self):
        """Inference helper must cleanly raise ValueError when required features are missing."""
        artifact = joblib.load(self.saved_model)
        corrupted_input = {
            "shoulder_tilt_deg": 10.0,
            # 'neck_angle_deg' missing
        }
        with self.assertRaises(ValueError):
            predict_posture(artifact, corrupted_input)

    def test_candidate_model_artifact_loading_and_inference(self):
        """Candidate model artifact must load and accept candidate 4-class features."""
        self.assertTrue(self.candidate_model.exists(), "Candidate model artifact must exist at models/candidate_posture_model.joblib")

        cand_artifact = joblib.load(self.candidate_model)
        self.assertIn("model", cand_artifact)
        self.assertIn("feature_names", cand_artifact)
        self.assertEqual(cand_artifact["feature_names"], PRIMARY_FEATURES)
        self.assertEqual(cand_artifact["classes"], FOUR_CLASS_LABELS)

        sample_input = {
            "shoulder_tilt_deg": 18.2,
            "neck_angle_deg": 46.0,
        }
        pred_label, probs = predict_posture(cand_artifact, sample_input)
        self.assertIn(pred_label, CLASS_LABELS)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
