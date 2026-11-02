"""
Unit Tests for MultiPosture Data Preprocessing & Model Evaluation Pipeline
--------------------------------------------------------------------------
Validates:
1. MultiPosture geometric feature extraction math and coordinate calculations.
2. Participant-aware partitioning and zero-leakage verification.
3. Unsupported class handling ('slouching' strictly marked as unsupported).
4. Candidate model bundle structure, metadata, and inference compatibility.
5. Non-interference: verifies active model and existing datasets remain unmodified.
"""

import json
from pathlib import Path
import sys
import unittest

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_multiposture import (
    calculate_head_to_shoulder_ratio_multiposture,
    calculate_neck_angle_multiposture,
    calculate_shoulder_tilt_multiposture,
    calculate_shoulder_to_ear_ratio_multiposture,
    calculate_torso_angle_multiposture,
    COMPATIBLE_LABEL_MAP,
    UNSUPPORTED_POSTUREGUARD_CLASSES,
)
from src.realtime_posture import PosturePredictor


class TestMultiPostureMath(unittest.TestCase):
    """Tests geometric math functions on hip-centered MultiPosture coordinates."""

    def test_shoulder_tilt_horizontal(self):
        # In hip-centered space: right shoulder x < 0, left shoulder x > 0
        # Horizontal: right shoulder y == left shoulder y
        tilt = calculate_shoulder_tilt_multiposture(
            ls_x=0.15, ls_y=-0.40, rs_x=-0.15, rs_y=-0.40
        )
        self.assertAlmostEqual(tilt, 0.0, places=1)

    def test_shoulder_tilt_lateral_leans(self):
        # Leaning left: left shoulder drops (y becomes larger / less negative)
        tilt_left = calculate_shoulder_tilt_multiposture(
            ls_x=0.15, ls_y=-0.30, rs_x=-0.15, rs_y=-0.40
        )
        self.assertGreater(tilt_left, 10.0)

        # Leaning right: right shoulder drops (y becomes larger / less negative)
        tilt_right = calculate_shoulder_tilt_multiposture(
            ls_x=0.15, ls_y=-0.40, rs_x=-0.15, rs_y=-0.30
        )
        self.assertLess(tilt_right, -10.0)

    def test_neck_angle(self):
        # Perfectly aligned upright head
        neck_upright = calculate_neck_angle_multiposture(
            nose_x=0.0, nose_y=-0.55, mid_sh_x=0.0, mid_sh_y=-0.40
        )
        self.assertAlmostEqual(neck_upright, 0.0, places=1)

        # Head tilted sideways (dx = 0.10)
        neck_tilted = calculate_neck_angle_multiposture(
            nose_x=0.10, nose_y=-0.55, mid_sh_x=0.0, mid_sh_y=-0.40
        )
        self.assertGreater(neck_tilted, 25.0)

    def test_torso_angle(self):
        # Vertical torso (origin at mid-hip)
        torso_vert = calculate_torso_angle_multiposture(
            mid_sh_x=0.0, mid_sh_y=-0.40, mid_hip_x=0.0, mid_hip_y=0.0
        )
        self.assertAlmostEqual(torso_vert, 0.0, places=1)

    def test_shoulder_to_ear_ratio(self):
        # Typical values: shoulder width ~0.30, ear distance ~0.14 -> ratio ~2.14
        ratio = calculate_shoulder_to_ear_ratio_multiposture(
            ls_x=0.15, ls_y=-0.40, rs_x=-0.15, rs_y=-0.40,
            le_x=0.07, le_y=-0.55, re_x=-0.07, re_y=-0.55
        )
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(ratio, 2.1429, places=2)

        # Missing / near-zero ear distance
        zero_ear_ratio = calculate_shoulder_to_ear_ratio_multiposture(
            ls_x=0.15, ls_y=-0.40, rs_x=-0.15, rs_y=-0.40,
            le_x=0.0, le_y=0.0, re_x=0.0, re_y=0.0
        )
        self.assertIsNone(zero_ear_ratio)

    def test_head_to_shoulder_ratio(self):
        # Head elevation = -0.40 - (-0.55) = 0.15 over shoulder distance 0.30 -> 0.50
        hr = calculate_head_to_shoulder_ratio_multiposture(
            nose_y=-0.55, mid_sh_y=-0.40, sh_dist=0.30
        )
        self.assertIsNotNone(hr)
        self.assertAlmostEqual(hr, 0.50, places=2)


class TestMultiPostureDatasetSplit(unittest.TestCase):
    """Verifies dataset partitioning, schema, and participant leakage prevention."""

    def setUp(self):
        self.split_json = PROJECT_ROOT / "data" / "external" / "multiposture" / "participant_split.json"
        self.features_csv = PROJECT_ROOT / "data" / "external" / "multiposture" / "multiposture_features.csv"

    def test_split_file_exists_and_valid(self):
        self.assertTrue(self.split_json.exists(), "participant_split.json does not exist!")
        with open(self.split_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        train_p = set(meta["train_participants"])
        test_p = set(meta["test_participants"])

        # Zero participant leakage test: disjoint participant sets
        overlap = train_p.intersection(test_p)
        self.assertEqual(len(overlap), 0, f"Participant leakage detected: {overlap}")
        self.assertGreater(len(train_p), 0)
        self.assertGreater(len(test_p), 0)

        # Verify class counts
        self.assertIn("correct", meta["compatible_classes"])
        self.assertIn("leaning_left", meta["compatible_classes"])
        self.assertIn("leaning_right", meta["compatible_classes"])
        self.assertIn("slouching", meta["unsupported_classes"])

    def test_features_csv_schema_and_counts(self):
        self.assertTrue(self.features_csv.exists(), "multiposture_features.csv does not exist!")
        import csv
        with open(self.features_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertGreater(len(rows), 2000, "Expected at least 2000 compatible samples.")
        expected_cols = {
            "shoulder_tilt_deg", "neck_angle_deg", "shoulder_distance_raw",
            "shoulder_to_ear_ratio", "torso_angle_deg", "head_to_shoulder_ratio",
            "calibrated_tilt_deg", "label", "subject", "split"
        }
        for col in expected_cols:
            self.assertIn(col, reader.fieldnames)

        labels = set(r["label"] for r in rows)
        self.assertEqual(labels, {"correct", "leaning_left", "leaning_right"})
        self.assertNotIn("slouching", labels, "slouching should not be in MultiPosture features!")


class TestCandidateModelArtifact(unittest.TestCase):
    """Tests candidate model artifact packaging, metadata, and inference."""

    def setUp(self):
        self.candidate_path = PROJECT_ROOT / "models" / "candidate_multiposture_model.joblib"

    def test_candidate_bundle_structure(self):
        self.assertTrue(self.candidate_path.exists(), "candidate_multiposture_model.joblib does not exist!")
        bundle = joblib.load(self.candidate_path)
        self.assertIsInstance(bundle, dict)
        required_keys = [
            "model", "model_name", "feature_names", "classes",
            "unsupported_classes", "train_accuracy", "test_accuracy", "macro_f1"
        ]
        for k in required_keys:
            self.assertIn(k, bundle, f"Missing key '{k}' in candidate bundle")

        self.assertEqual(bundle["classes"], ["correct", "leaning_left", "leaning_right"])
        self.assertEqual(bundle["unsupported_classes"], ["slouching"])
        self.assertGreater(bundle["test_accuracy"], 0.85)

    def test_inference_with_posture_predictor(self):
        predictor = PosturePredictor(self.candidate_path)
        self.assertEqual(predictor.classes, ["correct", "leaning_left", "leaning_right"])

        # Test upright prediction
        pred_c, probs_c, _ = predictor.predict({"shoulder_tilt_deg": 0.0, "neck_angle_deg": 2.0})
        self.assertEqual(pred_c, "correct")
        self.assertGreater(probs_c["correct"], 0.70)

        # Test left lean prediction
        pred_l, probs_l, _ = predictor.predict({"shoulder_tilt_deg": 18.0, "neck_angle_deg": 35.0})
        self.assertEqual(pred_l, "leaning_left")
        self.assertGreater(probs_l["leaning_left"], 0.70)

        # Test right lean prediction
        pred_r, probs_r, _ = predictor.predict({"shoulder_tilt_deg": -18.0, "neck_angle_deg": 35.0})
        self.assertEqual(pred_r, "leaning_right")
        self.assertGreater(probs_r["leaning_right"], 0.70)


class TestDatasetAndActiveModelIntegrity(unittest.TestCase):
    """Verifies that active model and original datasets are completely intact."""

    def test_existing_datasets_unmodified(self):
        raw_csv = PROJECT_ROOT / "data" / "posture_data.csv"
        clean_csv = PROJECT_ROOT / "data" / "posture_data_clean.csv"
        split_json = PROJECT_ROOT / "data" / "session_split.json"

        self.assertTrue(raw_csv.exists())
        self.assertTrue(clean_csv.exists())
        self.assertTrue(split_json.exists())

        # Clean CSV must retain exactly 396 samples
        import csv
        with open(clean_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        self.assertEqual(len(rows), 396, "Existing clean dataset rows were modified!")

    def test_active_model_unmodified(self):
        active_path = PROJECT_ROOT / "models" / "posture_model.joblib"
        self.assertTrue(active_path.exists())
        bundle = joblib.load(active_path)
        self.assertIn("slouching", bundle["classes"])
        self.assertEqual(
            bundle["classes"],
            ["correct", "slouching", "leaning_left", "leaning_right"]
        )


if __name__ == "__main__":
    unittest.main()

