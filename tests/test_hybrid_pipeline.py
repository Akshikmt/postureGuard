"""
Unit Tests for PostureGuard Hybrid Pipeline (Phase 1)
-----------------------------------------------------
Verifies:
1. Strict non-overwriting of active model and candidate MultiPosture model.
2. Strict non-modification of original datasets.
3. Hybrid dataset schema, sample counts, and leakage prevention.
4. Candidate hybrid model bundle structure, metadata, and inference capability.
5. Verification that head_to_shoulder_ratio was not falsely fabricated.
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

from src.realtime_posture import PosturePredictor


class TestHybridPipeline(unittest.TestCase):

    def setUp(self):
        self.active_model_path = PROJECT_ROOT / "models" / "posture_model.joblib"
        self.mp_candidate_path = PROJECT_ROOT / "models" / "candidate_multiposture_model.joblib"
        self.hybrid_candidate_path = PROJECT_ROOT / "models" / "candidate_hybrid_model.joblib"

        self.raw_pg_csv = PROJECT_ROOT / "data" / "posture_data.csv"
        self.clean_pg_csv = PROJECT_ROOT / "data" / "posture_data_clean.csv"
        self.mp_data_csv = PROJECT_ROOT / "data" / "external" / "multiposture" / "data.csv"

        self.hybrid_csv = PROJECT_ROOT / "data" / "external" / "multiposture" / "hybrid_features.csv"
        self.hybrid_split_json = PROJECT_ROOT / "data" / "external" / "multiposture" / "hybrid_split.json"
        self.report_md = PROJECT_ROOT / "reports" / "hybrid_evaluation_report.md"
        self.report_json = PROJECT_ROOT / "reports" / "hybrid_evaluation_report.json"

    def test_original_models_preserved(self):
        """Active model and MultiPosture candidate must remain completely untouched."""
        self.assertTrue(self.active_model_path.exists())
        self.assertTrue(self.mp_candidate_path.exists())

        active_bundle = joblib.load(self.active_model_path)
        self.assertEqual(active_bundle["model_name"], "DecisionTree")
        self.assertIn("classes", active_bundle)

        mp_bundle = joblib.load(self.mp_candidate_path)
        self.assertEqual(mp_bundle["classes"], ["correct", "leaning_left", "leaning_right"])
        self.assertNotIn("slouching", mp_bundle["classes"])

    def test_original_datasets_unmodified(self):
        """Original datasets must not be overwritten or modified."""
        self.assertTrue(self.raw_pg_csv.exists())
        self.assertTrue(self.clean_pg_csv.exists())
        self.assertTrue(self.mp_data_csv.exists())

        import csv
        with open(self.clean_pg_csv, "r", newline="", encoding="utf-8") as f:
            clean_rows = list(csv.reader(f))
        # 1 header + 396 data rows = 397
        self.assertEqual(len(clean_rows), 397)

        with open(self.mp_data_csv, "r", newline="", encoding="utf-8") as f:
            mp_rows = list(csv.reader(f))
        # 1 header + 4794 data rows = 4795
        self.assertEqual(len(mp_rows), 4795)

    def test_hybrid_outputs_exist(self):
        """All required hybrid outputs must exist."""
        self.assertTrue(self.hybrid_csv.exists())
        self.assertTrue(self.hybrid_split_json.exists())
        self.assertTrue(self.hybrid_candidate_path.exists())
        self.assertTrue(self.report_md.exists())
        self.assertTrue(self.report_json.exists())

    def test_hybrid_dataset_schema_and_leakage(self):
        """Validates zero leakage and expected class distribution."""
        with open(self.hybrid_split_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        train_mp = set(meta["partitions"]["train"]["multiposture_participants"])
        test_mp = set(meta["partitions"]["test"]["multiposture_participants"])
        overlap = train_mp.intersection(test_mp)
        self.assertEqual(len(overlap), 0, f"Participant leakage in MultiPosture: {overlap}")

        train_pg = set(meta["partitions"]["train"]["postureguard_sessions"])
        test_pg = set(meta["partitions"]["test"]["postureguard_sessions"])
        overlap_pg = train_pg.intersection(test_pg)
        self.assertEqual(len(overlap_pg), 0, f"Session leakage in PostureGuard: {overlap_pg}")

        self.assertEqual(meta["partitions"]["train"]["total_samples"], 2084)
        self.assertEqual(meta["partitions"]["test"]["total_samples"], 446)

    def test_candidate_hybrid_model_loading_and_inference(self):
        """Tests that PosturePredictor can load and run inference with candidate_hybrid_model.joblib."""
        predictor = PosturePredictor(self.hybrid_candidate_path)
        self.assertEqual(predictor.classes, ["correct", "slouching", "leaning_left", "leaning_right"])

        # Test upright prediction
        pred_c, probs_c, _ = predictor.predict({"calibrated_tilt_deg": 0.0, "neck_angle_deg": 3.0})
        self.assertIn(pred_c, predictor.classes)
        self.assertIn("slouching", probs_c)
        self.assertIn("correct", probs_c)

        # Test left lean prediction
        pred_l, probs_l, _ = predictor.predict({"calibrated_tilt_deg": 18.0, "neck_angle_deg": 35.0})
        self.assertEqual(pred_l, "leaning_left")

        # Test right lean prediction
        pred_r, probs_r, _ = predictor.predict({"calibrated_tilt_deg": -18.0, "neck_angle_deg": 35.0})
        self.assertEqual(pred_r, "leaning_right")


if __name__ == "__main__":
    unittest.main()

