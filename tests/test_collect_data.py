"""
Unit tests for PostureGuard dataset collector (src/collect_data.py).
Tests CSV initialization, validation, row appending, and error handling.
"""

import csv
import math
from pathlib import Path
import tempfile
import unittest

from src.collect_data import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_MAP,
    append_sample,
    initialize_dataset_file,
    validate_and_prepare_features,
)


class TestCollectData(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test CSVs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_csv = Path(self.temp_dir.name) / "test_posture_data.csv"

        # Sample valid feature dict
        self.valid_features = {
            "shoulder_distance_raw": 0.22,
            "hip_distance_raw": 0.18,
            "torso_length": 0.35,
            "shoulder_distance_norm": 0.628,
            "hip_distance_norm": 0.514,
            "shoulder_tilt_deg": 1.5,
            "torso_angle_deg": 3.2,
            "neck_angle_deg": 4.1,
            "left_hip_angle_deg": 88.0,
            "right_hip_angle_deg": 87.5,
            "left_knee_angle_deg": 91.0,
            "right_knee_angle_deg": 90.5,
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialize_new_csv(self):
        """Test that initialize_dataset_file creates the file with exact headers and 0 counts."""
        self.assertFalse(self.test_csv.exists())
        counts = initialize_dataset_file(self.test_csv)

        self.assertTrue(self.test_csv.exists())
        for lbl in LABEL_MAP.values():
            self.assertEqual(counts[lbl], 0)

        # Verify CSV header
        with open(self.test_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ALL_COLUMNS)

    def test_append_and_count_existing_samples(self):
        """Test appending samples and validating that re-opening counts them properly."""
        initialize_dataset_file(self.test_csv)

        # Append 2 'correct' and 1 'slouching'
        append_sample(self.test_csv, self.valid_features, "correct", "session_1", "2026-10-20T12:00:00")
        append_sample(self.test_csv, self.valid_features, "correct", "session_1", "2026-10-20T12:00:01")
        append_sample(self.test_csv, self.valid_features, "slouching", "session_1", "2026-10-20T12:00:02")

        # Re-initialize to test sample counting from existing file
        counts = initialize_dataset_file(self.test_csv)
        self.assertEqual(counts["correct"], 2)
        self.assertEqual(counts["slouching"], 1)
        self.assertEqual(counts["leaning_left"], 0)
        self.assertEqual(counts["leaning_right"], 0)
        self.assertEqual(counts["leaning_forward"], 0)

    def test_incompatible_csv_header_detection(self):
        """Test that an incompatible CSV raises a descriptive ValueError."""
        # Write corrupted header
        with open(self.test_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["col1", "col2", "label"])

        with self.assertRaises(ValueError) as ctx:
            initialize_dataset_file(self.test_csv)
        self.assertIn("Incompatible CSV header", str(ctx.exception))

    def test_feature_validation_valid_input(self):
        """Valid feature dictionary should pass validation cleanly."""
        is_valid, cleaned, msg = validate_and_prepare_features(self.valid_features)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Valid")
        self.assertIsNotNone(cleaned)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, cleaned)
            self.assertIsInstance(cleaned[col], float)

    def test_feature_validation_missing_core_features_skipped(self):
        """Missing core upper-body features (e.g. shoulder_tilt_deg or neck_angle_deg is None) must be skipped."""
        corrupted = self.valid_features.copy()
        corrupted["shoulder_tilt_deg"] = None

        is_valid, cleaned, msg = validate_and_prepare_features(corrupted)
        self.assertFalse(is_valid)
        self.assertIsNone(cleaned)
        self.assertIn("Shoulders not in frame", msg)

    def test_feature_validation_laptop_close_up(self):
        """When hips and torso are off-screen at laptop distance, should cleanly succeed and default to -1.0."""
        laptop_sitting = self.valid_features.copy()
        for col in [
            "hip_distance_raw",
            "torso_length",
            "shoulder_distance_norm",
            "hip_distance_norm",
            "torso_angle_deg",
            "left_hip_angle_deg",
            "right_hip_angle_deg",
            "left_knee_angle_deg",
            "right_knee_angle_deg",
        ]:
            laptop_sitting[col] = None

        is_valid, cleaned, msg = validate_and_prepare_features(laptop_sitting)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Valid")
        for col in [
            "hip_distance_raw",
            "torso_length",
            "shoulder_distance_norm",
            "hip_distance_norm",
            "torso_angle_deg",
            "left_hip_angle_deg",
            "right_hip_angle_deg",
            "left_knee_angle_deg",
            "right_knee_angle_deg",
        ]:
            self.assertEqual(cleaned[col], -1.0)

    def test_feature_validation_desk_occlusion(self):
        """When legs are occluded by a desk (None), they should cleanly default to -1.0 sentinel."""
        desk_sitting = self.valid_features.copy()
        desk_sitting["left_hip_angle_deg"] = None
        desk_sitting["right_hip_angle_deg"] = None
        desk_sitting["left_knee_angle_deg"] = None
        desk_sitting["right_knee_angle_deg"] = None

        is_valid, cleaned, msg = validate_and_prepare_features(desk_sitting)
        self.assertTrue(is_valid)
        self.assertEqual(cleaned["left_knee_angle_deg"], -1.0)
        self.assertEqual(cleaned["right_knee_angle_deg"], -1.0)
        self.assertEqual(cleaned["left_hip_angle_deg"], -1.0)
        self.assertEqual(cleaned["right_hip_angle_deg"], -1.0)

    def test_label_keys_mapping(self):
        """Verify the 5 expected posture classes exist and map to keys '1' through '5'."""
        expected = {
            ord("1"): "correct",
            ord("2"): "slouching",
            ord("3"): "leaning_left",
            ord("4"): "leaning_right",
            ord("5"): "leaning_forward",
        }
        self.assertEqual(LABEL_MAP, expected)


if __name__ == "__main__":
    unittest.main()
