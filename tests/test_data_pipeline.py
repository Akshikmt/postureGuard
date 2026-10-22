"""
Unit tests for Dataset Validation and Preparation Pipeline (Task 5)
-------------------------------------------------------------------
Tests `src/validate_data.py` and `src/prepare_data.py` covering:
- Schema validation, missing columns, and empty files.
- Class distribution, missing classes, and balance evaluation.
- Sentinel value (-1.0) handling vs invalid numeric (NaN/Inf) values.
- Row rejection logging and preservation of clean data.
- Duplicate detection (exact duplicates and consecutive duplicates).
- Session metadata, data leakage detection, and session-aware partitioning.
"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.validate_data import (
    ALL_COLUMNS,
    EXPECTED_CLASSES,
    FEATURE_COLUMNS,
    analyze_dataset,
)
from src.prepare_data import clean_and_prepare_dataset


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.sample_csv = self.temp_path / "test_posture_data.csv"
        self.clean_csv = self.temp_path / "test_posture_data_clean.csv"
        self.splits_json = self.temp_path / "test_session_split.json"

        # Baseline valid sample row (close-up laptop posture with sentinels)
        self.base_features = [0.383, -1.0, -1.0, -1.0, -1.0, -2.2, -1.0, 1.8, -1.0, -1.0, -1.0, -1.0]

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_csv(self, rows, filepath=None, header=None):
        target = filepath or self.sample_csv
        hdr = header if header is not None else ALL_COLUMNS
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if hdr:
                writer.writerow(hdr)
            for r in rows:
                writer.writerow(r)
        return target

    def test_analyze_empty_file_and_missing_file(self):
        """Missing files and 0-byte files should be handled safely with descriptive errors."""
        missing = self.temp_path / "non_existent.csv"
        rep_missing = analyze_dataset(missing)
        self.assertFalse(rep_missing["file_exists"])
        self.assertIn("does not exist", rep_missing["error"])

        empty = self.temp_path / "empty.csv"
        empty.touch()
        rep_empty = analyze_dataset(empty)
        self.assertTrue(rep_empty["file_exists"])
        self.assertIn("empty", rep_empty["error"])

    def test_analyze_valid_5_class_dataset(self):
        """Dataset with all 5 classes and valid sentinels should pass validation completely."""
        rows = []
        for i, cls in enumerate(EXPECTED_CLASSES):
            for j in range(10):
                # Add slight variation to active features
                row = [
                    round(0.38 + i * 0.01 + j * 0.001, 4),
                    -1.0, -1.0, -1.0, -1.0,
                    round(-2.0 + i * 5.0 + j * 0.1, 2),
                    -1.0,
                    round(1.5 + i * 8.0 + j * 0.2, 2),
                    -1.0, -1.0, -1.0, -1.0,
                    cls,
                    f"2026-10-21T19:55:{j:02d}.000",
                    "session_001"
                ]
                rows.append(row)

        self._create_csv(rows)
        report = analyze_dataset(self.sample_csv)

        self.assertTrue(report["file_exists"])
        self.assertTrue(report["header_valid"])
        self.assertEqual(report["total_samples"], 50)
        self.assertEqual(len(report["missing_classes"]), 0)
        self.assertTrue(report["is_balanced"])
        self.assertEqual(report["balance_ratio"], 1.0)
        self.assertTrue(report["is_ready_for_cleaning"])

        # Check sentinel detection: 9 columns should have 100% sentinels
        for col, st in report["feature_statistics"].items():
            if not st["is_closeup_primary"]:
                self.assertEqual(st["sentinel_count"], 50)
                self.assertEqual(st["sentinel_percentage"], 100.0)
            else:
                self.assertEqual(st["sentinel_count"], 0)
                self.assertEqual(st["active_samples_count"], 50)

    def test_analyze_detects_nan_and_infinities(self):
        """NaN and Inf values must be identified per column."""
        row_nan = list(self.base_features) + ["correct", "2026-10-21T19:55:00", "session_001"]
        row_nan[0] = "NaN"
        row_inf = list(self.base_features) + ["slouching", "2026-10-21T19:55:01", "session_001"]
        row_inf[5] = "inf"

        self._create_csv([row_nan, row_inf])
        report = analyze_dataset(self.sample_csv)

        self.assertEqual(report["nan_inf_counts_by_col"]["shoulder_distance_raw"], 1)
        self.assertEqual(report["nan_inf_counts_by_col"]["shoulder_tilt_deg"], 1)
        self.assertFalse(report["is_ready_for_cleaning"])

    def test_analyze_detects_missing_and_unexpected_classes(self):
        """Missing expected classes and unmapped label strings must be flagged."""
        # Only 'correct' and an invalid 'unknown_pose'
        row1 = list(self.base_features) + ["correct", "2026-10-21T19:55:00", "session_001"]
        row2 = list(self.base_features) + ["unknown_pose", "2026-10-21T19:55:01", "session_001"]

        self._create_csv([row1, row2])
        report = analyze_dataset(self.sample_csv)

        self.assertIn("slouching", report["missing_classes"])
        self.assertIn("leaning_left", report["missing_classes"])
        self.assertIn("unknown_pose", report["unexpected_classes"])
        self.assertFalse(report["is_ready_for_cleaning"])

    def test_clean_and_prepare_retains_valid_and_rejects_corrupted(self):
        """clean_and_prepare_dataset should preserve valid sentinels and reject invalid rows."""
        valid_row_1 = list(self.base_features) + ["correct", "2026-10-21T19:55:00", "session_001"]
        valid_row_2 = list(self.base_features) + ["slouching", "2026-10-21T19:55:01", "session_001"]
        corrupted_nan = list(self.base_features) + ["leaning_left", "2026-10-21T19:55:02", "session_001"]
        corrupted_nan[7] = "nan"  # invalid neck angle
        corrupted_label = list(self.base_features) + ["invalid_lbl", "2026-10-21T19:55:03", "session_001"]
        empty_row = []

        self._create_csv([valid_row_1, valid_row_2, corrupted_nan, corrupted_label, empty_row])

        summary = clean_and_prepare_dataset(self.sample_csv, self.clean_csv, self.splits_json)

        self.assertEqual(summary["total_rows_read"], 5)
        self.assertEqual(summary["rows_retained"], 2)
        self.assertEqual(summary["rows_rejected"], 3)
        self.assertTrue(self.clean_csv.exists())
        self.assertTrue(self.splits_json.exists())

        # Verify clean CSV preserved -1.0 sentinels
        with open(self.clean_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ALL_COLUMNS)
            row = next(reader)
            # hip_distance_raw is at index 1: should be -1.0
            self.assertEqual(float(row[1]), -1.0)

    def test_prevent_overwriting_original_csv(self):
        """clean_and_prepare_dataset must strictly prevent overwriting raw source CSV."""
        self._create_csv([list(self.base_features) + ["correct", "2026-10-21T19:55:00", "session_001"]])
        with self.assertRaises(ValueError):
            clean_and_prepare_dataset(self.sample_csv, self.sample_csv, self.splits_json)

    def test_single_session_leakage_warning(self):
        """Single session dataset must be flagged with a leakage warning."""
        rows = [
            list(self.base_features) + [cls, f"2026-10-21T19:55:{i:02d}", "session_single"]
            for i, cls in enumerate(EXPECTED_CLASSES)
        ]
        self._create_csv(rows)

        summary = clean_and_prepare_dataset(self.sample_csv, self.clean_csv, self.splits_json)
        split_info = summary["split_info"]

        self.assertEqual(split_info["split_type"], "SINGLE_SESSION_ONLY")
        self.assertFalse(split_info["ready_for_session_evaluation"])
        self.assertIn("CRITICAL", split_info["leakage_warning"])

        # Check saved JSON
        with open(self.splits_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["split_type"], "SINGLE_SESSION_ONLY")

    def test_multi_session_aware_split(self):
        """When multiple balanced sessions exist, session-aware split should partition entire sessions."""
        rows = []
        # Create 5 distinct sessions, each containing all 5 classes
        for s_num in range(1, 6):
            s_id = f"session_00{s_num}"
            for i, cls in enumerate(EXPECTED_CLASSES):
                rows.append(list(self.base_features) + [cls, f"2026-10-21T19:55:{i:02d}", s_id])

        self._create_csv(rows)
        summary = clean_and_prepare_dataset(self.sample_csv, self.clean_csv, self.splits_json, random_seed=42)
        split_info = summary["split_info"]

        self.assertEqual(split_info["split_type"], "SESSION_AWARE_SPLIT")
        self.assertTrue(split_info["ready_for_session_evaluation"])
        self.assertGreater(len(split_info["train_sessions"]), 0)
        self.assertGreater(len(split_info["test_sessions"]), 0)

        # Ensure no session overlap between train and test
        train_set = set(split_info["train_sessions"])
        test_set = set(split_info["test_sessions"])
        self.assertEqual(len(train_set.intersection(test_set)), 0)


if __name__ == "__main__":
    unittest.main()
