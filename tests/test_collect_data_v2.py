"""
Unit tests for PostureGuard V2 dataset collector and feature validation (Phase 2).
Tests raw landmark extraction, scale-invariant head_to_shoulder_ratio calculation,
boundary conditions, missing/low-visibility handling, coordinate conventions,
V2 CSV persistence, metadata propagation, duplicate detection, and legacy protection.
"""

import csv
import math
from pathlib import Path
import tempfile
import unittest

from src.landmarks import LandmarkPoint
from src.features import (
    calculate_head_to_shoulder_ratio,
    calculate_neck_angle,
    calculate_shoulder_tilt,
    euclidean_distance_2d,
)
from src.collect_data import (
    DEFAULT_DATASET_V2_PATH,
    DuplicateFrameDetector,
    FOUR_CLASS_LABELS,
    LEGACY_ALL_COLUMNS,
    V2_ALL_COLUMNS,
    V2_FEATURE_COLUMNS,
    V2_LANDMARK_COLUMNS,
    V2_METADATA_COLUMNS,
    append_sample_v2,
    initialize_dataset_file_v2,
    validate_and_prepare_v2,
)


class TestCollectDataV2(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_csv = Path(self.temp_dir.name) / "posture_data_v2_test.csv"

        # Standard canonical upright pose landmarks (normalized image space: y increases downwards)
        self.valid_landmarks = {
            "nose": LandmarkPoint(x=0.50, y=0.20, z=-0.05, visibility=0.95),
            "left_shoulder": LandmarkPoint(x=0.65, y=0.35, z=0.00, visibility=0.90),
            "right_shoulder": LandmarkPoint(x=0.35, y=0.35, z=0.00, visibility=0.90),
            "left_ear": LandmarkPoint(x=0.56, y=0.19, z=-0.02, visibility=0.85),
            "right_ear": LandmarkPoint(x=0.44, y=0.19, z=-0.02, visibility=0.85),
            "left_hip": LandmarkPoint(x=0.60, y=0.75, z=0.05, visibility=0.75),
            "right_hip": LandmarkPoint(x=0.40, y=0.75, z=0.05, visibility=0.75),
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    # =========================================================================
    # 1. Raw Landmark Extraction & Formatting
    # =========================================================================

    def test_validate_and_prepare_v2_extracts_all_raw_landmarks(self):
        """Valid landmarks should be extracted with exact (x, y, z, v) fields rounded to 4 decimals."""
        is_valid, row_data, msg = validate_and_prepare_v2(self.valid_landmarks, min_visibility=0.40)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Valid")
        self.assertIsNotNone(row_data)

        # Check critical landmarks
        for pt_name in ("nose", "left_shoulder", "right_shoulder"):
            for coord in ("x", "y", "z", "v"):
                col = f"{pt_name}_{coord}"
                self.assertIn(col, row_data)
                self.assertIsInstance(row_data[col], float)

        self.assertEqual(row_data["nose_x"], 0.50)
        self.assertEqual(row_data["nose_y"], 0.20)
        self.assertEqual(row_data["nose_v"], 0.95)

        # Check optional landmarks
        for pt_name in ("left_ear", "right_ear", "left_hip", "right_hip"):
            for coord in ("x", "y", "z", "v"):
                col = f"{pt_name}_{coord}"
                self.assertIn(col, row_data)
                self.assertIsInstance(row_data[col], float)

    # =========================================================================
    # 2. Ratio Calculation and Boundary Conditions
    # =========================================================================

    def test_head_to_shoulder_ratio_upright_posture(self):
        """Upright posture should yield expected head_to_shoulder_ratio."""
        # mid_shoulder_y = 0.35, nose_y = 0.20 -> elevation = 0.15
        # shoulder_width = 0.65 - 0.35 = 0.30 -> ratio = 0.15 / 0.30 = 0.50
        ratio = calculate_head_to_shoulder_ratio(
            nose=self.valid_landmarks["nose"],
            left_shoulder=self.valid_landmarks["left_shoulder"],
            right_shoulder=self.valid_landmarks["right_shoulder"],
            min_visibility=0.40
        )
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(ratio, 0.50, places=3)

    def test_head_to_shoulder_ratio_slouched_posture(self):
        """Slouched posture (head drops lower in frame -> higher y) yields lower ratio."""
        # nose_y drops to 0.29 -> elevation = 0.35 - 0.29 = 0.06
        # shoulder_width = 0.30 -> ratio = 0.06 / 0.30 = 0.20
        slouched_nose = LandmarkPoint(x=0.50, y=0.29, z=0.08, visibility=0.90)
        ratio = calculate_head_to_shoulder_ratio(
            nose=slouched_nose,
            left_shoulder=self.valid_landmarks["left_shoulder"],
            right_shoulder=self.valid_landmarks["right_shoulder"],
            min_visibility=0.40
        )
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(ratio, 0.20, places=3)

    def test_head_to_shoulder_ratio_zero_elevation(self):
        """Head level with shoulder midpoint (elevation = 0.0) must return None."""
        level_nose = LandmarkPoint(x=0.50, y=0.35, z=0.0, visibility=0.90)
        ratio = calculate_head_to_shoulder_ratio(
            nose=level_nose,
            left_shoulder=self.valid_landmarks["left_shoulder"],
            right_shoulder=self.valid_landmarks["right_shoulder"]
        )
        self.assertIsNone(ratio)

    def test_head_to_shoulder_ratio_inverted_head(self):
        """Inverted head (nose below shoulders in image space -> nose_y > mid_shoulder_y) must return None."""
        inverted_nose = LandmarkPoint(x=0.50, y=0.45, z=0.0, visibility=0.90)
        ratio = calculate_head_to_shoulder_ratio(
            nose=inverted_nose,
            left_shoulder=self.valid_landmarks["left_shoulder"],
            right_shoulder=self.valid_landmarks["right_shoulder"]
        )
        self.assertIsNone(ratio)

    def test_head_to_shoulder_ratio_zero_shoulder_width(self):
        """Overlapping shoulders (width <= 1e-4) must return None to avoid division by zero."""
        left = LandmarkPoint(x=0.50, y=0.35, z=0.0, visibility=0.90)
        right = LandmarkPoint(x=0.50, y=0.35, z=0.0, visibility=0.90)
        ratio = calculate_head_to_shoulder_ratio(
            nose=self.valid_landmarks["nose"],
            left_shoulder=left,
            right_shoulder=right
        )
        self.assertIsNone(ratio)

    def test_head_to_shoulder_ratio_nan_and_inf_handling(self):
        """NaN or Inf coordinates must return None safely."""
        nan_nose = LandmarkPoint(x=float("nan"), y=0.20, z=0.0, visibility=0.90)
        ratio = calculate_head_to_shoulder_ratio(
            nose=nan_nose,
            left_shoulder=self.valid_landmarks["left_shoulder"],
            right_shoulder=self.valid_landmarks["right_shoulder"]
        )
        self.assertIsNone(ratio)

        inf_shoulder = LandmarkPoint(x=0.65, y=float("inf"), z=0.0, visibility=0.90)
        ratio = calculate_head_to_shoulder_ratio(
            nose=self.valid_landmarks["nose"],
            left_shoulder=inf_shoulder,
            right_shoulder=self.valid_landmarks["right_shoulder"]
        )
        self.assertIsNone(ratio)

    # =========================================================================
    # 3. Missing and Low-Visibility Landmark Handling
    # =========================================================================

    def test_missing_critical_landmark_rejected(self):
        """Missing nose or shoulder must cause validate_and_prepare_v2 to reject the frame."""
        for missing_key in ("nose", "left_shoulder", "right_shoulder"):
            landmarks = dict(self.valid_landmarks)
            landmarks[missing_key] = None
            is_valid, row_data, msg = validate_and_prepare_v2(landmarks, min_visibility=0.40)
            self.assertFalse(is_valid)
            self.assertIsNone(row_data)
            self.assertIn("Missing critical landmark", msg)

    def test_low_visibility_critical_landmark_rejected(self):
        """Critical landmark with visibility < 0.40 must cause validate_and_prepare_v2 to reject."""
        landmarks = dict(self.valid_landmarks)
        landmarks["nose"] = LandmarkPoint(x=0.50, y=0.20, z=-0.05, visibility=0.35)
        is_valid, row_data, msg = validate_and_prepare_v2(landmarks, min_visibility=0.40)
        self.assertFalse(is_valid)
        self.assertIsNone(row_data)
        self.assertIn("Low visibility", msg)

    def test_optional_landmarks_missing_persisted_as_empty_string(self):
        """When ears or hips are missing (None), frame is accepted but empty strings are persisted."""
        landmarks = dict(self.valid_landmarks)
        landmarks["left_ear"] = None
        landmarks["right_ear"] = None
        landmarks["left_hip"] = None
        landmarks["right_hip"] = None

        is_valid, row_data, msg = validate_and_prepare_v2(landmarks, min_visibility=0.40)
        self.assertTrue(is_valid)
        self.assertIsNotNone(row_data)

        for pt_name in ("left_ear", "right_ear", "left_hip", "right_hip"):
            for coord in ("x", "y", "z", "v"):
                self.assertEqual(row_data[f"{pt_name}_{coord}"], "")

    def test_optional_landmarks_low_visibility_persisted_as_empty_string(self):
        """When ears or hips have visibility < 0.40, empty strings are persisted without failing frame."""
        landmarks = dict(self.valid_landmarks)
        landmarks["left_ear"] = LandmarkPoint(x=0.56, y=0.19, z=-0.02, visibility=0.25)
        landmarks["right_hip"] = LandmarkPoint(x=0.40, y=0.75, z=0.05, visibility=0.15)

        is_valid, row_data, msg = validate_and_prepare_v2(landmarks, min_visibility=0.40)
        self.assertTrue(is_valid)
        self.assertIsNotNone(row_data)

        # left_ear and right_hip should be empty string
        for coord in ("x", "y", "z", "v"):
            self.assertEqual(row_data[f"left_ear_{coord}"], "")
            self.assertEqual(row_data[f"right_hip_{coord}"], "")

        # right_ear and left_hip were high confidence, so they should be populated
        self.assertIsInstance(row_data["right_ear_x"], float)
        self.assertIsInstance(row_data["left_hip_x"], float)

    # =========================================================================
    # 4. Coordinate Direction Consistency
    # =========================================================================

    def test_coordinate_direction_image_space(self):
        """
        Verify MediaPipe normalized image-space conventions:
        - y=0.0 is the top edge, y=1.0 is the bottom edge.
        - When the head is physically above the shoulders, nose.y < mid_shoulder.y.
        - Therefore, head elevation is calculated as (mid_shoulder.y - nose.y) > 0.
        """
        nose = self.valid_landmarks["nose"]
        ls = self.valid_landmarks["left_shoulder"]
        rs = self.valid_landmarks["right_shoulder"]
        mid_sh_y = (ls.y + rs.y) / 2.0

        self.assertLess(nose.y, mid_sh_y, "In image space, head above shoulders must have nose.y < mid_sh_y")
        elevation = mid_sh_y - nose.y
        self.assertGreater(elevation, 0.0)

    # =========================================================================
    # 5. CSV Schema and Persistence
    # =========================================================================

    def test_initialize_new_v2_csv(self):
        """initialize_dataset_file_v2 should create the file with 37 V2 columns and 0 counts for 4 classes."""
        self.assertFalse(self.test_csv.exists())
        counts = initialize_dataset_file_v2(self.test_csv)
        self.assertTrue(self.test_csv.exists())

        # Exact 4 classes
        self.assertEqual(set(counts.keys()), set(FOUR_CLASS_LABELS))
        for lbl in FOUR_CLASS_LABELS:
            self.assertEqual(counts[lbl], 0)

        # Header verification (37 columns)
        with open(self.test_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, V2_ALL_COLUMNS)
            self.assertEqual(len(header), 37)

    def test_append_sample_v2_and_reload_counts(self):
        """Appending samples should write exact columns and reload accurate class counts."""
        initialize_dataset_file_v2(self.test_csv)
        _, row_data, _ = validate_and_prepare_v2(self.valid_landmarks)

        # Append samples across all 4 classes
        append_sample_v2(self.test_csv, row_data, "correct", "user_1", "sess_001", 0, "2026-10-31T12:00:00Z")
        append_sample_v2(self.test_csv, row_data, "correct", "user_1", "sess_001", 1, "2026-10-31T12:00:01Z")
        append_sample_v2(self.test_csv, row_data, "slouching", "user_1", "sess_001", 2, "2026-10-31T12:00:02Z")
        append_sample_v2(self.test_csv, row_data, "leaning_left", "user_1", "sess_001", 3, "2026-10-31T12:00:03Z")
        append_sample_v2(self.test_csv, row_data, "leaning_right", "user_1", "sess_001", 4, "2026-10-31T12:00:04Z")

        # Reload counts
        reloaded_counts = initialize_dataset_file_v2(self.test_csv)
        self.assertEqual(reloaded_counts["correct"], 2)
        self.assertEqual(reloaded_counts["slouching"], 1)
        self.assertEqual(reloaded_counts["leaning_left"], 1)
        self.assertEqual(reloaded_counts["leaning_right"], 1)

    # =========================================================================
    # 6. Participant, Session, Frame, and Timestamp Metadata
    # =========================================================================

    def test_metadata_fields_persisted_correctly(self):
        """Participant ID, session ID, frame index, and ISO timestamp must be accurately preserved in CSV."""
        initialize_dataset_file_v2(self.test_csv)
        _, row_data, _ = validate_and_prepare_v2(self.valid_landmarks)

        append_sample_v2(
            filepath=self.test_csv,
            row_data=row_data,
            label="slouching",
            participant_id="subject_42",
            session_id="session_20261031_180000",
            frame_index=157,
            timestamp="2026-10-31T18:00:00.123456+00:00"
        )

        with open(self.test_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

            self.assertEqual(row["participant_id"], "subject_42")
            self.assertEqual(row["session_id"], "session_20261031_180000")
            self.assertEqual(row["frame_index"], "157")
            self.assertEqual(row["timestamp"], "2026-10-31T18:00:00.123456+00:00")
            self.assertEqual(row["label"], "slouching")
            self.assertAlmostEqual(float(row["head_to_shoulder_ratio"]), 0.50, places=3)
            self.assertAlmostEqual(float(row["nose_x"]), 0.50, places=3)

    # =========================================================================
    # 7. Duplicate Frame Detection
    # =========================================================================

    def test_duplicate_frame_detector_identical_frames(self):
        """DuplicateFrameDetector flags exact repeated coordinate buffers."""
        detector = DuplicateFrameDetector(precision=6)
        nose = self.valid_landmarks["nose"]
        ls = self.valid_landmarks["left_shoulder"]
        rs = self.valid_landmarks["right_shoulder"]

        # First frame is never duplicate
        self.assertFalse(detector.is_duplicate(nose, ls, rs))

        # Re-sending identical frame must be flagged
        self.assertTrue(detector.is_duplicate(nose, ls, rs))

    def test_duplicate_frame_detector_allows_micro_movements(self):
        """Legitimate stationary human posture with natural micro-sway (>= 1e-4) must NOT be flagged as duplicate."""
        detector = DuplicateFrameDetector(precision=6)
        nose = self.valid_landmarks["nose"]
        ls = self.valid_landmarks["left_shoulder"]
        rs = self.valid_landmarks["right_shoulder"]

        self.assertFalse(detector.is_duplicate(nose, ls, rs))

        # Simulating physiological tremor / photon noise: shift by 0.0005 (about 0.3-0.5 pixels)
        jittered_nose = LandmarkPoint(x=nose.x + 0.0005, y=nose.y, z=nose.z, visibility=nose.visibility)
        self.assertFalse(detector.is_duplicate(jittered_nose, ls, rs))

    def test_duplicate_frame_detector_reset(self):
        """Calling reset() clears the cached signature."""
        detector = DuplicateFrameDetector(precision=6)
        nose = self.valid_landmarks["nose"]
        ls = self.valid_landmarks["left_shoulder"]
        rs = self.valid_landmarks["right_shoulder"]

        detector.is_duplicate(nose, ls, rs)
        detector.reset()
        self.assertFalse(detector.is_duplicate(nose, ls, rs))

    # =========================================================================
    # 8. Legacy Dataset Protection
    # =========================================================================

    def test_legacy_dataset_protection_raises_error(self):
        """
        Calling initialize_dataset_file_v2 on a legacy CSV file (e.g. 15 or 16 columns)
        must raise ValueError and refuse to overwrite or append to it.
        """
        legacy_csv = Path(self.temp_dir.name) / "legacy_posture_data.csv"
        with open(legacy_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LEGACY_ALL_COLUMNS)
            # Add a fake legacy row
            writer.writerow([0.22, 0.18, 0.35, 0.628, 0.514, 1.5, 3.2, 4.1, 88.0, 87.5, 91.0, 90.5, "correct", "2026-10-20", "sess_0"])

        # Attempting to initialize with V2 should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            initialize_dataset_file_v2(legacy_csv)

        err_msg = str(ctx.exception)
        self.assertIn("Incompatible schema in target file", err_msg)
        self.assertIn("V2 requires 37 columns", err_msg)

        # Verify legacy file was NOT overwritten or corrupted
        with open(legacy_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, LEGACY_ALL_COLUMNS)
            rows = list(reader)
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()

