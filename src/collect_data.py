"""
PostureGuard Dataset Collector V2 (Laptop Close-Up & 4-Class Enhanced)
----------------------------------------------------------------------
Captures webcam frames, extracts MediaPipe pose landmarks, computes validated
geometric metrics (including scale-invariant head_to_shoulder_ratio), and
guides the user through collecting balanced training samples for four active
sitting posture classes:

1. correct       - Upright, ergonomically aligned posture.
2. slouching     - Forward/downward spinal slump (head drop, rounded shoulders).
3. leaning_left  - Lateral spinal lean toward the user's left.
4. leaning_right - Lateral spinal lean toward the user's right.

NOTE: 'leaning_forward' is excluded from active collection until its difference
from slouching is formally defined and validated.

Key V2 Enhancements:
- Raw Landmark Persistence: Persists full (x, y, z, v) coordinates for key
  landmarks (nose, shoulders, ears, hips) to support future feature engineering.
- Head-to-Shoulder Ratio: Persists scale-invariant (mid_shoulder_y - nose_y) / shoulder_width.
- Missing Landmark Policy: Critical upper landmarks (nose, shoulders) require visibility >= 0.40.
  Optional landmarks (ears, hips) remain empty strings "" when occluded, never fabricated.
- Rigorous Metadata: Records participant_id, session_id, frame_index, and UTC ISO timestamp.
- Anti-Duplicate Buffer: Discards duplicate hardware buffer re-reads while preserving legitimate
  stationary human postures.
- Legacy Dataset Protection: Defaults to `data/posture_data_v2.csv` and strictly rejects writing
  to legacy CSVs to avoid corrupting historical datasets.
"""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.landmarks import LandmarkPoint, extract_key_landmarks
    from src.features import (
        calculate_head_to_shoulder_ratio,
        calculate_neck_angle,
        calculate_shoulder_tilt,
        euclidean_distance_2d,
        extract_posture_features,
    )
except ModuleNotFoundError:
    from landmarks import LandmarkPoint, extract_key_landmarks
    from features import (
        calculate_head_to_shoulder_ratio,
        calculate_neck_angle,
        calculate_shoulder_tilt,
        euclidean_distance_2d,
        extract_posture_features,
    )

# File Paths
DEFAULT_DATASET_V2_PATH = PROJECT_ROOT / "data" / "posture_data_v2.csv"
LEGACY_DATASET_PATH = PROJECT_ROOT / "data" / "posture_data.csv"
DATASET_CSV_PATH = DEFAULT_DATASET_V2_PATH  # Default for module-level references

# Active 4-Class Target Taxonomy
FOUR_CLASS_LABELS = [
    "correct",
    "slouching",
    "leaning_left",
    "leaning_right",
]

# V2 Schema Definitions (37 columns total)
V2_METADATA_COLUMNS = [
    "participant_id",
    "session_id",
    "frame_index",
    "timestamp",
    "label",
]

V2_FEATURE_COLUMNS = [
    "shoulder_tilt_deg",
    "neck_angle_deg",
    "head_to_shoulder_ratio",
    "shoulder_width",
]

V2_LANDMARK_COLUMNS = [
    "nose_x", "nose_y", "nose_z", "nose_v",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z", "left_shoulder_v",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z", "right_shoulder_v",
    "left_ear_x", "left_ear_y", "left_ear_z", "left_ear_v",
    "right_ear_x", "right_ear_y", "right_ear_z", "right_ear_v",
    "left_hip_x", "left_hip_y", "left_hip_z", "left_hip_v",
    "right_hip_x", "right_hip_y", "right_hip_z", "right_hip_v",
]

V2_ALL_COLUMNS = V2_METADATA_COLUMNS + V2_FEATURE_COLUMNS + V2_LANDMARK_COLUMNS

# Automated Guided 4-Pose Sequence
AUTO_POSES_V2 = [
    {
        "label": "correct",
        "title": "1/4: Correct Posture",
        "instruction": "Sit upright, shoulders relaxed and level, head aligned forward",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
    {
        "label": "slouching",
        "title": "2/4: Slouching",
        "instruction": "Slouch back or hunch/round shoulders and head downward naturally",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
    {
        "label": "leaning_left",
        "title": "3/4: Leaning Left",
        "instruction": "Lean your head and shoulders towards your LEFT side",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
    {
        "label": "leaning_right",
        "title": "4/4: Leaning Right",
        "instruction": "Lean your head and shoulders towards your RIGHT side",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
]

LABEL_MAP_V2 = {
    ord("1"): "correct",
    ord("2"): "slouching",
    ord("3"): "leaning_left",
    ord("4"): "leaning_right",
}

# =========================================================================
# Legacy Constants (Strictly preserved for backward compatibility & tests)
# =========================================================================
LEGACY_FEATURE_COLUMNS = [
    "shoulder_distance_raw",
    "hip_distance_raw",
    "torso_length",
    "shoulder_distance_norm",
    "hip_distance_norm",
    "shoulder_tilt_deg",
    "torso_angle_deg",
    "neck_angle_deg",
    "left_hip_angle_deg",
    "right_hip_angle_deg",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
]
EXTENDED_FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS + ["shoulder_to_ear_ratio"]
FEATURE_COLUMNS = EXTENDED_FEATURE_COLUMNS
REQUIRED_CORE_FEATURES = [
    "shoulder_distance_raw",
    "shoulder_tilt_deg",
    "neck_angle_deg",
]
OPTIONAL_DESK_FEATURES = [
    "hip_distance_raw",
    "torso_length",
    "shoulder_distance_norm",
    "hip_distance_norm",
    "torso_angle_deg",
    "left_hip_angle_deg",
    "right_hip_angle_deg",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "shoulder_to_ear_ratio",
]
METADATA_COLUMNS = ["label", "timestamp", "session_id"]
LEGACY_ALL_COLUMNS = LEGACY_FEATURE_COLUMNS + METADATA_COLUMNS
EXTENDED_ALL_COLUMNS = EXTENDED_FEATURE_COLUMNS + METADATA_COLUMNS
ALL_COLUMNS = EXTENDED_ALL_COLUMNS

# Legacy poses and map
LABEL_MAP = {
    ord("1"): "correct",
    ord("2"): "slouching",
    ord("3"): "leaning_left",
    ord("4"): "leaning_right",
    ord("5"): "leaning_forward",
}
AUTO_POSES = [
    {
        "label": "correct",
        "title": "1/5: Correct Posture",
        "instruction": "Sit upright, shoulders relaxed, head aligned forward",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
    {
        "label": "slouching",
        "title": "2/5: Slouching",
        "instruction": "Slouch back or round shoulders down naturally",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
    {
        "label": "leaning_left",
        "title": "3/5: Leaning Left",
        "instruction": "Lean your head and shoulders towards your LEFT side",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
    {
        "label": "leaning_right",
        "title": "4/5: Leaning Right",
        "instruction": "Lean your head and shoulders towards your RIGHT side",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
    {
        "label": "leaning_forward",
        "title": "5/5: Leaning Forward",
        "instruction": "Lean your upper body forward closer to the screen",
        "prep_sec": 4.0,
        "record_sec": 6.0,
        "target_samples": 30,
    },
]

# Sampling rate configuration (5 samples per second)
TARGET_SAMPLES_PER_SEC = 5.0
SAMPLE_INTERVAL_SEC = 1.0 / TARGET_SAMPLES_PER_SEC


def play_sound_async(sound_type: str = "tick"):
    """Plays non-blocking audio cues using winsound on Windows."""
    def _run():
        try:
            import winsound
            if sound_type == "tick":
                winsound.Beep(950, 60)
            elif sound_type == "start":
                winsound.Beep(1400, 160)
            elif sound_type == "pose_done":
                winsound.Beep(1200, 90)
                time.sleep(0.04)
                winsound.Beep(1600, 140)
            elif sound_type == "session_complete":
                winsound.Beep(1000, 100)
                time.sleep(0.04)
                winsound.Beep(1300, 100)
                time.sleep(0.04)
                winsound.Beep(1750, 250)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


# =========================================================================
# V2 Implementation: Validation, Duplicate Detection, File Operations
# =========================================================================

class DuplicateFrameDetector:
    """
    Prevents saving identical repeated video buffers caused by webcam sensor frame repeat
    or video decoder frame drops.

    Implementation & Threshold:
    - Compares the 9 core 3D coordinates (x, y, z) across nose, left_shoulder, right_shoulder
      rounded to 6 decimal places (micro-precision).
    - In legitimate human postures (even stationary), physiological micro-sway and sensor photon
      noise produce spatial coordinate variations >= 1e-4 (~0.5 - 2 pixels).
    - If and only if all 9 coordinates match the previous frame down to 1e-6,
      the sample is flagged as a duplicate camera frame repeat and discarded.
    """
    def __init__(self, precision: int = 6):
        self.precision = precision
        self.last_signature: Optional[Tuple[float, ...]] = None

    def is_duplicate(
        self,
        nose: Optional[LandmarkPoint],
        left_shoulder: Optional[LandmarkPoint],
        right_shoulder: Optional[LandmarkPoint]
    ) -> bool:
        if nose is None or left_shoulder is None or right_shoulder is None:
            return False

        sig = (
            round(nose.x, self.precision),
            round(nose.y, self.precision),
            round(nose.z, self.precision),
            round(left_shoulder.x, self.precision),
            round(left_shoulder.y, self.precision),
            round(left_shoulder.z, self.precision),
            round(right_shoulder.x, self.precision),
            round(right_shoulder.y, self.precision),
            round(right_shoulder.z, self.precision),
        )
        if self.last_signature is not None and sig == self.last_signature:
            return True
        self.last_signature = sig
        return False

    def reset(self):
        self.last_signature = None


def validate_and_prepare_v2(
    landmarks: Optional[Dict[str, Optional[LandmarkPoint]]],
    min_visibility: float = 0.40
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Validates upper-body landmarks and computes features for V2 dataset schema.

    Validation Rules:
    - Critical landmarks (nose, left_shoulder, right_shoulder) must all be present
      with finite coordinates and visibility >= min_visibility.
    - shoulder_width must be strictly positive (> 1e-4).
    - head_to_shoulder_ratio must be positive (> 0.0), indicating head is above shoulders.
    - Optional landmarks (ears, hips): if present, finite, and visibility >= min_visibility,
      persists formatted float values; otherwise persists empty strings "".
    - Returns False with an explicit diagnostic message if any check fails.

    Returns:
        (is_valid, row_data_dict, status_message)
    """
    if not landmarks:
        return False, None, "No person detected"

    nose = landmarks.get("nose")
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")

    # 1. Critical landmarks presence, finite checks, and visibility
    for name, pt in (("nose", nose), ("left_shoulder", ls), ("right_shoulder", rs)):
        if pt is None:
            return False, None, f"Missing critical landmark: {name}"
        if (
            math.isnan(pt.x) or math.isnan(pt.y) or math.isnan(pt.z) or math.isnan(pt.visibility) or
            math.isinf(pt.x) or math.isinf(pt.y) or math.isinf(pt.z) or math.isinf(pt.visibility)
        ):
            return False, None, f"Non-finite value in {name}"
        if pt.visibility < min_visibility:
            return False, None, f"Low visibility in {name} ({pt.visibility:.2f} < {min_visibility})"

    # 2. Shoulder width calculation
    sh_width = euclidean_distance_2d(ls, rs)
    if sh_width is None or sh_width <= 1e-4:
        return False, None, "Shoulder width too small or shoulders overlapped"

    # 3. Head-to-shoulder ratio calculation
    head_ratio = calculate_head_to_shoulder_ratio(nose, ls, rs, min_visibility=min_visibility)
    if head_ratio is None or head_ratio <= 0.0:
        return False, None, "Invalid head elevation: nose at or below shoulder midpoint"

    # 4. Angle calculations
    tilt = calculate_shoulder_tilt(ls, rs)
    mid_sh = LandmarkPoint(
        x=(ls.x + rs.x) / 2.0,
        y=(ls.y + rs.y) / 2.0,
        z=(ls.z + rs.z) / 2.0,
        visibility=min(ls.visibility, rs.visibility)
    )
    neck = calculate_neck_angle(nose, mid_sh)

    if tilt is None or neck is None:
        return False, None, "Failed to compute shoulder tilt or neck angle"

    row_data: Dict[str, Any] = {
        "shoulder_tilt_deg": tilt,
        "neck_angle_deg": neck,
        "head_to_shoulder_ratio": head_ratio,
        "shoulder_width": round(sh_width, 4),
        "nose_x": round(nose.x, 4),
        "nose_y": round(nose.y, 4),
        "nose_z": round(nose.z, 4),
        "nose_v": round(nose.visibility, 4),
        "left_shoulder_x": round(ls.x, 4),
        "left_shoulder_y": round(ls.y, 4),
        "left_shoulder_z": round(ls.z, 4),
        "left_shoulder_v": round(ls.visibility, 4),
        "right_shoulder_x": round(rs.x, 4),
        "right_shoulder_y": round(rs.y, 4),
        "right_shoulder_z": round(rs.z, 4),
        "right_shoulder_v": round(rs.visibility, 4),
    }

    # 5. Optional landmarks (ears and hips)
    for pt_name in ("left_ear", "right_ear", "left_hip", "right_hip"):
        pt = landmarks.get(pt_name)
        if (
            pt is not None and
            not math.isnan(pt.x) and not math.isnan(pt.y) and not math.isnan(pt.z) and not math.isnan(pt.visibility) and
            not math.isinf(pt.x) and not math.isinf(pt.y) and not math.isinf(pt.z) and not math.isinf(pt.visibility) and
            pt.visibility >= min_visibility
        ):
            row_data[f"{pt_name}_x"] = round(pt.x, 4)
            row_data[f"{pt_name}_y"] = round(pt.y, 4)
            row_data[f"{pt_name}_z"] = round(pt.z, 4)
            row_data[f"{pt_name}_v"] = round(pt.visibility, 4)
        else:
            row_data[f"{pt_name}_x"] = ""
            row_data[f"{pt_name}_y"] = ""
            row_data[f"{pt_name}_z"] = ""
            row_data[f"{pt_name}_v"] = ""

    return True, row_data, "Valid"


def initialize_dataset_file_v2(filepath: Path) -> Dict[str, int]:
    """
    Initializes a V2 dataset CSV file with correct V2 headers if it does not exist.
    If it exists, validates that the header strictly matches V2_ALL_COLUMNS and
    counts existing samples per label.

    Safety:
    If the target file has an incompatible schema (e.g. legacy dataset), raises a
    descriptive ValueError rather than silently modifying or overwriting it.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    counts = {lbl: 0 for lbl in FOUR_CLASS_LABELS}

    if filepath.exists() and filepath.stat().st_size > 0:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != V2_ALL_COLUMNS:
                raise ValueError(
                    f"Incompatible schema in target file: {filepath}\n"
                    f"Found {len(header) if header else 0} columns, but V2 requires {len(V2_ALL_COLUMNS)} columns.\n"
                    f"To prevent data corruption, the V2 collector will never overwrite or append to legacy or incompatible files."
                )
            label_idx = V2_ALL_COLUMNS.index("label")
            for row in reader:
                if len(row) == len(V2_ALL_COLUMNS):
                    lbl = row[label_idx]
                    if lbl in counts:
                        counts[lbl] += 1
    else:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(V2_ALL_COLUMNS)

    return counts


def append_sample_v2(
    filepath: Path,
    row_data: Dict[str, Any],
    label: str,
    participant_id: str,
    session_id: str,
    frame_index: int,
    timestamp: str
) -> None:
    """Appends a single verified sample row to the V2 dataset CSV matching V2_ALL_COLUMNS."""
    row = [
        participant_id,
        session_id,
        frame_index,
        timestamp,
        label,
        row_data["shoulder_tilt_deg"],
        row_data["neck_angle_deg"],
        row_data["head_to_shoulder_ratio"],
        row_data["shoulder_width"],
    ]
    for col in V2_LANDMARK_COLUMNS:
        row.append(row_data.get(col, ""))

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =========================================================================
# Legacy Functions (Strictly preserved for existing tests & backward compat)
# =========================================================================

def initialize_dataset_file(filepath: Path) -> Dict[str, int]:
    """Legacy initialization function preserved for backward compatibility and tests."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    counts = {pose["label"]: 0 for pose in AUTO_POSES}

    if filepath.exists() and filepath.stat().st_size > 0:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header == EXTENDED_ALL_COLUMNS:
                active_columns = EXTENDED_ALL_COLUMNS
            elif header == LEGACY_ALL_COLUMNS:
                active_columns = LEGACY_ALL_COLUMNS
            else:
                raise ValueError(
                    f"Incompatible CSV header in {filepath}.\n"
                    f"Expected: {EXTENDED_ALL_COLUMNS}\n"
                    f"Or Legacy: {LEGACY_ALL_COLUMNS}\n"
                    f"Found:    {header}"
                )
            label_idx = active_columns.index("label")
            for row in reader:
                if len(row) == len(active_columns):
                    lbl = row[label_idx]
                    if lbl in counts:
                        counts[lbl] += 1
    else:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ALL_COLUMNS)

    return counts


def validate_and_prepare_features(
    features: Optional[Dict[str, Optional[float]]]
) -> Tuple[bool, Optional[Dict[str, float]], str]:
    """Legacy validation function preserved for backward compatibility and tests."""
    if not features:
        return False, None, "No person detected"

    cleaned: Dict[str, float] = {}
    missing_upper = []
    for col in REQUIRED_CORE_FEATURES:
        val = features.get(col)
        if val is None:
            missing_upper.append(col)
        elif not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            return False, None, f"Invalid {col}: {val}"
        else:
            cleaned[col] = float(val)

    if missing_upper:
        if "shoulder_distance_raw" in missing_upper or "shoulder_tilt_deg" in missing_upper:
            return False, None, "Shoulders not in frame"
        if "neck_angle_deg" in missing_upper:
            return False, None, "Head/Nose not in frame"
        return False, None, "Missing upper features"

    for col in OPTIONAL_DESK_FEATURES:
        val = features.get(col)
        if val is None:
            cleaned[col] = -1.0
        elif not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            return False, None, f"Invalid {col}: {val}"
        else:
            cleaned[col] = float(val)

    return True, cleaned, "Valid"


def append_sample(
    filepath: Path,
    features: Dict[str, Any],
    label: str,
    session_id: str,
    timestamp: str
) -> None:
    """Legacy sample append function preserved for backward compatibility and tests."""
    target_feature_cols = FEATURE_COLUMNS
    if filepath.exists() and filepath.stat().st_size > 0:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header == LEGACY_ALL_COLUMNS:
                target_feature_cols = LEGACY_FEATURE_COLUMNS

    row = [features.get(col, -1.0) for col in target_feature_cols] + [label, timestamp, session_id]
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =========================================================================
# HUD Rendering
# =========================================================================

def draw_collector_hud(
    frame: Any,
    state: str,
    current_pose_idx: int,
    current_label: Optional[str],
    timer_remaining: float,
    current_pose_samples: int,
    target_pose_samples: int,
    class_counts: Dict[str, int],
    session_samples_count: int,
    validation_status: str,
    fps: float,
    participant_id: str = "user_1",
    v2_features: Optional[Dict[str, Any]] = None,
):
    """Renders the V2 Collector HUD overlay on the webcam frame."""
    h, w, _ = frame.shape

    # Semi-transparent background card on left side
    overlay = frame.copy()
    box_w, box_h = 360, 290
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + box_w, 15 + box_h), (70, 70, 70), 1)

    # Title
    cv2.putText(
        frame,
        "PostureGuard Collector V2 (4-Class)",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Status Line
    if state == "AUTO_RECORDING":
        status_text = f"RECORDING: {current_label.upper() if current_label else '--'}"
        status_color = (60, 240, 60)
    elif state in ("AUTO_PREP", "COUNTDOWN"):
        status_text = f"PREPARING: {current_label.upper() if current_label else '--'}"
        status_color = (0, 215, 255)
    elif state == "AUTO_TRANSITION":
        status_text = "NEXT POSE UPCOMING..."
        status_color = (255, 180, 50)
    elif state == "AUTO_COMPLETE":
        status_text = "ROUND COMPLETE! [Enter] Repeat"
        status_color = (80, 255, 80)
    elif state == "PAUSED":
        status_text = "PAUSED (Press Space)"
        status_color = (0, 165, 255)
    else:
        status_text = "INITIALIZING..."
        status_color = (200, 200, 200)

    cv2.putText(frame, status_text, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA)

    # Timer and Pose Counter
    pose_display_num = current_pose_idx + 1 if current_pose_idx < len(AUTO_POSES_V2) else len(AUTO_POSES_V2)
    cv2.putText(
        frame,
        f"Pose {pose_display_num}/4  |  Timer: {timer_remaining:4.1f}s",
        (25, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Progress Bar
    bar_x, bar_y, bar_w, bar_h = 25, 102, box_w - 20, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_ratio = min(1.0, current_pose_samples / max(1, target_pose_samples))
    fill_w = int(fill_ratio * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (60, 240, 60), -1)

    # Metrics Display (V2 Features)
    hr_str = f"{v2_features.get('head_to_shoulder_ratio', '--'):.3f}" if v2_features and v2_features.get('head_to_shoulder_ratio') is not None else "--"
    tilt_str = f"{v2_features.get('shoulder_tilt_deg', '--'):+.1f} deg" if v2_features and v2_features.get('shoulder_tilt_deg') is not None else "--"
    neck_str = f"{v2_features.get('neck_angle_deg', '--'):.1f} deg" if v2_features and v2_features.get('neck_angle_deg') is not None else "--"
    sw_str = f"{v2_features.get('shoulder_width', '--'):.3f}" if v2_features and v2_features.get('shoulder_width') is not None else "--"

    metrics_y = 135
    cv2.putText(frame, f"Head Ratio : {hr_str}", (25, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tilt       : {tilt_str}", (200, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Neck Angle : {neck_str}", (25, metrics_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Sh. Width  : {sw_str}", (200, metrics_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # Class sample counts summary (4 classes)
    c_y = 188
    cv2.putText(
        frame,
        f"Correct: {class_counts.get('correct', 0)}  |  Slouch: {class_counts.get('slouching', 0)}",
        (25, c_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Lean L : {class_counts.get('leaning_left', 0)}  |  Lean R : {class_counts.get('leaning_right', 0)}",
        (25, c_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # Participant & Session & Status
    cv2.putText(
        frame,
        f"Participant: {participant_id} | Added: {session_samples_count}",
        (25, c_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 215, 0),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Status: {validation_status[:38]}",
        (25, c_y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (80, 255, 120) if "Saved" in validation_status else (240, 140, 40),
        1,
        cv2.LINE_AA
    )

    # Bottom Instructions Bar
    bottom_overlay = frame.copy()
    cv2.rectangle(bottom_overlay, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(bottom_overlay, 0.75, frame, 0.25, 0, frame)
    instruction_text = "Controls: [Space] Pause/Resume  |  [N] Next Pose  |  [R] Redo  |  [1-4] Jump  |  [Q] Quit"
    cv2.putText(
        frame,
        instruction_text,
        (15, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2.LINE_AA
    )


# =========================================================================
# Main Guided Collection Workflow
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="PostureGuard Dataset Collector V2")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_DATASET_V2_PATH),
        help="Target CSV dataset path. Defaults to data/posture_data_v2.csv."
    )
    parser.add_argument(
        "--participant",
        type=str,
        default="user_1",
        help="Participant identifier (e.g. user_1, subject_2)."
    )
    args = parser.parse_args()
    target_csv = Path(args.output)
    participant_id = args.participant.strip()

    print("=" * 68)
    print("PostureGuard Hands-Free Dataset Collector V2 (4-Class Mode)")
    print(f"Target CSV dataset: {target_csv}")
    print(f"Participant ID:     {participant_id}")
    print("Captures raw landmarks + scale-invariant head_to_shoulder_ratio.")
    print("=" * 68)

    # Initialize / validate dataset CSV with legacy protection
    try:
        class_counts = initialize_dataset_file_v2(target_csv)
        print("Existing samples loaded from V2 CSV:")
        for lbl in FOUR_CLASS_LABELS:
            print(f"  - {lbl:<16}: {class_counts[lbl]} samples")
    except Exception as e:
        print(f"\nError initializing dataset CSV:\n{e}", file=sys.stderr)
        sys.exit(1)

    # Unique session identifier and duplicate detector
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Active Session ID:  {session_id}\n")
    duplicate_detector = DuplicateFrameDetector(precision=6)

    # Initialize OpenCV webcam with DirectShow fallback
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not access webcam. Ensure no other application is using it.", file=sys.stderr)
        sys.exit(1)

    # State machine variables
    state = "AUTO_INTRO"
    state_start_time = time.time()
    current_pose_idx = 0
    current_label = AUTO_POSES_V2[0]["label"]
    current_pose_samples = 0
    session_samples_count = 0
    frame_index = 0
    last_sample_time = 0.0
    last_tick_int = -1
    validation_status = "Ready"
    prev_frame_time = time.time()
    fps = 0.0

    print("Starting Automated Guided Collection in 4 seconds...")
    print("Follow the on-screen instructions. No keyboard clicks needed during recording.")
    play_sound_async("tick")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to grab frame from camera. Exiting...")
                    break

                curr_time = time.time()
                dt = curr_time - prev_frame_time
                prev_frame_time = curr_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # MediaPipe inference
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                row_data = None
                is_valid = False

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    landmarks = extract_key_landmarks(results.pose_landmarks, min_visibility=0.40)
                    is_valid, row_data, validation_msg = validate_and_prepare_v2(landmarks, min_visibility=0.40)
                    validation_status = validation_msg
                else:
                    landmarks = None
                    validation_status = "No person detected"

                # State machine
                elapsed_in_state = curr_time - state_start_time
                timer_remaining = 0.0

                if state == "AUTO_INTRO":
                    intro_duration = 4.0
                    timer_remaining = max(0.0, intro_duration - elapsed_in_state)
                    cur_int = int(math.ceil(timer_remaining))
                    if cur_int != last_tick_int and cur_int > 0:
                        play_sound_async("tick")
                        last_tick_int = cur_int

                    if timer_remaining <= 0.0:
                        state = "AUTO_PREP"
                        state_start_time = curr_time
                        current_pose_samples = 0
                        last_tick_int = -1
                        play_sound_async("start")
                        print(f"\n[NEXT POSE] {AUTO_POSES_V2[current_pose_idx]['title']}: {AUTO_POSES_V2[current_pose_idx]['instruction']}")

                elif state == "AUTO_PREP":
                    pose_info = AUTO_POSES_V2[current_pose_idx]
                    timer_remaining = max(0.0, pose_info["prep_sec"] - elapsed_in_state)
                    cur_int = int(math.ceil(timer_remaining))
                    if cur_int != last_tick_int and cur_int > 0:
                        play_sound_async("tick")
                        last_tick_int = cur_int

                    if timer_remaining <= 0.0:
                        state = "AUTO_RECORDING"
                        state_start_time = curr_time
                        last_sample_time = curr_time
                        current_pose_samples = 0
                        duplicate_detector.reset()
                        play_sound_async("start")
                        print(f"[RECORDING STARTED] Class: '{current_label}'")

                elif state == "AUTO_RECORDING":
                    pose_info = AUTO_POSES_V2[current_pose_idx]
                    target_samples = pose_info["target_samples"]
                    timer_remaining = max(0.0, pose_info["record_sec"] - elapsed_in_state)

                    # Sampling interval rate limiter (~5 Hz)
                    if curr_time - last_sample_time >= SAMPLE_INTERVAL_SEC:
                        if is_valid and row_data is not None and current_label and landmarks:
                            nose = landmarks.get("nose")
                            ls = landmarks.get("left_shoulder")
                            rs = landmarks.get("right_shoulder")

                            # Check duplicate hardware buffer repeat
                            if duplicate_detector.is_duplicate(nose, ls, rs):
                                validation_status = "Skipped: Duplicate hardware frame repeat"
                            else:
                                utc_timestamp = datetime.now(timezone.utc).isoformat()
                                append_sample_v2(
                                    filepath=target_csv,
                                    row_data=row_data,
                                    label=current_label,
                                    participant_id=participant_id,
                                    session_id=session_id,
                                    frame_index=frame_index,
                                    timestamp=utc_timestamp
                                )
                                class_counts[current_label] += 1
                                session_samples_count += 1
                                current_pose_samples += 1
                                frame_index += 1
                                last_sample_time = curr_time
                                validation_status = "Sample Saved (+1)"
                        else:
                            validation_status = f"Skipped: {validation_status}"

                    # Advance pose
                    if timer_remaining <= 0.0 or current_pose_samples >= target_samples:
                        play_sound_async("pose_done")
                        print(f"[POSE COMPLETED] '{current_label}': {current_pose_samples} samples collected.")
                        if current_pose_idx + 1 < len(AUTO_POSES_V2):
                            state = "AUTO_TRANSITION"
                            state_start_time = curr_time
                            last_tick_int = -1
                        else:
                            state = "AUTO_COMPLETE"
                            state_start_time = curr_time
                            play_sound_async("session_complete")
                            print("\n[SESSION COMPLETE] All 4 posture classes recorded successfully!")

                elif state == "AUTO_TRANSITION":
                    timer_remaining = max(0.0, 3.0 - elapsed_in_state)
                    cur_int = int(math.ceil(timer_remaining))
                    if cur_int != last_tick_int and cur_int > 0:
                        play_sound_async("tick")
                        last_tick_int = cur_int

                    if timer_remaining <= 0.0:
                        current_pose_idx += 1
                        current_label = AUTO_POSES_V2[current_pose_idx]["label"]
                        state = "AUTO_PREP"
                        state_start_time = curr_time
                        current_pose_samples = 0
                        last_tick_int = -1
                        play_sound_async("start")
                        print(f"\n[NEXT POSE] {AUTO_POSES_V2[current_pose_idx]['title']}: {AUTO_POSES_V2[current_pose_idx]['instruction']}")

                elif state == "AUTO_COMPLETE":
                    timer_remaining = 0.0

                # Render HUD
                target_pose_samples = AUTO_POSES_V2[current_pose_idx]["target_samples"] if current_pose_idx < len(AUTO_POSES_V2) else 30
                draw_collector_hud(
                    frame=frame,
                    state=state,
                    current_pose_idx=current_pose_idx,
                    current_label=current_label,
                    timer_remaining=timer_remaining,
                    current_pose_samples=current_pose_samples,
                    target_pose_samples=target_pose_samples,
                    class_counts=class_counts,
                    session_samples_count=session_samples_count,
                    validation_status=validation_status,
                    fps=fps,
                    participant_id=participant_id,
                    v2_features=row_data
                )

                cv2.imshow("PostureGuard - Dataset Collector V2", frame)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nExit requested ('q'). Shutting down collector...")
                    break
                elif key in (32, ord("s"), ord("0")):
                    if state == "PAUSED":
                        state = "AUTO_PREP"
                        state_start_time = time.time()
                        last_tick_int = -1
                        print(f"[RESUMED] Resuming '{current_label}'...")
                    else:
                        state = "PAUSED"
                        print("[PAUSED] Recording paused. Press Space to resume.")
                elif key in (ord("n"), ord("N")):
                    if current_pose_idx + 1 < len(AUTO_POSES_V2):
                        current_pose_idx += 1
                        current_label = AUTO_POSES_V2[current_pose_idx]["label"]
                        state = "AUTO_PREP"
                        state_start_time = time.time()
                        current_pose_samples = 0
                        last_tick_int = -1
                        print(f"\n[SKIPPED] Moved to next pose: {AUTO_POSES_V2[current_pose_idx]['title']}")
                    else:
                        state = "AUTO_COMPLETE"
                elif key in (ord("r"), ord("R")):
                    state = "AUTO_PREP"
                    state_start_time = time.time()
                    current_pose_samples = 0
                    last_tick_int = -1
                    print(f"\n[RESTARTING] Redoing current pose: '{current_label}'")
                elif key == 13:
                    if state == "AUTO_COMPLETE":
                        current_pose_idx = 0
                        current_label = AUTO_POSES_V2[0]["label"]
                        state = "AUTO_PREP"
                        state_start_time = time.time()
                        current_pose_samples = 0
                        last_tick_int = -1
                        print("\n[NEW ROUND] Starting a new round across all 4 poses!")
                elif key in LABEL_MAP_V2:
                    current_label = LABEL_MAP_V2[key]
                    for i, p in enumerate(AUTO_POSES_V2):
                        if p["label"] == current_label:
                            current_pose_idx = i
                            break
                    state = "AUTO_PREP"
                    state_start_time = time.time()
                    current_pose_samples = 0
                    last_tick_int = -1
                    print(f"\n[MANUAL SELECT] Jumped to: '{current_label}'")
                elif key == ord("5"):
                    print("[NOTICE] 'leaning_forward' is disabled from 4-class collection.")

        except KeyboardInterrupt:
            print("\nCollector interrupted by user (Ctrl+C). Cleaning up...")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("=" * 68)
            print("Data Collection Summary:")
            print(f"Participant ID:             {participant_id}")
            print(f"Session ID:                 {session_id}")
            print(f"Samples added this session: {session_samples_count}")
            print("Current dataset totals:")
            for lbl in FOUR_CLASS_LABELS:
                print(f"  - {lbl:<16}: {class_counts[lbl]} samples")
            print(f"CSV saved at: {target_csv}")
            print("Webcam released and OpenCV display closed.")
            print("=" * 68)


if __name__ == "__main__":
    main()
