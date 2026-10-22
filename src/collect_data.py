"""
PostureGuard Dataset Collector (Auto-Guided & Laptop-Optimized)
---------------------------------------------------------------
Captures webcam frames, extracts MediaPipe upper-body landmarks (shoulders & head),
calculates posture metrics, and automatically guides you through collecting balanced
training samples into `data/posture_data.csv` for five sitting posture classes:

1. correct          (Sit upright, shoulders relaxed, look straight)
2. slouching        (Slouch back or hunch shoulders downward)
3. leaning_left     (Lean upper body and head towards your left)
4. leaning_right    (Lean upper body and head towards your right)
5. leaning_forward  (Lean upper body forward closer to the screen)

Key Features:
- Fully Automated Guided Mode: Hands-free sequence with automatic timers and audio cues!
- Laptop Close-up Optimized: Only shoulders and head are required. Hips/legs are
  gracefully treated as optional sentinels (-1.0) so you can sit at your normal laptop distance.
- Full Manual Overrides: Press [1]-[5] to select any pose, [Space] to pause/resume,
  [N] for next pose, [R] to redo, [Q] to quit safely.
"""

import csv
from datetime import datetime
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
    from src.landmarks import extract_key_landmarks
    from src.features import extract_posture_features
except ModuleNotFoundError:
    from landmarks import extract_key_landmarks
    from features import extract_posture_features


# Target CSV location
DATASET_CSV_PATH = PROJECT_ROOT / "data" / "posture_data.csv"

# Label mapping for keyboard triggers
LABEL_MAP = {
    ord("1"): "correct",
    ord("2"): "slouching",
    ord("3"): "leaning_left",
    ord("4"): "leaning_right",
    ord("5"): "leaning_forward",
}

# Automated Guided Pose Sequence
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

# Ordered feature columns matching src/features.py
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

# Required upper-body features visible at laptop distance (shoulders and head)
REQUIRED_CORE_FEATURES = [
    "shoulder_distance_raw",
    "shoulder_tilt_deg",
    "neck_angle_deg",
]

# Lower body, hip, and head-ratio features (optional when close to a laptop or desk)
# If missing or occluded, filled with -1.0 sentinel so CSV rows remain complete and NaN-free
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

# Metadata columns
METADATA_COLUMNS = ["label", "timestamp", "session_id"]
LEGACY_ALL_COLUMNS = LEGACY_FEATURE_COLUMNS + METADATA_COLUMNS
EXTENDED_ALL_COLUMNS = EXTENDED_FEATURE_COLUMNS + METADATA_COLUMNS
ALL_COLUMNS = EXTENDED_ALL_COLUMNS

# Sampling configuration
TARGET_SAMPLES_PER_SEC = 5.0
SAMPLE_INTERVAL_SEC = 1.0 / TARGET_SAMPLES_PER_SEC


def play_sound_async(sound_type: str = "tick"):
    """
    Plays non-blocking audio cues using winsound on Windows.
    Runs in a background daemon thread to ensure zero lag on OpenCV frame processing.
    """
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


def initialize_dataset_file(filepath: Path) -> Dict[str, int]:
    """
    Initializes the dataset CSV file with correct headers if it does not exist.
    If it exists, validates column compatibility and counts existing samples per label.
    """
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
    """
    Validates posture features optimized for laptop sitting distance.

    Rules:
    - Requires upper body (shoulders and head/neck) to be visible.
    - Hips, torso length, and legs are optional. If they are off-screen or
      hidden under a desk (None), they cleanly default to -1.0 sentinel values.
    - Returns False if person is not detected or values are NaN/Inf.
    """
    if not features:
        return False, None, "No person detected"

    cleaned: Dict[str, float] = {}

    # 1. Validate required upper-body features (head & shoulders)
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
        return False, None, f"Missing upper features"

    # 2. Process lower-body & hip features (optional at laptop distance)
    for col in OPTIONAL_DESK_FEATURES:
        val = features.get(col)
        if val is None:
            # Occluded by desk or below camera frame: default to -1.0 sentinel
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
    """Appends a single verified sample row to the dataset CSV matching its active schema."""
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


def draw_collector_hud(
    frame,
    state: str,
    current_pose_idx: int,
    current_label: Optional[str],
    timer_remaining: float,
    current_pose_samples: int,
    target_pose_samples: int,
    class_counts: Dict[str, int],
    session_samples_count: int,
    validation_status: str,
    fps: float
):
    """Renders a modern, highly legible HUD overlay on the webcam frame."""
    h, w, _ = frame.shape

    # Semi-transparent background card on left side
    overlay = frame.copy()
    box_w, box_h = 350, 275
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + box_w, 15 + box_h), (70, 70, 70), 1)

    # Title
    cv2.putText(
        frame,
        "PostureGuard Collector (Auto Mode)",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Status Line
    if state == "AUTO_RECORDING":
        status_text = f"RECORDING: {current_label.upper()}"
        status_color = (60, 240, 60)
    elif state in ("AUTO_PREP", "COUNTDOWN"):
        status_text = f"PREPARING: {current_label.upper()}"
        status_color = (0, 215, 255)
    elif state == "AUTO_TRANSITION":
        status_text = "NEXT POSE UPCOMING..."
        status_color = (255, 180, 50)
    elif state == "AUTO_COMPLETE":
        status_text = "ROUND COMPLETE! [Enter] to repeat"
        status_color = (80, 255, 80)
    elif state == "PAUSED":
        status_text = "PAUSED (Press Space to resume)"
        status_color = (180, 180, 180)
    else:
        status_text = f"INITIALIZING (Starting in {timer_remaining:.0f}s)"
        status_color = (0, 220, 255)

    cv2.putText(
        frame,
        status_text,
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        status_color,
        2,
        cv2.LINE_AA
    )

    # Frame Check
    val_color = (100, 255, 100) if validation_status in ("Valid", "Sample Saved (+1)") else (100, 120, 255)
    cv2.putText(
        frame,
        f"Tracking: {validation_status}",
        (25, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        val_color,
        1,
        cv2.LINE_AA
    )

    # Divider
    cv2.line(frame, (25, 96), (15 + box_w - 10, 96), (70, 70, 70), 1)

    # Pose Scoreboard
    y = 116
    for idx, pose in enumerate(AUTO_POSES):
        lbl = pose["label"]
        count = class_counts[lbl]
        is_active = (lbl == current_label and state in ("AUTO_PREP", "AUTO_RECORDING", "COUNTDOWN"))
        prefix = "-> " if is_active else "   "
        text = f"{prefix}[{idx+1}] {lbl:<15} : {count:>4} samples"
        color = (80, 255, 80) if is_active else (210, 210, 210)
        cv2.putText(
            frame,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA
        )
        y += 22

    # Session Total
    total_samples = sum(class_counts.values())
    cv2.putText(
        frame,
        f"Total: {total_samples} samples | Session: +{session_samples_count}",
        (25, y + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 215, 0),
        1,
        cv2.LINE_AA
    )

    # Bottom Instructions Bar
    bottom_overlay = frame.copy()
    cv2.rectangle(bottom_overlay, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(bottom_overlay, 0.75, frame, 0.25, 0, frame)
    instruction_text = "Controls: [Space] Pause/Resume  |  [N] Next Pose  |  [R] Redo  |  [1-5] Jump  |  [Q] Quit"
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

    # Big Center Banner for Automated Routine
    if state == "AUTO_INTRO":
        banner_w, banner_h = 480, 130
        bx = (w - banner_w) // 2
        by = (h - banner_h) // 2
        card = frame.copy()
        cv2.rectangle(card, (bx, by), (bx + banner_w, by + banner_h), (12, 12, 12), -1)
        cv2.addWeighted(card, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (0, 215, 255), 2)

        cv2.putText(
            frame,
            "AUTOMATED GUIDED SESSION",
            (bx + 55, by + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Sit comfortably at normal laptop distance",
            (bx + 40, by + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        countdown_num = int(math.ceil(timer_remaining))
        cv2.putText(
            frame,
            f"Starting in {countdown_num}...",
            (bx + 160, by + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 215, 255),
            2,
            cv2.LINE_AA
        )

    elif state in ("AUTO_PREP", "COUNTDOWN"):
        pose_info = AUTO_POSES[current_pose_idx] if 0 <= current_pose_idx < len(AUTO_POSES) else {}
        title = pose_info.get("title", f"Prepare: {current_label}")
        inst = pose_info.get("instruction", "Get into position")

        banner_w, banner_h = 520, 140
        bx = (w - banner_w) // 2
        by = (h - banner_h) // 2
        card = frame.copy()
        cv2.rectangle(card, (bx, by), (bx + banner_w, by + banner_h), (12, 12, 12), -1)
        cv2.addWeighted(card, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (0, 215, 255), 2)

        cv2.putText(
            frame,
            f"GET READY -> {title.upper()}",
            (bx + 25, by + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            inst,
            (bx + 25, by + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        countdown_num = int(math.ceil(timer_remaining))
        cv2.putText(
            frame,
            f"Recording in {countdown_num}...",
            (bx + 25, by + 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 215, 255),
            2,
            cv2.LINE_AA
        )

    elif state == "AUTO_RECORDING":
        pose_info = AUTO_POSES[current_pose_idx] if 0 <= current_pose_idx < len(AUTO_POSES) else {}
        title = pose_info.get("title", f"Recording: {current_label}")
        inst = pose_info.get("instruction", "Hold posture steady")

        # Top banner for active recording with progress bar
        banner_w, banner_h = 480, 95
        bx = (w - banner_w) // 2
        by = 25
        card = frame.copy()
        cv2.rectangle(card, (bx, by), (bx + banner_w, by + banner_h), (10, 30, 10), -1)
        cv2.addWeighted(card, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (60, 240, 60), 2)

        cv2.putText(
            frame,
            f"RECORDING: {title.upper()}",
            (bx + 20, by + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 255, 80),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            inst,
            (bx + 20, by + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA
        )

        # Progress bar
        bar_x = bx + 20
        bar_y = by + 68
        bar_w = banner_w - 40
        bar_h = 14
        progress = min(1.0, max(0.0, current_pose_samples / max(1, target_pose_samples)))
        fill_w = int(bar_w * progress)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (60, 230, 60), -1)
        cv2.putText(
            frame,
            f"{current_pose_samples}/{target_pose_samples} samples ({timer_remaining:.1f}s)",
            (bar_x + bar_w - 150, bar_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    elif state == "AUTO_TRANSITION":
        banner_w, banner_h = 440, 100
        bx = (w - banner_w) // 2
        by = (h - banner_h) // 2
        card = frame.copy()
        cv2.rectangle(card, (bx, by), (bx + banner_w, by + banner_h), (15, 15, 15), -1)
        cv2.addWeighted(card, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (255, 180, 50), 2)

        next_idx = current_pose_idx + 1
        next_name = AUTO_POSES[next_idx]["title"] if next_idx < len(AUTO_POSES) else "Wrap-up"

        cv2.putText(
            frame,
            "POSE COMPLETE! (+Samples Saved)",
            (bx + 40, by + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (100, 255, 100),
            2,
            cv2.LINE_AA
        )
        countdown_num = int(math.ceil(timer_remaining))
        cv2.putText(
            frame,
            f"Next: {next_name} in {countdown_num}s...",
            (bx + 40, by + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 80),
            2,
            cv2.LINE_AA
        )

    elif state == "AUTO_COMPLETE":
        banner_w, banner_h = 500, 130
        bx = (w - banner_w) // 2
        by = (h - banner_h) // 2
        card = frame.copy()
        cv2.rectangle(card, (bx, by), (bx + banner_w, by + banner_h), (10, 25, 10), -1)
        cv2.addWeighted(card, 0.90, frame, 0.10, 0, frame)
        cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (80, 255, 80), 2)

        cv2.putText(
            frame,
            "ALL 5 POSES COLLECTED SUCCESSFULLY!",
            (bx + 25, by + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Added +{session_samples_count} clean samples to dataset.",
            (bx + 25, by + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 240, 200),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "[Enter] Start Another Round   |   [Q] Quit & Save",
            (bx + 25, by + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 215, 255),
            2,
            cv2.LINE_AA
        )


def main():
    print("=" * 68)
    print("PostureGuard Hands-Free Dataset Collector (Auto Mode)")
    print(f"Target CSV dataset: {DATASET_CSV_PATH}")
    print("Optimized for laptop distance: Only shoulders and head are required.")
    print("=" * 68)

    # Initialize / validate dataset CSV
    try:
        class_counts = initialize_dataset_file(DATASET_CSV_PATH)
        print("Existing samples loaded from CSV:")
        for pose in AUTO_POSES:
            lbl = pose["label"]
            print(f"  - {lbl:<16}: {class_counts[lbl]} samples")
    except Exception as e:
        print(f"Error initializing dataset CSV: {e}")
        sys.exit(1)

    # Generate unique session identifier
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Active Session ID: {session_id}")

    # Initialize OpenCV webcam with robust backend fallback (DirectShow on Windows avoids MSMF/FFMPEG/obsensor errors)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("=" * 60)
        print("Error: Could not access the webcam.")
        print("Troubleshooting tips:")
        print(" 1. Ensure no other app (Zoom, Teams, Chrome, Windows Camera) is using the camera.")
        print(" 2. Check Windows Settings -> Privacy & Security -> Camera -> Allow apps to access camera.")
        print(" 3. Verify your webcam connection.")
        print("=" * 60)
        sys.exit(1)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Automated state machine variables
    state = "AUTO_INTRO"  # 'AUTO_INTRO', 'AUTO_PREP', 'AUTO_RECORDING', 'AUTO_TRANSITION', 'AUTO_COMPLETE', 'PAUSED'
    current_pose_idx = 0
    current_label = AUTO_POSES[0]["label"]
    state_start_time = time.time()
    last_sample_time = 0.0
    session_samples_count = 0
    current_pose_samples = 0
    validation_status = "Upper body tracking ready"
    last_tick_int = -1

    prev_frame_time = time.time()
    fps = 0.0

    print("\nStarting Automated Guided Collection in 4 seconds...")
    print("No button clicks required! Sit in front of your laptop and follow the on-screen prompts.")
    play_sound_async("tick")

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
                fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_frame_time, 1e-4))
                prev_frame_time = curr_time

                # Convert to RGB for MediaPipe inference
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                # Draw skeleton landmarks on webcam feed
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    # Extract landmarks (min_visibility 0.40 for robust head/shoulder tracking)
                    landmarks = extract_key_landmarks(results.pose_landmarks, min_visibility=0.40)
                    features = extract_posture_features(landmarks)
                    is_valid, cleaned_features, validation_msg = validate_and_prepare_features(features)
                    validation_status = validation_msg
                else:
                    landmarks = None
                    features = None
                    cleaned_features = None
                    is_valid = False
                    validation_status = "No person detected"

                # Calculate timer for the current state
                elapsed_in_state = curr_time - state_start_time
                timer_remaining = 0.0

                # ----------------- Automated State Machine -----------------
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
                        current_pose_info = AUTO_POSES[current_pose_idx]
                        print(f"\n[NEXT POSE] {current_pose_info['title']}: {current_pose_info['instruction']}")

                elif state == "AUTO_PREP":
                    current_pose_info = AUTO_POSES[current_pose_idx]
                    prep_duration = current_pose_info["prep_sec"]
                    timer_remaining = max(0.0, prep_duration - elapsed_in_state)
                    cur_int = int(math.ceil(timer_remaining))
                    if cur_int != last_tick_int and cur_int > 0:
                        play_sound_async("tick")
                        last_tick_int = cur_int

                    if timer_remaining <= 0.0:
                        state = "AUTO_RECORDING"
                        state_start_time = curr_time
                        last_sample_time = curr_time
                        current_pose_samples = 0
                        play_sound_async("start")
                        print(f"[RECORDING STARTED] Class: '{current_label}'")

                elif state == "AUTO_RECORDING":
                    current_pose_info = AUTO_POSES[current_pose_idx]
                    record_duration = current_pose_info["record_sec"]
                    target_samples = current_pose_info["target_samples"]
                    timer_remaining = max(0.0, record_duration - elapsed_in_state)

                    # Sample rate limiting (~5 samples per second)
                    if curr_time - last_sample_time >= SAMPLE_INTERVAL_SEC:
                        if is_valid and cleaned_features is not None and current_label:
                            ts_str = datetime.now().isoformat()
                            append_sample(
                                DATASET_CSV_PATH,
                                cleaned_features,
                                current_label,
                                session_id,
                                ts_str
                            )
                            class_counts[current_label] += 1
                            session_samples_count += 1
                            current_pose_samples += 1
                            last_sample_time = curr_time
                            validation_status = "Sample Saved (+1)"
                        else:
                            validation_status = f"Skipped: {validation_status}"

                    # Advance to next pose when time or sample target is reached
                    if timer_remaining <= 0.0 or current_pose_samples >= target_samples:
                        play_sound_async("pose_done")
                        print(f"[POSE COMPLETED] '{current_label}': {current_pose_samples} samples collected.")
                        if current_pose_idx + 1 < len(AUTO_POSES):
                            state = "AUTO_TRANSITION"
                            state_start_time = curr_time
                            last_tick_int = -1
                        else:
                            state = "AUTO_COMPLETE"
                            state_start_time = curr_time
                            play_sound_async("session_complete")
                            print("\n[SESSION COMPLETE] All 5 posture classes recorded successfully!")

                elif state == "AUTO_TRANSITION":
                    trans_duration = 3.0
                    timer_remaining = max(0.0, trans_duration - elapsed_in_state)
                    cur_int = int(math.ceil(timer_remaining))
                    if cur_int != last_tick_int and cur_int > 0:
                        play_sound_async("tick")
                        last_tick_int = cur_int

                    if timer_remaining <= 0.0:
                        current_pose_idx += 1
                        current_label = AUTO_POSES[current_pose_idx]["label"]
                        state = "AUTO_PREP"
                        state_start_time = curr_time
                        current_pose_samples = 0
                        last_tick_int = -1
                        play_sound_async("start")
                        current_pose_info = AUTO_POSES[current_pose_idx]
                        print(f"\n[NEXT POSE] {current_pose_info['title']}: {current_pose_info['instruction']}")

                elif state == "AUTO_COMPLETE":
                    timer_remaining = 0.0

                elif state == "COUNTDOWN":
                    # Manual countdown fallback
                    countdown_duration = 3.0
                    timer_remaining = max(0.0, countdown_duration - elapsed_in_state)
                    if timer_remaining <= 0.0:
                        state = "AUTO_RECORDING"
                        state_start_time = curr_time
                        last_sample_time = curr_time
                        current_pose_samples = 0
                        play_sound_async("start")

                # Target samples for progress bar
                target_pose_samples = (
                    AUTO_POSES[current_pose_idx]["target_samples"]
                    if 0 <= current_pose_idx < len(AUTO_POSES)
                    else 30
                )

                # Render Collector HUD
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
                    fps=fps
                )

                cv2.imshow("PostureGuard - Dataset Collector", frame)

                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nExit requested ('q'). Shutting down collector...")
                    break

                elif key in (32, ord("s"), ord("0")):  # Spacebar, 's', or '0' to pause/resume
                    if state == "PAUSED":
                        state = "AUTO_PREP"
                        state_start_time = time.time()
                        last_tick_int = -1
                        print(f"[RESUMED] Resuming '{current_label}'...")
                    else:
                        state = "PAUSED"
                        print(f"[PAUSED] Recording paused. Press Space to resume.")

                elif key in (ord("n"), ord("N")):  # Skip to next pose
                    if current_pose_idx + 1 < len(AUTO_POSES):
                        current_pose_idx += 1
                        current_label = AUTO_POSES[current_pose_idx]["label"]
                        state = "AUTO_PREP"
                        state_start_time = time.time()
                        current_pose_samples = 0
                        last_tick_int = -1
                        print(f"\n[SKIPPED] Moved to next pose: {AUTO_POSES[current_pose_idx]['title']}")
                    else:
                        state = "AUTO_COMPLETE"

                elif key in (ord("r"), ord("R")):  # Redo current pose
                    state = "AUTO_PREP"
                    state_start_time = time.time()
                    current_pose_samples = 0
                    last_tick_int = -1
                    print(f"\n[RESTARTING] Redoing current pose: '{current_label}'")

                elif key == 13:  # Enter key: Start another round if completed
                    if state == "AUTO_COMPLETE":
                        current_pose_idx = 0
                        current_label = AUTO_POSES[0]["label"]
                        state = "AUTO_PREP"
                        state_start_time = time.time()
                        current_pose_samples = 0
                        last_tick_int = -1
                        print("\n[NEW ROUND] Starting a new round across all 5 poses!")

                elif key in LABEL_MAP:  # Direct numeric hotkey [1]-[5]
                    selected_label = LABEL_MAP[key]
                    # Find matching index in AUTO_POSES
                    for i, p in enumerate(AUTO_POSES):
                        if p["label"] == selected_label:
                            current_pose_idx = i
                            break
                    current_label = selected_label
                    state = "AUTO_PREP"
                    state_start_time = time.time()
                    current_pose_samples = 0
                    last_tick_int = -1
                    print(f"\n[MANUAL SELECT] Jumped to: '{current_label}'")

        except KeyboardInterrupt:
            print("\nCollector interrupted by user (Ctrl+C). Cleaning up...")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("=" * 68)
            print("Data Collection Summary:")
            print(f"Session ID: {session_id}")
            print(f"Samples added this session: {session_samples_count}")
            print("Current dataset totals:")
            for pose in AUTO_POSES:
                lbl = pose["label"]
                print(f"  - {lbl:<16}: {class_counts[lbl]} samples")
            print(f"CSV saved at: {DATASET_CSV_PATH}")
            print("Webcam released and OpenCV display closed.")
            print("=" * 68)


if __name__ == "__main__":
    main()
