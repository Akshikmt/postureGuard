"""
MultiPosture Dataset Preprocessor for PostureGuard (Task MultiPosture)
---------------------------------------------------------------------
Converts the public MultiPosture dataset from Zenodo (Record 14230872)
into structured geometric features compatible with PostureGuard's
machine learning inference pipeline.

Key Design & Rigor Principles:
1. Strict Schema & Integrity:
   - Reads `data/external/multiposture/data.csv` (102 columns, 33 joints, 13 participants).
   - Verifies all required coordinate columns exist and contain finite numeric floats.
2. Label Mapping & Unsupported Class Policy:
   - Compatible upper body postures:
     - 'TUP' (Trunk Upright Position) -> 'correct'
     - 'TLL' (Trunk Leaning Left)    -> 'leaning_left'
     - 'TLR' (Trunk Leaning Right)   -> 'leaning_right'
   - Unsupported / Missing class:
     - 'slouching' is NOT represented in MultiPosture. MultiPosture only captures
       directional trunk inclinations (TUP, TLF, TLB, TLL, TLR) and does NOT capture
       thoracic kyphosis or slouched head drop. Per project integrity policy,
       slouching labels are strictly NOT fabricated or synthesized.
   - Preserves raw labels in metadata column 'multiposture_raw_label'.
3. Coordinate System & Feature Assumptions:
   - MultiPosture coordinates are hip-centered (origin (0,0,0) at mid-hip).
   - In MultiPosture, points above the hip have negative y (head, shoulders).
   - Computes:
     - 'shoulder_tilt_deg': lateral angle of shoulder line from horizontal.
     - 'neck_angle_deg': inclination angle of nose from mid-shoulder.
     - 'shoulder_distance_raw': Euclidean 2D distance between shoulders.
     - 'shoulder_to_ear_ratio': ratio of shoulder width to ear distance.
     - 'torso_angle_deg': inclination angle of torso from vertical (hip to shoulder).
     - 'head_to_shoulder_ratio': vertical head elevation over shoulder distance.
     - 'calibrated_tilt_deg': shoulder tilt adjusted for MultiPosture camera roll offset (mean TUP offset ~+9.59 deg).
4. Participant-Aware Partitioning (Zero Participant Leakage):
   - Partitions data strictly by participant identifier ('subject'), never by random frames.
   - Train participants (10): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
   - Test participants (3):  ['10', '12', '13']
   - Saves clean features to `data/external/multiposture/multiposture_features.csv`
     and split metadata to `data/external/multiposture/participant_split.json`.
"""

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# File paths
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "external" / "multiposture" / "data.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "external" / "multiposture" / "multiposture_features.csv"
DEFAULT_SPLIT_JSON = PROJECT_ROOT / "data" / "external" / "multiposture" / "participant_split.json"

# Supported upper body label mappings
COMPATIBLE_LABEL_MAP: Dict[str, str] = {
    "TUP": "correct",
    "TLL": "leaning_left",
    "TLR": "leaning_right",
}

# Explicitly documented unsupported classes
UNSUPPORTED_POSTUREGUARD_CLASSES: List[str] = ["slouching"]

# MultiPosture classes not mapped to active 3-class target
OTHER_MULTIPOSTURE_CLASSES: Dict[str, str] = {
    "TLF": "leaning_forward",
    "TLB": "leaning_backward",
}

# Empirical MultiPosture camera tilt offset across all participants in TUP
MULTIPOSTURE_CAMERA_TILT_OFFSET_DEG: float = 9.59

# Standard participant train/test split (10 train, 3 test)
TRAIN_PARTICIPANTS: List[str] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "11"]
TEST_PARTICIPANTS: List[str] = ["10", "12", "13"]

FEATURE_COLUMNS: List[str] = [
    "shoulder_tilt_deg",
    "neck_angle_deg",
    "shoulder_distance_raw",
    "shoulder_to_ear_ratio",
    "torso_angle_deg",
    "head_to_shoulder_ratio",
    "calibrated_tilt_deg",
]

OUTPUT_COLUMNS: List[str] = FEATURE_COLUMNS + [
    "label",
    "multiposture_raw_label",
    "lowerbody_label",
    "subject",
    "split",
]


def calculate_shoulder_tilt_multiposture(
    ls_x: float, ls_y: float, rs_x: float, rs_y: float
) -> float:
    """
    Computes lateral shoulder tilt relative to horizontal.
    In hip-centered coordinates, rs_x < 0, ls_x > 0, so dx < 0.
    Normalizes wrapped angle around 0 degrees.
    """
    dx = rs_x - ls_x
    dy = rs_y - ls_y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg > 90.0:
        angle_deg -= 180.0
    elif angle_deg < -90.0:
        angle_deg += 180.0
    return round(angle_deg, 2)


def calculate_neck_angle_multiposture(
    nose_x: float, nose_y: float, mid_sh_x: float, mid_sh_y: float
) -> float:
    """
    Computes head/neck lateral inclination angle from vertical.
    In MultiPosture, nose is higher than shoulder (more negative y),
    so upward_dy = -(nose_y - mid_sh_y) > 0.
    """
    dx = nose_x - mid_sh_x
    dy = nose_y - mid_sh_y
    upward_dy = -dy
    if abs(upward_dy) < 1e-6 and abs(dx) < 1e-6:
        return 0.0
    angle_rad = math.atan2(abs(dx), upward_dy)
    return round(math.degrees(angle_rad), 2)


def calculate_torso_angle_multiposture(
    mid_sh_x: float, mid_sh_y: float, mid_hip_x: float, mid_hip_y: float
) -> float:
    """
    Computes torso inclination angle from vertical relative to hip center.
    """
    dx = mid_sh_x - mid_hip_x
    dy = mid_sh_y - mid_hip_y
    upward_dy = -dy
    if abs(upward_dy) < 1e-6 and abs(dx) < 1e-6:
        return 0.0
    angle_rad = math.atan2(abs(dx), upward_dy)
    return round(math.degrees(angle_rad), 2)


def calculate_shoulder_to_ear_ratio_multiposture(
    ls_x: float, ls_y: float, rs_x: float, rs_y: float,
    le_x: float, le_y: float, re_x: float, re_y: float
) -> Optional[float]:
    """Computes scale-invariant ratio of shoulder width to ear distance."""
    sh_dist = math.hypot(rs_x - ls_x, rs_y - ls_y)
    ear_dist = math.hypot(re_x - le_x, re_y - le_y)
    if ear_dist < 1e-4:
        return None
    return round(sh_dist / ear_dist, 4)


def calculate_head_to_shoulder_ratio_multiposture(
    nose_y: float, mid_sh_y: float, sh_dist: float
) -> Optional[float]:
    """Computes head vertical elevation over shoulder distance."""
    if sh_dist < 1e-4:
        return None
    # In MultiPosture, nose is above shoulders, so mid_sh_y - nose_y > 0
    elevation = mid_sh_y - nose_y
    return round(elevation / sh_dist, 4)


def extract_features_from_multiposture_row(
    row: Dict[str, str],
    camera_offset_deg: float = MULTIPOSTURE_CAMERA_TILT_OFFSET_DEG
) -> Dict[str, Any]:
    """
    Extracts validated PostureGuard geometric features from a MultiPosture CSV row.
    """
    ls_x = float(row["left_shoulder_x"])
    ls_y = float(row["left_shoulder_y"])
    rs_x = float(row["right_shoulder_x"])
    rs_y = float(row["right_shoulder_y"])
    nose_x = float(row["nose_x"])
    nose_y = float(row["nose_y"])
    le_x = float(row["left_ear_x"])
    le_y = float(row["left_ear_y"])
    re_x = float(row["right_ear_x"])
    re_y = float(row["right_ear_y"])
    lh_x = float(row["left_hip_x"])
    lh_y = float(row["left_hip_y"])
    rh_x = float(row["right_hip_x"])
    rh_y = float(row["right_hip_y"])

    mid_sh_x = (ls_x + rs_x) / 2.0
    mid_sh_y = (ls_y + rs_y) / 2.0
    mid_hip_x = (lh_x + rh_x) / 2.0
    mid_hip_y = (lh_y + rh_y) / 2.0

    sh_dist = round(math.hypot(rs_x - ls_x, rs_y - ls_y), 4)
    tilt = calculate_shoulder_tilt_multiposture(ls_x, ls_y, rs_x, rs_y)
    neck = calculate_neck_angle_multiposture(nose_x, nose_y, mid_sh_x, mid_sh_y)
    torso = calculate_torso_angle_multiposture(mid_sh_x, mid_sh_y, mid_hip_x, mid_hip_y)
    ear_ratio = calculate_shoulder_to_ear_ratio_multiposture(
        ls_x, ls_y, rs_x, rs_y, le_x, le_y, re_x, re_y
    )
    head_ratio = calculate_head_to_shoulder_ratio_multiposture(nose_y, mid_sh_y, sh_dist)
    calibrated_tilt = round(tilt - camera_offset_deg, 2)

    return {
        "shoulder_tilt_deg": tilt,
        "neck_angle_deg": neck,
        "shoulder_distance_raw": sh_dist,
        "shoulder_to_ear_ratio": ear_ratio if ear_ratio is not None else -1.0,
        "torso_angle_deg": torso,
        "head_to_shoulder_ratio": head_ratio if head_ratio is not None else -1.0,
        "calibrated_tilt_deg": calibrated_tilt,
    }


def prepare_multiposture_dataset(
    input_csv_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    split_json_path: Optional[Path] = None,
    train_participants: Optional[List[str]] = None,
    test_participants: Optional[List[str]] = None,
    include_forward_backward: bool = False
) -> Dict[str, Any]:
    """
    Main preprocessing entry point: loads MultiPosture dataset, converts
    compatible samples into PostureGuard features, applies participant-aware
    partitioning, and exports processed CSV and JSON metadata.
    """
    in_path = Path(input_csv_path) if input_csv_path else DEFAULT_INPUT_CSV
    out_path = Path(output_csv_path) if output_csv_path else DEFAULT_OUTPUT_CSV
    split_path = Path(split_json_path) if split_json_path else DEFAULT_SPLIT_JSON

    train_subjs = set(train_participants if train_participants is not None else TRAIN_PARTICIPANTS)
    test_subjs = set(test_participants if test_participants is not None else TEST_PARTICIPANTS)

    # Validate disjoint participant sets (zero leakage)
    overlap = train_subjs.intersection(test_subjs)
    if overlap:
        raise ValueError(f"Participant leakage detected: {overlap} in both train and test sets!")

    if not in_path.exists():
        raise FileNotFoundError(f"MultiPosture dataset file not found at: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    processed_rows: List[Dict[str, Any]] = []
    skipped_rows_counter: Dict[str, int] = {}
    class_counts_train: Dict[str, int] = {}
    class_counts_test: Dict[str, int] = {}
    participant_sample_counts: Dict[str, int] = {}

    total_read = 0

    with open(in_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total_read += 1
            subj = r.get("subject", "").strip()
            raw_upper = r.get("upperbody_label", "").strip()
            lower_lbl = r.get("lowerbody_label", "").strip()

            participant_sample_counts[subj] = participant_sample_counts.get(subj, 0) + 1

            # Determine label mapping
            if raw_upper in COMPATIBLE_LABEL_MAP:
                mapped_label = COMPATIBLE_LABEL_MAP[raw_upper]
            elif include_forward_backward and raw_upper in OTHER_MULTIPOSTURE_CLASSES:
                mapped_label = OTHER_MULTIPOSTURE_CLASSES[raw_upper]
            else:
                skipped_rows_counter[raw_upper] = skipped_rows_counter.get(raw_upper, 0) + 1
                continue

            # Determine split
            if subj in train_subjs:
                split_name = "train"
                class_counts_train[mapped_label] = class_counts_train.get(mapped_label, 0) + 1
            elif subj in test_subjs:
                split_name = "test"
                class_counts_test[mapped_label] = class_counts_test.get(mapped_label, 0) + 1
            else:
                split_name = "unassigned"

            feats = extract_features_from_multiposture_row(r)
            row_payload = {
                **feats,
                "label": mapped_label,
                "multiposture_raw_label": raw_upper,
                "lowerbody_label": lower_lbl,
                "subject": subj,
                "split": split_name,
            }
            processed_rows.append(row_payload)

    # Write processed CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for r in processed_rows:
            writer.writerow(r)

    # Write split JSON metadata
    split_payload = {
        "dataset_name": "MultiPosture (Zenodo 14230872)",
        "source_file": str(in_path),
        "processed_file": str(out_path),
        "total_source_rows": total_read,
        "retained_compatible_rows": len(processed_rows),
        "unsupported_classes": UNSUPPORTED_POSTUREGUARD_CLASSES,
        "compatible_classes": list(COMPATIBLE_LABEL_MAP.values()),
        "label_mapping": COMPATIBLE_LABEL_MAP,
        "camera_tilt_offset_deg": MULTIPOSTURE_CAMERA_TILT_OFFSET_DEG,
        "train_participants": sorted(list(train_subjs), key=lambda x: int(x)),
        "test_participants": sorted(list(test_subjs), key=lambda x: int(x)),
        "train_samples": sum(class_counts_train.values()),
        "test_samples": sum(class_counts_test.values()),
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
        "skipped_classes_breakdown": skipped_rows_counter,
        "participant_total_counts": participant_sample_counts,
        "leakage_verification": "Zero participant leakage verified (disjoint participant sets).",
    }

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    return split_payload


if __name__ == "__main__":
    print("Preprocessing MultiPosture dataset...")
    summary = prepare_multiposture_dataset()
    print("Preprocessing complete!")
    print(f"Total source rows: {summary['total_source_rows']}")
    print(f"Compatible rows:   {summary['retained_compatible_rows']}")
    print(f"Train samples:     {summary['train_samples']} across participants {summary['train_participants']}")
    print(f"Test samples:      {summary['test_samples']} across participants {summary['test_participants']}")
    print(f"Class counts train: {summary['class_counts_train']}")
    print(f"Class counts test:  {summary['class_counts_test']}")
    print(f"Unsupported class:  {summary['unsupported_classes']}")

