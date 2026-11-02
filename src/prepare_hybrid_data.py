"""
Hybrid Dataset Preparation Pipeline for PostureGuard (Phase 1)
--------------------------------------------------------------
Combines the multi-participant public MultiPosture dataset (Zenodo 14230872)
with PostureGuard's single-participant slouching recordings.

Data Sources:
1. MultiPosture:
   - Provides: 'correct', 'leaning_left', 'leaning_right'.
   - 13 participants with full 3D coordinates.
   - Camera tilt offset (+9.59 deg) calibrated to align with flat laptop webcams.
   - Participant-aware partitioning:
     - Train: Participants ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11'] (N=2034)
     - Test:  Participants ['10', '12', '13'] (N=421)
2. PostureGuard:
   - Provides: 'slouching' exclusively (unsupported by MultiPosture).
   - 1 participant across 3 sessions.
   - Session-aware partitioning:
     - Train: Sessions ['session_20261021_195509', 'session_20261021_210838'] (N=50)
     - Test:  Session  ['session_20261021_201127'] (N=25)
   - IMPORTANT EVALUATION DISTINCTION:
     PostureGuard test set evaluates an unseen session from the SAME participant,
     NOT an unseen participant. Generalization across participants for slouching
     cannot be claimed from this single-participant dataset.

Feature Compatibility & Absence of head_to_shoulder_ratio:
- 'shoulder_tilt_deg' and 'neck_angle_deg' are calculated with identical angular formulas.
- MultiPosture raw coordinates are hip-centered; PostureGuard features are image-normalized.
- 'head_to_shoulder_ratio' CANNOT be calculated for PostureGuard's historical data
  because raw nose/shoulder coordinates were not persisted in posture_data.csv.
  In strict accordance with project integrity policy, head_to_shoulder_ratio is
  NOT fabricated, synthesized, or imputed for PostureGuard samples.

Outputs:
- CSV:  data/external/multiposture/hybrid_features.csv
- JSON: data/external/multiposture/hybrid_split.json
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

from src.prepare_multiposture import (
    calculate_neck_angle_multiposture,
    calculate_shoulder_tilt_multiposture,
    MULTIPOSTURE_CAMERA_TILT_OFFSET_DEG,
    TRAIN_PARTICIPANTS as MP_TRAIN_PARTICIPANTS,
    TEST_PARTICIPANTS as MP_TEST_PARTICIPANTS,
)

# File Paths
MULTIPOSTURE_RAW_CSV = PROJECT_ROOT / "data" / "external" / "multiposture" / "data.csv"
POSTUREGUARD_CLEAN_CSV = PROJECT_ROOT / "data" / "posture_data_clean.csv"
OUTPUT_HYBRID_CSV = PROJECT_ROOT / "data" / "external" / "multiposture" / "hybrid_features.csv"
OUTPUT_SPLIT_JSON = PROJECT_ROOT / "data" / "external" / "multiposture" / "hybrid_split.json"

# Target 4 classes
HYBRID_CLASSES = ["correct", "slouching", "leaning_left", "leaning_right"]

# PostureGuard Sessions
PG_TRAIN_SESSIONS = ["session_20261021_195509", "session_20261021_210838"]
PG_TEST_SESSIONS = ["session_20261021_201127"]

HYBRID_COLUMNS = [
    "shoulder_tilt_deg",
    "neck_angle_deg",
    "calibrated_tilt_deg",
    "label",
    "source_dataset",
    "participant_id",
    "session_id",
    "split",
    "split_type",
]


def prepare_hybrid_dataset(
    mp_csv_path: Optional[Path] = None,
    pg_csv_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    output_split_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Builds the combined hybrid dataset with strict partition separation
    and verified feature definitions.
    """
    mp_path = Path(mp_csv_path) if mp_csv_path else MULTIPOSTURE_RAW_CSV
    pg_path = Path(pg_csv_path) if pg_csv_path else POSTUREGUARD_CLEAN_CSV
    out_csv = Path(output_csv_path) if output_csv_path else OUTPUT_HYBRID_CSV
    out_split = Path(output_split_path) if output_split_path else OUTPUT_SPLIT_JSON

    if not mp_path.exists():
        raise FileNotFoundError(f"MultiPosture raw CSV not found at: {mp_path}")
    if not pg_path.exists():
        raise FileNotFoundError(f"PostureGuard clean CSV not found at: {pg_path}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_split.parent.mkdir(parents=True, exist_ok=True)

    hybrid_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Process MultiPosture Samples (correct, leaning_left, leaning_right)
    # -------------------------------------------------------------
    mp_label_map = {
        "TUP": "correct",
        "TLL": "leaning_left",
        "TLR": "leaning_right",
    }
    mp_train_set = set(MP_TRAIN_PARTICIPANTS)
    mp_test_set = set(MP_TEST_PARTICIPANTS)

    # Participant leakage validation
    if mp_train_set.intersection(mp_test_set):
        raise ValueError("Participant leakage in MultiPosture split definition!")

    mp_counts = {"train": {c: 0 for c in HYBRID_CLASSES}, "test": {c: 0 for c in HYBRID_CLASSES}}

    with open(mp_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            raw_upper = r.get("upperbody_label", "").strip()
            if raw_upper not in mp_label_map:
                continue

            mapped_label = mp_label_map[raw_upper]
            subj = r.get("subject", "").strip()

            if subj in mp_train_set:
                split_name = "train"
                split_type = "unseen_participant_train"
            elif subj in mp_test_set:
                split_name = "test"
                split_type = "unseen_participant_test"
            else:
                continue

            mp_counts[split_name][mapped_label] += 1

            ls_x, ls_y = float(r["left_shoulder_x"]), float(r["left_shoulder_y"])
            rs_x, rs_y = float(r["right_shoulder_x"]), float(r["right_shoulder_y"])
            nose_x, nose_y = float(r["nose_x"]), float(r["nose_y"])

            mid_sh_x = (ls_x + rs_x) / 2.0
            mid_sh_y = (ls_y + rs_y) / 2.0

            raw_tilt = calculate_shoulder_tilt_multiposture(ls_x, ls_y, rs_x, rs_y)
            neck = calculate_neck_angle_multiposture(nose_x, nose_y, mid_sh_x, mid_sh_y)
            calibrated_tilt = round(raw_tilt - MULTIPOSTURE_CAMERA_TILT_OFFSET_DEG, 2)

            hybrid_rows.append({
                "shoulder_tilt_deg": raw_tilt,
                "neck_angle_deg": neck,
                "calibrated_tilt_deg": calibrated_tilt,
                "label": mapped_label,
                "source_dataset": "MultiPosture",
                "participant_id": f"MP_Subj_{subj}",
                "session_id": "MultiPosture_Recording",
                "split": split_name,
                "split_type": split_type,
            })

    # -------------------------------------------------------------
    # 2. Process PostureGuard Samples (slouching only)
    # -------------------------------------------------------------
    pg_train_sessions_set = set(PG_TRAIN_SESSIONS)
    pg_test_sessions_set = set(PG_TEST_SESSIONS)

    # Session leakage validation
    if pg_train_sessions_set.intersection(pg_test_sessions_set):
        raise ValueError("Session leakage in PostureGuard split definition!")

    pg_counts = {"train": {c: 0 for c in HYBRID_CLASSES}, "test": {c: 0 for c in HYBRID_CLASSES}}

    with open(pg_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            lbl = r.get("label", "").strip()
            # Strictly slouching only as instructed
            if lbl != "slouching":
                continue

            s_id = r.get("session_id", "").strip()
            if s_id in pg_train_sessions_set:
                split_name = "train"
                split_type = "unseen_session_train"
            elif s_id in pg_test_sessions_set:
                split_name = "test"
                split_type = "unseen_session_test"
            else:
                continue

            pg_counts[split_name]["slouching"] += 1

            raw_tilt = float(r["shoulder_tilt_deg"])
            neck = float(r["neck_angle_deg"])
            # PostureGuard data was recorded on a flat laptop webcam; calibrated_tilt equals raw tilt
            calibrated_tilt = raw_tilt

            hybrid_rows.append({
                "shoulder_tilt_deg": raw_tilt,
                "neck_angle_deg": neck,
                "calibrated_tilt_deg": calibrated_tilt,
                "label": "slouching",
                "source_dataset": "PostureGuard",
                "participant_id": "PG_User_1",
                "session_id": s_id,
                "split": split_name,
                "split_type": split_type,
            })

    # -------------------------------------------------------------
    # 3. Duplicate and Contradiction Verification
    # -------------------------------------------------------------
    seen_samples = set()
    duplicate_count = 0
    for r in hybrid_rows:
        sig = (r["source_dataset"], r["participant_id"], r["session_id"], r["calibrated_tilt_deg"], r["neck_angle_deg"], r["label"])
        if sig in seen_samples:
            duplicate_count += 1
        seen_samples.add(sig)

    # -------------------------------------------------------------
    # 4. Save Hybrid CSV and Metadata
    # -------------------------------------------------------------
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HYBRID_COLUMNS)
        writer.writeheader()
        for r in hybrid_rows:
            writer.writerow(r)

    total_train = sum(mp_counts["train"].values()) + pg_counts["train"]["slouching"]
    total_test = sum(mp_counts["test"].values()) + pg_counts["test"]["slouching"]

    split_payload = {
        "dataset_name": "PostureGuard Hybrid Dataset (Phase 1)",
        "created_at": "2026-10-31",
        "target_classes": HYBRID_CLASSES,
        "features_available": ["calibrated_tilt_deg", "neck_angle_deg"],
        "feature_compatibility_audit": {
            "shoulder_tilt_deg": "Identical atan2(dy, dx) formula. MultiPosture camera roll offset (+9.59 deg) calibrated.",
            "neck_angle_deg": "Identical lateral head deviation formula atan2(|dx|, -dy).",
            "head_to_shoulder_ratio": (
                "NOT AVAILABLE in PostureGuard historical data. "
                "Raw nose/shoulder coordinates were not stored in posture_data.csv. "
                "Per project integrity guidelines, head_to_shoulder_ratio was NOT fabricated or synthetic."
            ),
        },
        "evaluation_scope_documentation": {
            "unseen_participants": "Tested for correct, leaning_left, and leaning_right via MultiPosture test subjects ['10', '12', '13'].",
            "unseen_sessions": "Tested for slouching via PostureGuard Session 2 (same participant, unseen recording session).",
            "cross_participant_generalization_warning": (
                "Slouching is tested ONLY on an unseen session of the SINGLE PostureGuard participant. "
                "This evaluation DOES NOT measure slouching generalization across different individuals."
            ),
        },
        "partitions": {
            "train": {
                "total_samples": total_train,
                "multiposture_participants": sorted(list(mp_train_set), key=lambda x: int(x)),
                "multiposture_samples": sum(mp_counts["train"].values()),
                "postureguard_sessions": PG_TRAIN_SESSIONS,
                "postureguard_slouch_samples": pg_counts["train"]["slouching"],
                "class_distribution": {
                    "correct": mp_counts["train"]["correct"],
                    "slouching": pg_counts["train"]["slouching"],
                    "leaning_left": mp_counts["train"]["leaning_left"],
                    "leaning_right": mp_counts["train"]["leaning_right"],
                },
            },
            "test": {
                "total_samples": total_test,
                "multiposture_participants": sorted(list(mp_test_set), key=lambda x: int(x)),
                "multiposture_samples": sum(mp_counts["test"].values()),
                "postureguard_sessions": PG_TEST_SESSIONS,
                "postureguard_slouch_samples": pg_counts["test"]["slouching"],
                "class_distribution": {
                    "correct": mp_counts["test"]["correct"],
                    "slouching": pg_counts["test"]["slouching"],
                    "leaning_left": mp_counts["test"]["leaning_left"],
                    "leaning_right": mp_counts["test"]["leaning_right"],
                },
            },
        },
        "duplicate_sample_count": duplicate_count,
        "leakage_verification": "PASSED (disjoint participant and session sets).",
    }

    with open(out_split, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    return split_payload


if __name__ == "__main__":
    print("Preparing Hybrid Dataset...")
    res = prepare_hybrid_dataset()
    print("Hybrid Dataset preparation complete!")
    print(f"Train samples: {res['partitions']['train']['total_samples']} -> {res['partitions']['train']['class_distribution']}")
    print(f"Test samples:  {res['partitions']['test']['total_samples']} -> {res['partitions']['test']['class_distribution']}")
    print(f"Leakage status: {res['leakage_verification']}")

