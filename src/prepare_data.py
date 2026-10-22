"""
PostureGuard Dataset Cleaner & Preparation Pipeline (Task 5)
------------------------------------------------------------
Processes raw collected samples from `data/posture_data.csv` into a validated,
clean dataset saved at `data/posture_data_clean.csv`.

Key Responsibilities:
1. Strict schema & data-type validation: ensures all 12 feature columns and
   3 metadata columns match the project specification.
2. Rejection & filtering: eliminates corrupted rows, NaNs, infinities, empty strings,
   or unexpected posture labels with explicit logged reasons.
3. Sentinel preservation: preserves intentional `-1.0` sentinels representing
   unobserved lower-body landmarks (hips/legs) in close-up laptop mode.
   Does NOT convert sentinels to 0.0 or invent synthetic samples.
4. Data leakage prevention: analyzes session metadata (`session_id`).
   - If multiple sessions exist, performs reproducible session-aware train/test partitioning.
   - If only 1 session exists, warns against naive random row splitting (which leaks near-duplicate
     temporal frames between train and test) and documents the need for additional sessions.
5. Saves clean data to `data/posture_data_clean.csv` and split metadata to `data/session_split.json`.
"""

import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Standard Schema Definitions
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
FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS

METADATA_COLUMNS = ["label", "timestamp", "session_id"]
LEGACY_ALL_COLUMNS = LEGACY_FEATURE_COLUMNS + METADATA_COLUMNS
EXTENDED_ALL_COLUMNS = EXTENDED_FEATURE_COLUMNS + METADATA_COLUMNS
ALL_COLUMNS = LEGACY_ALL_COLUMNS

EXPECTED_CLASSES = [
    "correct",
    "slouching",
    "leaning_left",
    "leaning_right",
    "leaning_forward",
]

# Primary features active during laptop close-up mode
PRIMARY_CLOSEUP_FEATURES = [
    "shoulder_distance_raw",
    "shoulder_tilt_deg",
    "neck_angle_deg",
    "shoulder_to_ear_ratio",
]

# Off-screen features holding valid -1.0 sentinels
SENTINEL_FEATURES = [
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


def clean_and_prepare_dataset(
    raw_csv_path: Path,
    clean_csv_path: Path,
    splits_json_path: Path,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Validates, filters, and prepares posture data.

    Returns:
        Summary dictionary with counts, rejection breakdown, and split details.
    """
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {raw_csv_path}")

    # Never allow overwriting the raw CSV
    if raw_csv_path.resolve() == clean_csv_path.resolve():
        raise ValueError("Target clean CSV cannot be the same path as the raw CSV!")

    clean_csv_path.parent.mkdir(parents=True, exist_ok=True)
    splits_json_path.parent.mkdir(parents=True, exist_ok=True)

    retained_rows: List[List[Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    rejection_reasons_counter: Dict[str, int] = {}

    total_read = 0

    with open(raw_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if header == EXTENDED_ALL_COLUMNS:
            active_all_columns = EXTENDED_ALL_COLUMNS
            active_feature_columns = EXTENDED_FEATURE_COLUMNS
        elif header == LEGACY_ALL_COLUMNS:
            active_all_columns = LEGACY_ALL_COLUMNS
            active_feature_columns = LEGACY_FEATURE_COLUMNS
        else:
            raise ValueError(
                f"Incompatible CSV header in {raw_csv_path}.\n"
                f"Expected Extended: {EXTENDED_ALL_COLUMNS}\n"
                f"Or Legacy: {LEGACY_ALL_COLUMNS}\n"
                f"Found:    {header}"
            )

        for line_num, row in enumerate(reader, start=2):
            total_read += 1
            if not row or all(c.strip() == "" for c in row):
                reason = "Empty row"
                rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                continue

            if len(row) != len(active_all_columns):
                reason = f"Field count mismatch ({len(row)} != {len(active_all_columns)})"
                rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                continue

            row_dict = dict(zip(active_all_columns, row))

            # 1. Label Validation
            label = row_dict.get("label", "").strip()
            if not label:
                reason = "Missing label"
                rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                continue

            if label not in EXPECTED_CLASSES:
                reason = f"Unexpected label '{label}'"
                rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                continue

            # 2. Metadata Validation
            timestamp = row_dict.get("timestamp", "").strip()
            session_id = row_dict.get("session_id", "").strip()
            if not timestamp:
                reason = "Missing timestamp"
                rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                continue
            if not session_id:
                reason = "Missing session_id"
                rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                continue

            # 3. Numeric Feature Validation
            row_features_valid = True
            cleaned_row_features: List[float] = []

            for col in active_feature_columns:
                val_str = row_dict.get(col, "").strip()
                if not val_str:
                    reason = f"Missing feature '{col}'"
                    rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                    rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                    row_features_valid = False
                    break

                try:
                    num_val = float(val_str)
                    if math.isnan(num_val) or math.isinf(num_val):
                        reason = f"Invalid numeric value ({val_str}) in '{col}'"
                        rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                        rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                        row_features_valid = False
                        break
                    cleaned_row_features.append(num_val)
                except ValueError:
                    reason = f"Non-numeric value '{val_str}' in '{col}'"
                    rejected_rows.append({"line": line_num, "reason": reason, "content": row})
                    rejection_reasons_counter[reason] = rejection_reasons_counter.get(reason, 0) + 1
                    row_features_valid = False
                    break

            if not row_features_valid:
                continue

            # Assemble clean row preserving exact schema
            clean_row = cleaned_row_features + [label, timestamp, session_id]
            retained_rows.append(clean_row)

    # Write clean dataset
    with open(clean_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(active_all_columns)
        writer.writerows(retained_rows)

    # 4. Session & Partition Analysis (Data Leakage Prevention)
    session_map: Dict[str, List[int]] = {}
    session_classes: Dict[str, Dict[str, int]] = {}

    for idx, r in enumerate(retained_rows):
        s_id = r[active_all_columns.index("session_id")]
        lbl = r[active_all_columns.index("label")]
        if s_id not in session_map:
            session_map[s_id] = []
            session_classes[s_id] = {c: 0 for c in EXPECTED_CLASSES}
        session_map[s_id].append(idx)
        session_classes[s_id][lbl] += 1

    distinct_sessions = list(session_map.keys())
    split_info: Dict[str, Any] = {
        "random_seed": random_seed,
        "total_samples": len(retained_rows),
        "total_sessions": len(distinct_sessions),
        "sessions": distinct_sessions,
        "session_sample_counts": {s: len(session_map[s]) for s in distinct_sessions},
        "session_class_breakdown": session_classes,
        "split_type": None,
        "train_indices": [],
        "test_indices": [],
        "train_sessions": [],
        "test_sessions": [],
        "leakage_warning": None,
        "ready_for_session_evaluation": False,
    }

    if len(distinct_sessions) >= 2:
        # Multi-session: Check if sessions have all classes
        rng = random.Random(random_seed)
        shuffled_sessions = list(distinct_sessions)
        rng.shuffle(shuffled_sessions)

        # Target ~80/20 session split
        n_test_sessions = max(1, int(round(len(shuffled_sessions) * 0.20)))
        test_sessions = shuffled_sessions[:n_test_sessions]
        train_sessions = shuffled_sessions[n_test_sessions:]

        # Verify test partition contains classes
        test_classes = set()
        for s in test_sessions:
            for cls, cnt in session_classes[s].items():
                if cnt > 0:
                    test_classes.add(cls)

        if len(test_classes) < len(EXPECTED_CLASSES):
            # Report limitation: sessions are unevenly populated
            split_info["split_type"] = "INSUFFICIENT_SESSIONS_FOR_BALANCED_SPLIT"
            split_info["leakage_warning"] = (
                f"Multiple sessions exist ({len(distinct_sessions)}), but assigning entire sessions "
                f"would omit one or more classes ({set(EXPECTED_CLASSES) - test_classes}) from the test set. "
                "Collect additional balanced sessions covering all 5 classes before partitioning."
            )
            split_info["ready_for_session_evaluation"] = False
        else:
            train_idx = []
            for s in train_sessions:
                train_idx.extend(session_map[s])
            test_idx = []
            for s in test_sessions:
                test_idx.extend(session_map[s])

            split_info["split_type"] = "SESSION_AWARE_SPLIT"
            split_info["train_sessions"] = train_sessions
            split_info["test_sessions"] = test_sessions
            split_info["train_indices"] = train_idx
            split_info["test_indices"] = test_idx
            split_info["ready_for_session_evaluation"] = True
            split_info["leakage_warning"] = (
                "Session-aware split generated successfully. No frames from the same recording session "
                "appear in both training and test partitions, preventing temporal data leakage."
            )
    else:
        # Single session limitation
        split_info["split_type"] = "SINGLE_SESSION_ONLY"
        split_info["train_sessions"] = distinct_sessions
        split_info["test_sessions"] = []
        split_info["ready_for_session_evaluation"] = False
        split_info["leakage_warning"] = (
            "CRITICAL: Exactly 1 recording session exists in the dataset. "
            "A naive random row split causes severe temporal data leakage because adjacent frames "
            "(captured every 200ms) would be shared between train and test sets, artificially inflating accuracy. "
            "To achieve valid, leak-free model evaluation in Task 6, record at least 1 or 2 additional independent "
            "sessions using `src/collect_data.py` (e.g., at slightly different laptop angles or in different lighting)."
        )

    # Save split metadata to JSON
    with open(splits_json_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    return {
        "raw_csv_path": str(raw_csv_path),
        "clean_csv_path": str(clean_csv_path),
        "splits_json_path": str(splits_json_path),
        "total_rows_read": total_read,
        "rows_retained": len(retained_rows),
        "rows_rejected": len(rejected_rows),
        "rejection_reasons": rejection_reasons_counter,
        "split_info": split_info,
        "primary_closeup_features": PRIMARY_CLOSEUP_FEATURES,
        "sentinel_features": SENTINEL_FEATURES,
    }


def print_preparation_summary(summary: Dict[str, Any]) -> None:
    """Prints a clear summary of dataset cleaning and preparation."""
    print("\n" + "=" * 74)
    print(" " * 20 + "POSTUREGUARD DATASET PREPARATION SUMMARY")
    print("=" * 74)
    print(f"Source Raw CSV      : {summary['raw_csv_path']}")
    print(f"Output Clean CSV    : {summary['clean_csv_path']}")
    print(f"Splits Metadata     : {summary['splits_json_path']}")
    print("-" * 74)
    print(f"Total Rows Read     : {summary['total_rows_read']}")
    print(f"Rows Retained       : {summary['rows_retained']} "
          f"({(summary['rows_retained']/max(1, summary['total_rows_read'])*100):.1f}%)")
    print(f"Rows Rejected       : {summary['rows_rejected']}")

    if summary["rows_rejected"] > 0:
        print("\nRejection Breakdown:")
        for reason, count in summary["rejection_reasons"].items():
            print(f"  - {reason}: {count} row(s)")
    else:
        print("Rejection Breakdown : 0 rows rejected (100% data integrity!)")

    print("-" * 74)
    print("FEATURE CLASSIFICATION FOR CLOSE-UP LAPTOP MODEL:")
    print("1. Primary Active Features (Carry rich discriminatory variance):")
    for f in summary["primary_closeup_features"]:
        print(f"   * {f}")
    print("2. Sentinel Features (-1.0 Preserved for off-screen lower body/hips):")
    for f in summary["sentinel_features"]:
        print(f"   * {f} (-1.0 sentinel preserved)")

    print("-" * 74)
    split_info = summary["split_info"]
    print("DATA LEAKAGE & SESSION PARTITIONING:")
    print(f"Distinct Sessions   : {split_info['total_sessions']}")
    print(f"Split Strategy      : {split_info['split_type']}")
    print(f"\n[LEAKAGE NOTE]:\n  {split_info['leakage_warning']}")
    print("=" * 74 + "\n")


def main():
    raw_csv = PROJECT_ROOT / "data" / "posture_data.csv"
    clean_csv = PROJECT_ROOT / "data" / "posture_data_clean.csv"
    splits_json = PROJECT_ROOT / "data" / "session_split.json"

    if len(sys.argv) > 1:
        raw_csv = Path(sys.argv[1])
    if len(sys.argv) > 2:
        clean_csv = Path(sys.argv[2])
    if len(sys.argv) > 3:
        splits_json = Path(sys.argv[3])

    try:
        summary = clean_and_prepare_dataset(raw_csv, clean_csv, splits_json)
        print_preparation_summary(summary)
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Preparation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
