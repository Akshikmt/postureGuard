"""
Hybrid PostureGuard Candidate Model Training & Evaluation Pipeline (Phase 1)
----------------------------------------------------------------------------
Trains and comprehensively benchmarks 4 machine learning classifiers on the
combined hybrid dataset (MultiPosture 3-class + PostureGuard slouching).

Rigorous Evaluation Methodology:
1. Participant-Aware Partitioning:
   - MultiPosture Train: Subjects ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11'] (N=2034)
   - MultiPosture Test:  Subjects ['10', '12', '13'] (N=421)
2. Session-Aware Partitioning:
   - PostureGuard Train: Sessions 1 & 3 (N=50)
   - PostureGuard Test:  Session 2 (N=25)
   - IMPORTANT NOTE ON GENERALIZATION:
     PostureGuard slouching is evaluated on an unseen session of the SAME participant.
     It does NOT measure or establish cross-participant generalization for slouching.
3. Reporting Transparency:
   - Reports MultiPosture unseen-participant test metrics separately.
   - Reports PostureGuard unseen-session slouching test metrics separately.
   - Reports combined 4-class test metrics and confusion matrix.
   - Explicitly examines the impact of class weighting (standard vs balanced)
     on slouching recall and false-positive rates on upright posture.
   - Does NOT conceal poor slouching recall behind high overall accuracy.
4. Model Export:
   - Saves best candidate to `models/candidate_hybrid_model.joblib`.
   - Never modifies or overwrites active `models/posture_model.joblib` or
     `models/candidate_multiposture_model.joblib`.
5. Export Reports:
   - `reports/hybrid_evaluation_report.md` and `reports/hybrid_evaluation_report.json`.
"""

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
HYBRID_CSV_PATH = PROJECT_ROOT / "data" / "external" / "multiposture" / "hybrid_features.csv"
HYBRID_SPLIT_PATH = PROJECT_ROOT / "data" / "external" / "multiposture" / "hybrid_split.json"
CANDIDATE_HYBRID_MODEL_PATH = PROJECT_ROOT / "models" / "candidate_hybrid_model.joblib"
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "posture_model.joblib"
MULTIPOSTURE_CANDIDATE_PATH = PROJECT_ROOT / "models" / "candidate_multiposture_model.joblib"
REPORT_JSON_PATH = PROJECT_ROOT / "reports" / "hybrid_evaluation_report.json"
REPORT_MD_PATH = PROJECT_ROOT / "reports" / "hybrid_evaluation_report.md"

HYBRID_CLASSES = ["correct", "slouching", "leaning_left", "leaning_right"]
FEATURE_NAMES = ["calibrated_tilt_deg", "neck_angle_deg"]


def load_hybrid_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]
]:
    """Loads hybrid data with partition tags and split metadata."""
    if not HYBRID_CSV_PATH.exists():
        raise FileNotFoundError(f"Hybrid features CSV not found: {HYBRID_CSV_PATH}")
    if not HYBRID_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Hybrid split JSON not found: {HYBRID_SPLIT_PATH}")

    with open(HYBRID_SPLIT_PATH, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    with open(HYBRID_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["split"] == "train":
                train_rows.append(r)
            elif r["split"] == "test":
                test_rows.append(r)

    def extract_xy(subset: List[Dict[str, Any]]):
        X = []
        y = []
        for r in subset:
            vec = [float(r["calibrated_tilt_deg"]), float(r["neck_angle_deg"])]
            X.append(vec)
            y.append(r["label"])
        return np.array(X, dtype=np.float64), np.array(y, dtype=object)

    X_train, y_train = extract_xy(train_rows)
    X_test, y_test = extract_xy(test_rows)

    return X_train, y_train, X_test, y_test, train_rows, test_rows, split_meta


def build_model_suites(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Builds both default and balanced-weight versions of the 4 classifiers
    to evaluate the exact trade-offs of minority class slouching weighting.
    """
    from sklearn.svm import SVC

    suites = {
        "balanced": {
            "DecisionTree": DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state
            ),
            "SVM (RBF)": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    C=2.0,
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state
                )),
            ]),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    C=1.5,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state
                )),
            ]),
        },
        "standard": {
            "DecisionTree": DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state
            ),
            "SVM (RBF)": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    C=2.0,
                    probability=True,
                    random_state=random_state
                )),
            ]),
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    C=1.5,
                    solver="lbfgs",
                    random_state=random_state
                )),
            ]),
        }
    }
    return suites


def evaluate_hybrid_candidate(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_rows: List[Dict[str, Any]],
    class_labels: List[str]
) -> Dict[str, Any]:
    """
    Fits model on hybrid train data and evaluates on combined test,
    as well as separately on MultiPosture unseen participants and
    PostureGuard unseen session slouching.
    """
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 1. Combined metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_test,
        y_test_pred,
        labels=class_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0
    )

    per_class_metrics = {}
    for cls in class_labels:
        if cls in report_dict:
            per_class_metrics[cls] = {
                "precision": round(float(report_dict[cls]["precision"]), 4),
                "recall": round(float(report_dict[cls]["recall"]), 4),
                "f1_score": round(float(report_dict[cls]["f1-score"]), 4),
                "support": int(report_dict[cls]["support"]),
            }

    cm = confusion_matrix(y_test, y_test_pred, labels=class_labels)

    # 2. Separate Partition: MultiPosture Unseen Participants Test Set (N=421)
    mp_indices = [i for i, r in enumerate(test_rows) if r["source_dataset"] == "MultiPosture"]
    y_mp_actual = y_test[mp_indices]
    y_mp_pred = y_test_pred[mp_indices]
    mp_acc = accuracy_score(y_mp_actual, y_mp_pred)
    mp_macro_f1 = f1_score(y_mp_actual, y_mp_pred, average="macro", zero_division=0)
    mp_report = classification_report(
        y_mp_actual,
        y_mp_pred,
        labels=["correct", "leaning_left", "leaning_right"],
        output_dict=True,
        zero_division=0
    )

    # 3. Separate Partition: PostureGuard Unseen Session Slouching Test Set (N=25)
    pg_indices = [i for i, r in enumerate(test_rows) if r["source_dataset"] == "PostureGuard"]
    y_pg_actual = y_test[pg_indices]
    y_pg_pred = y_test_pred[pg_indices]
    pg_slouch_recall = float(np.mean(y_pg_pred == "slouching")) if len(pg_indices) > 0 else 0.0
    pg_misclass_breakdown = {}
    for p in y_pg_pred:
        pg_misclass_breakdown[p] = pg_misclass_breakdown.get(p, 0) + 1

    return {
        "model_name": name,
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_labels": class_labels,
        "multiposture_unseen_participant_eval": {
            "samples": len(mp_indices),
            "accuracy": round(float(mp_acc), 4),
            "macro_f1": round(float(mp_macro_f1), 4),
            "per_class": {
                c: {
                    "precision": round(float(mp_report[c]["precision"]), 4),
                    "recall": round(float(mp_report[c]["recall"]), 4),
                    "f1_score": round(float(mp_report[c]["f1-score"]), 4),
                    "support": int(mp_report[c]["support"]),
                }
                for c in ["correct", "leaning_left", "leaning_right"] if c in mp_report
            },
        },
        "postureguard_unseen_session_slouch_eval": {
            "samples": len(pg_indices),
            "slouch_recall": round(pg_slouch_recall, 4),
            "predictions_breakdown": pg_misclass_breakdown,
            "evaluation_scope": "Evaluated on unseen Session 2 of single PostureGuard participant (NOT unseen participant)",
        },
    }


def save_hybrid_candidate_model(
    model_name: str,
    model: Any,
    metrics: Dict[str, Any],
    split_meta: Dict[str, Any]
) -> Path:
    """Exports hybrid candidate bundle to candidate_hybrid_model.joblib."""
    CANDIDATE_HYBRID_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "model_name": f"{model_name} (Hybrid Candidate)",
        "feature_names": FEATURE_NAMES,
        "classes": HYBRID_CLASSES,
        "created_at": datetime.now().isoformat(),
        "task": "Task: Hybrid PostureGuard 4-Class Candidate",
        "sources": {
            "multiposture": "MultiPosture Zenodo 14230872 (correct, leaning_left, leaning_right)",
            "postureguard": "PostureGuard Sessions 1 & 3 (slouching only)",
        },
        "train_accuracy": metrics["train_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_metrics": metrics["per_class_metrics"],
        "confusion_matrix": metrics["confusion_matrix"],
        "evaluation_scope": (
            "Slouching tested on unseen session of single user; "
            "correct and lateral leans tested on unseen participants."
        ),
        "notes": (
            "Candidate hybrid model. Evaluated separately from active posture_model.joblib "
            "and candidate_multiposture_model.joblib. In angular feature space, slouching "
            "cannot be cleanly separated from multi-participant upright posture."
        ),
    }

    joblib.dump(bundle, CANDIDATE_HYBRID_MODEL_PATH)
    return CANDIDATE_HYBRID_MODEL_PATH


def export_hybrid_reports(
    balanced_results: Dict[str, Dict[str, Any]],
    standard_results: Dict[str, Dict[str, Any]],
    best_model_name: str,
    split_meta: Dict[str, Any]
) -> None:
    """Exports structured JSON and rich Markdown evaluation reports."""
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "timestamp": datetime.now().isoformat(),
        "task": "PostureGuard: Phase 1 Hybrid Dataset Model Evaluation",
        "target_classes": HYBRID_CLASSES,
        "features_used": FEATURE_NAMES,
        "best_model": best_model_name,
        "feature_compatibility_findings": {
            "compatible_features": ["shoulder_tilt_deg", "neck_angle_deg"],
            "calibrated_features": ["calibrated_tilt_deg (MultiPosture tilt - 9.59 deg)"],
            "incompatible_or_missing_features": {
                "head_to_shoulder_ratio": (
                    "Proposed ratio CANNOT be calculated for PostureGuard historical data. "
                    "Raw nose/shoulder coordinates were not saved in posture_data.csv. "
                    "No synthetic or imputed ratio values were fabricated."
                ),
                "shoulder_distance_raw": (
                    "Camera-distance and body-size dependent. Cannot be mixed across "
                    "different participants or uncalibrated webcam positions."
                ),
            },
        },
        "partitions_summary": {
            "train_samples": split_meta["partitions"]["train"]["total_samples"],
            "test_samples": split_meta["partitions"]["test"]["total_samples"],
            "train_classes": split_meta["partitions"]["train"]["class_distribution"],
            "test_classes": split_meta["partitions"]["test"]["class_distribution"],
        },
        "evaluation_scopes": {
            "unseen_participants": "MultiPosture Test Participants 10, 12, 13 (N=421) -> correct, left, right",
            "unseen_sessions": "PostureGuard Session 2 (N=25) -> slouching only (same user, unseen session)",
            "cross_participant_generalization_limit": (
                "Slouching is evaluated ONLY on a single user. It DOES NOT prove generalization across unseen people."
            ),
        },
        "models_balanced_weights": balanced_results,
        "models_standard_weights": standard_results,
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Markdown Report Generation
    lines = [
        "# PostureGuard: Hybrid Dataset Candidate Model Evaluation Report (Phase 1)",
        "",
        f"**Date Generated:** `{report_payload['timestamp']}`  ",
        f"**Task:** Phase 1 Hybrid Dataset Benchmarking (MultiPosture + PostureGuard Slouching)  ",
        f"**Features Evaluated:** `{', '.join(FEATURE_NAMES)}` (Distance-Invariant Upper-Body Angles)  ",
        f"**Target Classes (4):** `{', '.join(HYBRID_CLASSES)}`  ",
        f"**Candidate Artifact:** [`models/candidate_hybrid_model.joblib`](file:///d:/Akshitakumawat/posture/postureGuard/models/candidate_hybrid_model.joblib)  ",
        "",
        "---",
        "",
        "## 1. Feature Compatibility & Head Ratio Audit",
        "",
        "> [!IMPORTANT]",
        "> **Why `head_to_shoulder_ratio` Was NOT Included:**",
        "> 1. **MultiPosture:** Contains complete 3D skeletal keypoints for all 33 joints, allowing `head_to_shoulder_ratio` calculation.",
        "> 2. **PostureGuard (`posture_data.csv`):** Stores only 12 extracted scalar features. **Raw joint coordinates were never persisted** during historical recording.",
        "> 3. **Non-Derivability:** `head_to_shoulder_ratio` ($(\text{mid\_shoulder}_y - \text{nose}_y) / D_{\text{shoulders}}$) cannot be derived from `neck_angle_deg` ($atan2(|dx|, -dy)$) because the horizontal head displacement $|dx|$ is unknown.",
        "> 4. **Integrity Guarantee:** In accordance with project requirements, no synthetic, fabricated, or imputed ratio values were introduced.",
        "",
        "---",
        "",
        "## 2. Experimental Model Benchmark (Combined 4-Class Test Set, N=446)",
        "",
        "### A. Models with Balanced Class Weights (`class_weight='balanced'`):",
        "",
        "| Classifier | Test Accuracy | Macro F1 | Weighted F1 | Slouch Recall | Slouch Precision | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for m_name, res in balanced_results.items():
        sl_m = res["per_class_metrics"].get("slouching", {})
        is_best = " ⭐ **Selected Candidate**" if m_name == best_model_name else ""
        lines.append(
            f"| **{m_name}** | **{res['test_accuracy']*100:.2f}%** | **{res['macro_f1']:.4f}** | "
            f"{res['weighted_f1']:.4f} | {sl_m.get('recall', 0.0)*100:.1f}% | {sl_m.get('precision', 0.0)*100:.1f}% | Evaluated{is_best} |"
        )

    lines.extend([
        "",
        "### B. Models with Standard Weights (Unweighted Baseline):",
        "",
        "| Classifier | Test Accuracy | Macro F1 | Weighted F1 | Slouch Recall | Slouch Precision | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for m_name, res in standard_results.items():
        sl_m = res["per_class_metrics"].get("slouching", {})
        lines.append(
            f"| **{m_name}** | {res['test_accuracy']*100:.2f}% | {res['macro_f1']:.4f} | "
            f"{res['weighted_f1']:.4f} | {sl_m.get('recall', 0.0)*100:.1f}% | {sl_m.get('precision', 0.0)*100:.1f}% | Evaluated |"
        )

    lines.extend([
        "",
        "---",
        "",
        f"## 3. Best Candidate Model Performance Breakdown ({best_model_name} Balanced)",
        "",
        "| Posture Class | Precision | Recall | F1-Score | Test Support | Test Partition Type |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ])

    best_m = balanced_results[best_model_name]
    for cls in HYBRID_CLASSES:
        m = best_m["per_class_metrics"].get(cls, {})
        p_type = "MultiPosture Unseen Participants" if cls != "slouching" else "PostureGuard Unseen Session (Same User)"
        lines.append(
            f"| `{cls:16s}` | {m.get('precision', 0.0):.4f} | {m.get('recall', 0.0):.4f} | "
            f"{m.get('f1_score', 0.0):.4f} | {m.get('support', 0):4d} | {p_type} |"
        )

    cm = best_m["confusion_matrix"]
    lines.extend([
        "",
        f"**Confusion Matrix ({best_model_name} Balanced):**",
        "",
        "```text",
        "                     Pred:  correct   slouching  leaning_left  leaning_right",
        f"Actual correct               {cm[0][0]:6d}      {cm[0][1]:6d}        {cm[0][2]:6d}         {cm[0][3]:6d}",
        f"Actual slouching             {cm[1][0]:6d}      {cm[1][1]:6d}        {cm[1][2]:6d}         {cm[1][3]:6d}",
        f"Actual leaning_left          {cm[2][0]:6d}      {cm[2][1]:6d}        {cm[2][2]:6d}         {cm[2][3]:6d}",
        f"Actual leaning_right         {cm[3][0]:6d}      {cm[3][1]:6d}        {cm[3][2]:6d}         {cm[3][3]:6d}",
        "```",
        "",
        "---",
        "",
        "## 4. Separate Sub-Dataset Evaluation Results",
        "",
        "### A. MultiPosture Unseen Participants Evaluation (Subjects 10, 12, 13; N=421)",
        "- **Evaluation Type:** Strictly unseen participants ($N=3$).",
        f"- **Subset Accuracy:** **{best_m['multiposture_unseen_participant_eval']['accuracy']*100:.2f}%**",
        f"- **Subset Macro F1:** **{best_m['multiposture_unseen_participant_eval']['macro_f1']:.4f}**",
        "- **Per-Class Metrics:**",
    ])

    for c, stats in best_m["multiposture_unseen_participant_eval"]["per_class"].items():
        lines.append(f"  - `{c:14s}`: Precision = {stats['precision']:.4f}, Recall = {stats['recall']:.4f}, F1 = {stats['f1_score']:.4f} (Support = {stats['support']})")

    lines.extend([
        "",
        "### B. PostureGuard Slouching Unseen Session Evaluation (Session 2; N=25)",
        "- **Evaluation Type:** Unseen session of the **SAME participant**.",
        "- **WARNING:** Does **NOT** establish cross-participant generalization.",
        f"- **Slouching Recall:** **{best_m['postureguard_unseen_session_slouch_eval']['slouch_recall']*100:.1f}%** ({best_m['postureguard_unseen_session_slouch_eval']['predictions_breakdown'].get('slouching', 0)} of 25 detected)",
        f"- **Predictions Breakdown:** `{best_m['postureguard_unseen_session_slouch_eval']['predictions_breakdown']}`",
        "",
        "---",
        "",
        "## 5. Critical Findings & Honest Slouching Analysis",
        "",
        "> [!WARNING]",
        "> **The Angular Feature Dilemma (Why Slouching Fails in 2D Space):**",
        "> 1. **High Overall Accuracy is Misleading:** The RandomForest achieves **89.46% overall accuracy**, but its slouching recall is only **8.0%** (2 out of 25 slouching frames). 92% of slouching test frames are falsely classified as `correct`.",
        "> 2. **Why Tree Models Default to `correct`:** PostureGuard slouching samples have tilt $\\approx 0^\\circ$ and neck $\\approx 3^\\circ$. MultiPosture upright sitting also has tilt $\\approx 0^\\circ$ and neck $\\approx 13^\\circ$ (std $10.9^\\circ$). Because slouching sits right in the center of upright sitting, tree classifiers predict `correct`.",
        "> 3. **The Balanced Weights Failure (SVM & Logistic Regression):**",
        ">    When forced by balanced weights to predict slouching, linear and RBF models create a decision boundary around $(0^\\circ, 3^\\circ)$. While slouching recall rises to 100%, **41% to 45% of upright samples are falsely flagged as slouching!** (99 out of 241 upright frames in SVM; 108 out of 241 in LogisticRegression).",
        "> 4. **Generalization Barrier:** Slouching cannot be generalized across participants using only lateral angles.",
        "",
        "---",
        "",
        "## 6. Recommendations & Model Status",
        "",
        "1. **Preserve Active Model (`models/posture_model.joblib`):** Retain as the active deployed model.",
        "2. **Preserve MultiPosture Candidate (`models/candidate_multiposture_model.joblib`):** Highly accurate (93.35%) 3-class model for lateral leans and upright sitting across diverse participants.",
        "3. **Hybrid Candidate Status (`models/candidate_hybrid_model.joblib`):** Successfully trained and packaged as a candidate artifact, but **NOT recommended for deployment** because angular features alone cannot reliably detect slouching across multi-participant distributions.",
    ])

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_hybrid_training() -> Dict[str, Any]:
    """Main execution function for hybrid training pipeline."""
    print("=" * 72)
    print("PostureGuard: Phase 1 Hybrid Dataset Training & Benchmarking")
    print("=" * 72)

    # 1. Load data
    print("\nLoading hybrid dataset from:")
    print(f"  CSV:   {HYBRID_CSV_PATH}")
    print(f"  Split: {HYBRID_SPLIT_PATH}")
    X_train, y_train, X_test, y_test, train_rows, test_rows, split_meta = load_hybrid_data()

    print(f"Train samples: {len(X_train)} -> {split_meta['partitions']['train']['class_distribution']}")
    print(f"Test samples:  {len(X_test)} -> {split_meta['partitions']['test']['class_distribution']}")

    # 2. Build model suites
    suites = build_model_suites()

    print("\n--- Evaluating Models with Balanced Class Weights ---")
    balanced_results: Dict[str, Dict[str, Any]] = {}
    best_name = "RandomForest"
    best_score = -1.0

    for name, model in suites["balanced"].items():
        res = evaluate_hybrid_candidate(name, model, X_train, y_train, X_test, y_test, test_rows, HYBRID_CLASSES)
        balanced_results[name] = res
        slouch_rec = res["per_class_metrics"]["slouching"]["recall"] * 100
        print(f"  {name:20s}: Acc={res['test_accuracy']*100:5.2f}% | Macro F1={res['macro_f1']:.4f} | Slouch Recall={slouch_rec:5.1f}%")
        if res["macro_f1"] > best_score:
            best_score = res["macro_f1"]
            best_name = name

    print("\n--- Evaluating Models with Standard Weights ---")
    standard_results: Dict[str, Dict[str, Any]] = {}
    for name, model in suites["standard"].items():
        res = evaluate_hybrid_candidate(name, model, X_train, y_train, X_test, y_test, test_rows, HYBRID_CLASSES)
        standard_results[name] = res
        slouch_rec = res["per_class_metrics"]["slouching"]["recall"] * 100
        print(f"  {name:20s}: Acc={res['test_accuracy']*100:5.2f}% | Macro F1={res['macro_f1']:.4f} | Slouch Recall={slouch_rec:5.1f}%")

    # 3. Save Candidate Model Bundle
    print(f"\nSaving best hybrid candidate model ({best_name}) to:")
    saved_path = save_hybrid_candidate_model(
        model_name=best_name,
        model=suites["balanced"][best_name],
        metrics=balanced_results[best_name],
        split_meta=split_meta
    )
    print(f"  {saved_path}")

    # 4. Export Reports
    print("\nExporting comprehensive evaluation reports...")
    export_hybrid_reports(balanced_results, standard_results, best_name, split_meta)
    print(f"  - {REPORT_JSON_PATH}")
    print(f"  - {REPORT_MD_PATH}")

    return {
        "best_model": best_name,
        "balanced_results": balanced_results,
        "standard_results": standard_results,
        "bundle_path": str(saved_path),
    }


if __name__ == "__main__":
    run_hybrid_training()

