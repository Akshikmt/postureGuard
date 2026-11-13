"""
PostureGuard Unified Model Training & Evaluation Pipeline
----------------------------------------------------------
Trains and benchmarks classifiers on the unified 4-class posture dataset
(MultiPosture multi-participant dataset + user webcam slouching data).

Active 4-Class Taxonomy:
1. 'correct'        - Upright, ergonomically aligned sitting posture.
2. 'slouching'      - Forward/downward spinal slump.
3. 'leaning_left'   - Lateral spinal lean toward the user's left.
4. 'leaning_right'  - Lateral spinal lean toward the user's right.

Features:
- 'shoulder_tilt_deg' (or 'calibrated_tilt_deg')
- 'neck_angle_deg'

Outputs:
- Active Model Bundle: models/posture_model.joblib
- Evaluation Report (MD): reports/model_evaluation_report.md
- Evaluation Report (JSON): reports/model_evaluation_report.json
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project Constants
DATASET_CSV = PROJECT_ROOT / "data" / "posture_data.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "posture_model.joblib"
REPORT_MD_PATH = PROJECT_ROOT / "reports" / "model_evaluation_report.md"
REPORT_JSON_PATH = PROJECT_ROOT / "reports" / "model_evaluation_report.json"

FOUR_CLASS_LABELS = ["correct", "slouching", "leaning_left", "leaning_right"]
CLASS_LABELS = FOUR_CLASS_LABELS
PRIMARY_FEATURES = ["shoulder_tilt_deg", "neck_angle_deg"]


def load_dataset_and_split(
    csv_path: Optional[Path] = None,
    splits_json_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads features and labels partitioned into train and test splits from posture_data.csv.

    Returns:
        X_train: (N_train, 2) numpy array of angular features
        y_train: (N_train,) numpy array of string labels
        X_test:  (N_test, 2) numpy array of angular features
        y_test:  (N_test,) numpy array of string labels
        split_meta: Dictionary summarizing split counts and class distribution
    """
    target_csv = Path(csv_path) if csv_path else DATASET_CSV
    if not target_csv.exists():
        raise FileNotFoundError(f"Posture dataset CSV not found: {target_csv}")

    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    with open(target_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            split_val = r.get("split", "train").lower().strip()
            if split_val == "train":
                train_rows.append(r)
            elif split_val == "test":
                test_rows.append(r)

    def _extract_feature(r: Dict[str, Any], feature_name: str) -> float:
        val = r.get(feature_name)
        if val is None or val == "":
            if feature_name == "shoulder_tilt_deg":
                val = r.get("calibrated_tilt_deg", 0.0)
            elif feature_name == "calibrated_tilt_deg":
                val = r.get("shoulder_tilt_deg", 0.0)
            else:
                val = 0.0
        return float(val)

    X_train = np.array([
        [_extract_feature(r, "calibrated_tilt_deg"), _extract_feature(r, "neck_angle_deg")]
        for r in train_rows
    ], dtype=np.float64)
    y_train = np.array([r["label"] for r in train_rows])

    X_test = np.array([
        [_extract_feature(r, "calibrated_tilt_deg"), _extract_feature(r, "neck_angle_deg")]
        for r in test_rows
    ], dtype=np.float64)
    y_test = np.array([r["label"] for r in test_rows])

    split_meta = {
        "dataset_path": str(target_csv),
        "total_samples": len(train_rows) + len(test_rows),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_class_counts": {c: int(np.sum(y_train == c)) for c in FOUR_CLASS_LABELS},
        "test_class_counts": {c: int(np.sum(y_test == c)) for c in FOUR_CLASS_LABELS},
    }

    return X_train, y_train, X_test, y_test, split_meta


def build_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Builds the suite of candidate posture classification models."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight="balanced"
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6,
            random_state=random_state,
            class_weight="balanced"
        ),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=random_state, class_weight="balanced"))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"))
        ]),
    }


def evaluate_classifier(
    name: str,
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fits and evaluates a classifier, computing comprehensive metrics and confusion matrix."""
    labels = class_labels or FOUR_CLASS_LABELS
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    slouch_metrics = report.get("slouching", {"recall": 0.0, "precision": 0.0, "f1-score": 0.0})

    return {
        "model_name": name,
        "test_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "slouch_recall": round(float(slouch_metrics.get("recall", 0.0)), 4),
        "slouch_precision": round(float(slouch_metrics.get("precision", 0.0)), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def predict_posture(
    model_bundle: Dict[str, Any],
    features: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    """
    Inference helper: predicts posture class and probabilities from raw feature values.

    Validates that required features are present and non-sentinel.
    """
    required_features = model_bundle.get("feature_names", PRIMARY_FEATURES)
    clf = model_bundle["model"]
    classes = model_bundle.get("classes", FOUR_CLASS_LABELS)

    vec = []
    for feat in required_features:
        if feat not in features:
            if feat == "shoulder_tilt_deg" and "calibrated_tilt_deg" in features:
                val = features["calibrated_tilt_deg"]
            elif feat == "calibrated_tilt_deg" and "shoulder_tilt_deg" in features:
                val = features["shoulder_tilt_deg"]
            else:
                raise ValueError(f"Missing required feature: '{feat}'")
        else:
            val = features[feat]

        if val is None or math.isnan(val) or math.isinf(val) or val == -1.0:
            raise ValueError(f"Invalid or sentinel value for feature '{feat}': {val}")
        vec.append(float(val))

    X_in = np.array([vec], dtype=np.float64)
    pred_label = clf.predict(X_in)[0]

    if hasattr(clf, "predict_proba"):
        probs_raw = clf.predict_proba(X_in)[0]
        clf_classes = list(clf.classes_)
        probs = {c: float(probs_raw[clf_classes.index(c)]) if c in clf_classes else 0.0 for c in classes}
    else:
        probs = {c: 1.0 if c == pred_label else 0.0 for c in classes}

    return pred_label, probs


def train_and_evaluate_all(
    csv_path: Optional[Path] = None,
    model_output_path: Optional[Path] = None,
    report_md_path: Optional[Path] = None,
    report_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Full execution pipeline for training and saving the production posture model."""
    target_csv = Path(csv_path) if csv_path else DATASET_CSV
    out_model = Path(model_output_path) if model_output_path else MODEL_OUTPUT_PATH
    out_md = Path(report_md_path) if report_md_path else REPORT_MD_PATH
    out_json = Path(report_json_path) if report_json_path else REPORT_JSON_PATH

    print("=" * 72)
    print("PostureGuard Unified Model Training & Evaluation")
    print("=" * 72)

    X_train, y_train, X_test, y_test, meta = load_dataset_and_split(target_csv)
    print(f"Loaded dataset: {target_csv.name} ({meta['total_samples']} samples)")
    print(f"  Training samples : {len(X_train)} {meta['train_class_counts']}")
    print(f"  Test samples     : {len(X_test)} {meta['test_class_counts']}")

    models = build_candidate_models(random_state=42)
    results: Dict[str, Any] = {}
    best_name = "RandomForest"
    best_model = None

    print("\nEvaluating Candidate Classifiers...")
    for name, clf in models.items():
        res = evaluate_classifier(name, clf, X_train, y_train, X_test, y_test, FOUR_CLASS_LABELS)
        results[name] = res
        print(f"  {name:22}: Acc={res['test_accuracy']*100:5.2f}% | Macro F1={res['macro_f1']:0.4f} | Slouch Rec={res['slouch_recall']*100:5.1f}%")

        if name == best_name:
            best_model = clf

    # Export Production Model Bundle
    out_model.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": PRIMARY_FEATURES,
        "classes": FOUR_CLASS_LABELS,
        "train_accuracy": round(float(accuracy_score(y_train, best_model.predict(X_train))), 4),
        "test_accuracy": results[best_name]["test_accuracy"],
        "macro_f1": results[best_name]["macro_f1"],
        "training_date": datetime.now().isoformat(),
        "dataset_source": "data/posture_data.csv (MultiPosture + User Slouching)",
        "forward_lean_status": "disabled",
    }
    joblib.dump(bundle, out_model)
    print(f"\nSaved Production Model Bundle: {out_model}")

    # Export Evaluation Reports
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rf_res = results[best_name]
    rf_cm = np.array(rf_res["confusion_matrix"])

    out_model_posix = str(out_model).replace("\\", "/")
    md_content = f"""# PostureGuard: 4-Class Posture Model Evaluation Report

**Date Generated:** `{datetime.now().isoformat()}`  
**Dataset:** `data/posture_data.csv` (Total: {meta['total_samples']} samples)  
**Features:** `{', '.join(PRIMARY_FEATURES)}`  
**Classes:** `{', '.join(FOUR_CLASS_LABELS)}`  
**Model Artifact:** [`models/{out_model.name}`](file:///{out_model_posix})  

---

## 1. Executive Summary

This model is trained on the unified PostureGuard dataset (`data/posture_data.csv`), combining:
1. **Multi-Participant MultiPosture Benchmark:** 13 subjects, providing multi-person diversity for `correct`, `leaning_left`, and `leaning_right`.
2. **User Webcam Recording:** High-fidelity verified `slouching` samples recorded on the user's laptop camera.

### Overall Performance Overview (Test Partition, N={len(y_test)}):
- **Overall Test Accuracy:** **{rf_res['test_accuracy']*100:.2f}%**
- **Macro F1-Score:** **{rf_res['macro_f1']:.4f}**
- **Weighted F1-Score:** **{rf_res['weighted_f1']:.4f}**
- **Slouching Recall:** **{rf_res['slouch_recall']*100:.1f}%**

---

## 2. Model Comparison Table

| Classifier | Test Acc | Macro F1 | Weighted F1 | Slouch Recall | Slouch Prec | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **RandomForest** | **{results['RandomForest']['test_accuracy']*100:.2f}%** | **{results['RandomForest']['macro_f1']:.4f}** | **{results['RandomForest']['weighted_f1']:.4f}** | **{results['RandomForest']['slouch_recall']*100:.1f}%** | {results['RandomForest']['slouch_precision']*100:.1f}% | ⭐ Selected Production |
| **DecisionTree** | {results['DecisionTree']['test_accuracy']*100:.2f}% | {results['DecisionTree']['macro_f1']:.4f} | {results['DecisionTree']['weighted_f1']:.4f} | {results['DecisionTree']['slouch_recall']*100:.1f}% | {results['DecisionTree']['slouch_precision']*100:.1f}% | Evaluated |
| **SVM (RBF)** | {results['SVM (RBF)']['test_accuracy']*100:.2f}% | {results['SVM (RBF)']['macro_f1']:.4f} | {results['SVM (RBF)']['weighted_f1']:.4f} | {results['SVM (RBF)']['slouch_recall']*100:.1f}% | {results['SVM (RBF)']['slouch_precision']*100:.1f}% | Evaluated |
| **LogisticRegression** | {results['LogisticRegression']['test_accuracy']*100:.2f}% | {results['LogisticRegression']['macro_f1']:.4f} | {results['LogisticRegression']['weighted_f1']:.4f} | {results['LogisticRegression']['slouch_recall']*100:.1f}% | {results['LogisticRegression']['slouch_precision']*100:.1f}% | Evaluated |

---

## 3. Best Model Per-Class Metrics (RandomForest)

| Posture Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `correct` | {rf_res['classification_report']['correct']['precision']:.4f} | {rf_res['classification_report']['correct']['recall']:.4f} | {rf_res['classification_report']['correct']['f1-score']:.4f} | {meta['test_class_counts']['correct']} |
| `slouching` | {rf_res['classification_report']['slouching']['precision']:.4f} | {rf_res['classification_report']['slouching']['recall']:.4f} | {rf_res['classification_report']['slouching']['f1-score']:.4f} | {meta['test_class_counts']['slouching']} |
| `leaning_left` | {rf_res['classification_report']['leaning_left']['precision']:.4f} | {rf_res['classification_report']['leaning_left']['recall']:.4f} | {rf_res['classification_report']['leaning_left']['f1-score']:.4f} | {meta['test_class_counts']['leaning_left']} |
| `leaning_right` | {rf_res['classification_report']['leaning_right']['precision']:.4f} | {rf_res['classification_report']['leaning_right']['recall']:.4f} | {rf_res['classification_report']['leaning_right']['f1-score']:.4f} | {meta['test_class_counts']['leaning_right']} |

### Confusion Matrix (Test Partition):

```text
                     Pred:  correct   slouching  leaning_left  leaning_right
Actual correct                 {rf_cm[0,0]:4d}        {rf_cm[0,1]:4d}          {rf_cm[0,2]:4d}           {rf_cm[0,3]:4d}
Actual slouching                {rf_cm[1,0]:4d}        {rf_cm[1,1]:4d}          {rf_cm[1,2]:4d}           {rf_cm[1,3]:4d}
Actual leaning_left             {rf_cm[2,0]:4d}        {rf_cm[2,1]:4d}          {rf_cm[2,2]:4d}           {rf_cm[2,3]:4d}
Actual leaning_right            {rf_cm[3,0]:4d}        {rf_cm[3,1]:4d}          {rf_cm[3,2]:4d}           {rf_cm[3,3]:4d}
```
"""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Saved Evaluation Report MD   : {out_md}")
    print(f"Saved Evaluation Report JSON : {out_json}")

    return results


if __name__ == "__main__":
    train_and_evaluate_all()
