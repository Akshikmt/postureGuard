# PostureGuard: Hybrid Dataset Candidate Model Evaluation Report (Phase 1)

**Date Generated:** `2026-10-31T18:27:56.084800`  
**Task:** Phase 1 Hybrid Dataset Benchmarking (MultiPosture + PostureGuard Slouching)  
**Features Evaluated:** `calibrated_tilt_deg, neck_angle_deg` (Distance-Invariant Upper-Body Angles)  
**Target Classes (4):** `correct, slouching, leaning_left, leaning_right`  
**Candidate Artifact:** [`models/candidate_hybrid_model.joblib`](file:///d:/Akshitakumawat/posture/postureGuard/models/candidate_hybrid_model.joblib)  

---

## 1. Feature Compatibility & Head Ratio Audit

> [!IMPORTANT]
> **Why `head_to_shoulder_ratio` Was NOT Included:**
> 1. **MultiPosture:** Contains complete 3D skeletal keypoints for all 33 joints, allowing `head_to_shoulder_ratio` calculation.
> 2. **PostureGuard (`posture_data.csv`):** Stores only 12 extracted scalar features. **Raw joint coordinates were never persisted** during historical recording.
> 3. **Non-Derivability:** `head_to_shoulder_ratio` ($(	ext{mid\_shoulder}_y - 	ext{nose}_y) / D_{	ext{shoulders}}$) cannot be derived from `neck_angle_deg` ($atan2(|dx|, -dy)$) because the horizontal head displacement $|dx|$ is unknown.
> 4. **Integrity Guarantee:** In accordance with project requirements, no synthetic, fabricated, or imputed ratio values were introduced.

---

## 2. Experimental Model Benchmark (Combined 4-Class Test Set, N=446)

### A. Models with Balanced Class Weights (`class_weight='balanced'`):

| Classifier | Test Accuracy | Macro F1 | Weighted F1 | Slouch Recall | Slouch Precision | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DecisionTree** | **87.67%** | **0.7139** | 0.8597 | 8.0% | 33.3% | Evaluated |
| **RandomForest** | **89.69%** | **0.7312** | 0.8797 | 8.0% | 33.3% | Evaluated ⭐ **Selected Candidate** |
| **SVM (RBF)** | **73.32%** | **0.7285** | 0.7805 | 100.0% | 19.7% | Evaluated |
| **LogisticRegression** | **72.65%** | **0.7291** | 0.7775 | 100.0% | 18.2% | Evaluated |

### B. Models with Standard Weights (Unweighted Baseline):

| Classifier | Test Accuracy | Macro F1 | Weighted F1 | Slouch Recall | Slouch Precision | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DecisionTree** | 79.15% | 0.5903 | 0.7597 | 0.0% | 0.0% | Evaluated |
| **RandomForest** | 88.34% | 0.6845 | 0.8581 | 0.0% | 0.0% | Evaluated |
| **SVM (RBF)** | 82.51% | 0.6211 | 0.7928 | 0.0% | 0.0% | Evaluated |
| **LogisticRegression** | 84.08% | 0.6379 | 0.8100 | 0.0% | 0.0% | Evaluated |

---

## 3. Best Candidate Model Performance Breakdown (RandomForest Balanced)

| Posture Class | Precision | Recall | F1-Score | Test Support | Test Partition Type |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `correct         ` | 0.8545 | 0.9751 | 0.9109 |  241 | MultiPosture Unseen Participants |
| `slouching       ` | 0.3333 | 0.0800 | 0.1290 |   25 | PostureGuard Unseen Session (Same User) |
| `leaning_left    ` | 0.9783 | 1.0000 | 0.9890 |   90 | MultiPosture Unseen Participants |
| `leaning_right   ` | 1.0000 | 0.8111 | 0.8957 |   90 | MultiPosture Unseen Participants |

**Confusion Matrix (RandomForest Balanced):**

```text
                     Pred:  correct   slouching  leaning_left  leaning_right
Actual correct                  235           4             2              0
Actual slouching                 23           2             0              0
Actual leaning_left               0           0            90              0
Actual leaning_right             17           0             0             73
```

---

## 4. Separate Sub-Dataset Evaluation Results

### A. MultiPosture Unseen Participants Evaluation (Subjects 10, 12, 13; N=421)
- **Evaluation Type:** Strictly unseen participants ($N=3$).
- **Subset Accuracy:** **94.54%**
- **Subset Macro F1:** **0.7095**
- **Per-Class Metrics:**
  - `correct       `: Precision = 0.9325, Recall = 0.9751, F1 = 0.9533 (Support = 241)
  - `leaning_left  `: Precision = 0.9783, Recall = 1.0000, F1 = 0.9890 (Support = 90)
  - `leaning_right `: Precision = 1.0000, Recall = 0.8111, F1 = 0.8957 (Support = 90)

### B. PostureGuard Slouching Unseen Session Evaluation (Session 2; N=25)
- **Evaluation Type:** Unseen session of the **SAME participant**.
- **WARNING:** Does **NOT** establish cross-participant generalization.
- **Slouching Recall:** **8.0%** (2 of 25 detected)
- **Predictions Breakdown:** `{'slouching': 2, 'correct': 23}`

---

## 5. Critical Findings & Honest Slouching Analysis

> [!WARNING]
> **The Angular Feature Dilemma (Why Slouching Fails in 2D Space):**
> 1. **High Overall Accuracy is Misleading:** The RandomForest achieves **89.46% overall accuracy**, but its slouching recall is only **8.0%** (2 out of 25 slouching frames). 92% of slouching test frames are falsely classified as `correct`.
> 2. **Why Tree Models Default to `correct`:** PostureGuard slouching samples have tilt $\approx 0^\circ$ and neck $\approx 3^\circ$. MultiPosture upright sitting also has tilt $\approx 0^\circ$ and neck $\approx 13^\circ$ (std $10.9^\circ$). Because slouching sits right in the center of upright sitting, tree classifiers predict `correct`.
> 3. **The Balanced Weights Failure (SVM & Logistic Regression):**
>    When forced by balanced weights to predict slouching, linear and RBF models create a decision boundary around $(0^\circ, 3^\circ)$. While slouching recall rises to 100%, **41% to 45% of upright samples are falsely flagged as slouching!** (99 out of 241 upright frames in SVM; 108 out of 241 in LogisticRegression).
> 4. **Generalization Barrier:** Slouching cannot be generalized across participants using only lateral angles.

---

## 6. Recommendations & Model Status

1. **Preserve Active Model (`models/posture_model.joblib`):** Retain as the active deployed model.
2. **Preserve MultiPosture Candidate (`models/candidate_multiposture_model.joblib`):** Highly accurate (93.35%) 3-class model for lateral leans and upright sitting across diverse participants.
3. **Hybrid Candidate Status (`models/candidate_hybrid_model.joblib`):** Successfully trained and packaged as a candidate artifact, but **NOT recommended for deployment** because angular features alone cannot reliably detect slouching across multi-participant distributions.
