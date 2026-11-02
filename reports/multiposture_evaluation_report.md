# PostureGuard: MultiPosture Dataset Candidate Model Evaluation Report

**Date Generated:** `2026-10-29T14:50:08.830622`  
**Source Dataset:** MultiPosture Public Dataset ([Zenodo 14230872](https://zenodo.org/records/14230872))  
**Partitioning Strategy:** Participant-Aware Partitioning (Zero Participant Leakage)  
**Training Participants (10):** `1, 2, 3, 4, 5, 6, 7, 8, 9, 11` (2034 samples)  
**Testing Participants (3):** `10, 12, 13` (421 samples)  
**Features Evaluated:** `shoulder_tilt_deg, neck_angle_deg` (Distance-Invariant Upper Body Angles)  
**Compatible Target Classes (3):** `correct, leaning_left, leaning_right`  
**Unsupported Classes:** `slouching` *(MultiPosture lacks slouching data; no synthetic labels fabricated)*  
**Camera Tilt Offset:** `+9.59°` *(calibrated for laptop webcam alignment)*  

---

## 1. MultiPosture Classifier Performance Comparison (Unseen Participants)

| Classifier | Test Accuracy | Macro F1 | Weighted F1 | Train Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DecisionTree** | **85.04%** | **0.8116** | 0.8374 | 98.92% | Evaluated |
| **RandomForest** | **93.35%** | **0.9248** | 0.9321 | 98.92% | Evaluated ⭐ **Best Candidate** |
| **SVM (RBF)** | **87.41%** | **0.8416** | 0.8629 | 98.53% | Evaluated |
| **LogisticRegression** | **89.07%** | **0.8642** | 0.8817 | 98.38% | Evaluated |

---

## 2. Best Candidate Detailed Breakdown (RandomForest)

| Posture Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| `correct         ` | 0.8959 | 1.0000 | 0.9451 | 241 |
| `leaning_left    ` | 1.0000 | 0.9000 | 0.9474 | 90 |
| `leaning_right   ` | 1.0000 | 0.7889 | 0.8820 | 90 |

**Confusion Matrix (RandomForest):**

Rows = Actual, Columns = Predicted `['correct', 'leaning_left', 'leaning_right']`

```text
                       Pred: correct  leaning_left  leaning_right
Actual correct                  241             0              0
Actual leaning_left               9            81              0
Actual leaning_right             19             0             71
```

---

## 3. Fair Cross-Domain Model Comparison

> [!IMPORTANT]
> **Why a 4-Class Comparison is Unfair:**
> MultiPosture contains no samples for `slouching` (thoracic slump / head drop).
> Any model trained strictly on MultiPosture has 0% recall on slouching.
> A fair comparison must evaluate both models on the **3 common classes**.

### Performance on PostureGuard Clean Test Partition (`session_20261021_201127`, N=100):

| Model | Test Accuracy (3 Classes) | Macro F1 (3 Classes) | Evaluation Partition |
| :--- | :---: | :---: | :--- |
| **Candidate Model (RandomForest)** | **97.00%** | **0.9703** | PostureGuard Unseen Session 2 |
| **Active Model (DecisionTree)** | **97.00%** | **0.9703** | PostureGuard Unseen Session 2 |

---

## 4. Key Architectural Findings & Recommendations

1. **Multi-Participant Robustness:**
   - The candidate RandomForest trained across 10 participants achieves **93.35% test accuracy** on 3 completely unseen participants.
   - Cross-domain evaluation on PostureGuard's test session achieves **97.00% accuracy**, confirming angular feature invariance.
2. **Critical Dataset Gap (`slouching`):**
   - MultiPosture **cannot replace** PostureGuard's training dataset for active deployment because it lacks slouching data.
   - If MultiPosture were deployed alone, the live system would lose the ability to detect slouching, misclassifying slouching as correct or uncertain.
3. **Camera Roll Offset Requirement:**
   - MultiPosture was captured with an empirical +9.59° camera/body roll tilt offset across all participants.
   - Calibrating for this offset is strictly necessary before evaluating MultiPosture models against standard level-camera laptop webcams.
4. **Deployment Decision:**
   - **Retain `models/posture_model.joblib` as the active model.**
   - Keep `models/candidate_multiposture_model.joblib` available for multi-participant lateral lean evaluation or future hybrid training.
