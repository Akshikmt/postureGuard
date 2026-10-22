# PostureGuard: Four-Class Machine Learning Model Evaluation Report

**Date Generated:** `2026-10-22T13:42:18.327660`  
**Evaluation Strategy:** Session-Aware Test Partition (Zero Temporal Leakage)  
**Training Sessions:** `session_20261021_195509, session_20261021_210838`  
**Testing Sessions:** `session_20261021_201127`  
**Selected Features:** `shoulder_tilt_deg, neck_angle_deg` (Distance-Invariant Upper Body Angles)  
**Active Classes (4):** `correct, slouching, leaning_left, leaning_right`  
**Excluded Classes:** `leaning_forward` *(temporarily disabled until multi-camera/depth data is collected)*  

---

## 1. Overall Model Comparison Table

| Classifier | Test Accuracy | Macro F1 | Weighted F1 | Train Accuracy | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DecisionTree** | **97.60%** | **0.9777** | 0.9763 | 100.00% | Evaluated ⭐ **Selected Best** |
| **RandomForest** | **96.80%** | **0.9706** | 0.9686 | 100.00% | Evaluated |
| **SVM (RBF)** | **83.20%** | **0.7764** | 0.8174 | 82.74% | Evaluated |
| **LogisticRegression** | **84.80%** | **0.8037** | 0.8386 | 81.22% | Evaluated |

---

## 2. Per-Class Precision, Recall, and F1-Scores

### Model: DecisionTree

| Posture Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| `correct         ` | 0.8889 | 1.0000 | 0.9412 | 24 |
| `slouching       ` | 1.0000 | 1.0000 | 1.0000 | 25 |
| `leaning_left    ` | 1.0000 | 1.0000 | 1.0000 | 25 |
| `leaning_right   ` | 1.0000 | 0.9412 | 0.9697 | 51 |

**Confusion Matrix (DecisionTree):**

Rows = Actual, Columns = Predicted `['correct', 'slouching', 'leaning_left', 'leaning_right']`

```text
                Pred:   correct   slouchi   leaning   leaning
Actual correct               24         0         0         0
Actual slouching              0        25         0         0
Actual leaning_left           0         0        25         0
Actual leaning_right          3         0         0        48
```

**Top Misclassifications:**
- Actual `leaning_right` misclassified as `correct`: 3 sample(s)

### Model: RandomForest

| Posture Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| `correct         ` | 0.8571 | 1.0000 | 0.9231 | 24 |
| `slouching       ` | 1.0000 | 1.0000 | 1.0000 | 25 |
| `leaning_left    ` | 1.0000 | 1.0000 | 1.0000 | 25 |
| `leaning_right   ` | 1.0000 | 0.9216 | 0.9592 | 51 |

**Confusion Matrix (RandomForest):**

Rows = Actual, Columns = Predicted `['correct', 'slouching', 'leaning_left', 'leaning_right']`

```text
                Pred:   correct   slouchi   leaning   leaning
Actual correct               24         0         0         0
Actual slouching              0        25         0         0
Actual leaning_left           0         0        25         0
Actual leaning_right          4         0         0        47
```

**Top Misclassifications:**
- Actual `leaning_right` misclassified as `correct`: 4 sample(s)

### Model: SVM (RBF)

| Posture Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| `correct         ` | 0.6364 | 0.2917 | 0.4000 | 24 |
| `slouching       ` | 0.5952 | 1.0000 | 0.7463 | 25 |
| `leaning_left    ` | 1.0000 | 1.0000 | 1.0000 | 25 |
| `leaning_right   ` | 1.0000 | 0.9216 | 0.9592 | 51 |

**Confusion Matrix (SVM (RBF)):**

Rows = Actual, Columns = Predicted `['correct', 'slouching', 'leaning_left', 'leaning_right']`

```text
                Pred:   correct   slouchi   leaning   leaning
Actual correct                7        17         0         0
Actual slouching              0        25         0         0
Actual leaning_left           0         0        25         0
Actual leaning_right          4         0         0        47
```

**Top Misclassifications:**
- Actual `correct` misclassified as `slouching`: 17 sample(s)
- Actual `leaning_right` misclassified as `correct`: 4 sample(s)

### Model: LogisticRegression

| Posture Class | Precision | Recall | F1-Score | Test Support |
| :--- | :---: | :---: | :---: | :---: |
| `correct         ` | 0.6923 | 0.3750 | 0.4865 | 24 |
| `slouching       ` | 0.6250 | 1.0000 | 0.7692 | 25 |
| `leaning_left    ` | 1.0000 | 1.0000 | 1.0000 | 25 |
| `leaning_right   ` | 1.0000 | 0.9216 | 0.9592 | 51 |

**Confusion Matrix (LogisticRegression):**

Rows = Actual, Columns = Predicted `['correct', 'slouching', 'leaning_left', 'leaning_right']`

```text
                Pred:   correct   slouchi   leaning   leaning
Actual correct                9        15         0         0
Actual slouching              0        25         0         0
Actual leaning_left           0         0        25         0
Actual leaning_right          4         0         0        47
```

**Top Misclassifications:**
- Actual `correct` misclassified as `slouching`: 15 sample(s)
- Actual `leaning_right` misclassified as `correct`: 4 sample(s)

---

## 3. Generalization Limitations & Data Leakage Assessment

- **Session-Aware Split Integrity:** Recording sessions are strictly partitioned into training and test sets so no temporal adjacent frames leak across partitions.
- **Single-Participant Constraint:** All 3 current recording sessions originate from a single participant. While within-participant cross-session accuracy is 96.8%+, true cross-person generalization requires multi-participant data collection.
- **Forward-Lean Status:** Forward lean detection is temporarily disabled to prevent false positives under camera distance variation. Reintroducing it requires multi-participant depth and vertical-compression calibration.
