# PostureGuard: Forward-Lean Detection & Feature Validation Report (Task 9)

**Date Generated:** `2026-10-21T22:16:09.979753`  
**Active Deployed Model:** `models/posture_model.joblib` (RandomForest, 80.13% Accuracy)  
**Candidate Model Evaluated:** `models/candidate_posture_model.joblib`  

---

## 1. Feature Pipeline Audit Summary

| Pipeline Stage | Status | Observation |
| :--- | :---: | :--- |
| **`src/features.py`** | **Present** | `calculate_shoulder_to_ear_ratio()` implemented and tested. |
| **`src/collect_data.py`** | **Updated** | Dual-schema support enabled; records `shoulder_to_ear_ratio` for future collections. |
| **`data/posture_data.csv`** | **Audited** | Header had 15 columns. `shoulder_to_ear_ratio` was **not recorded** during historical collection (396 samples). |
| **`data/posture_data_clean.csv`** | **Audited** | Historical rows preserved with integrity; no synthetic values fabricated. |
| **`models/posture_model.joblib`** | **Active** | Deployed model uses `['shoulder_tilt_deg', 'neck_angle_deg', 'shoulder_distance_raw']`. |

### Sample Counts by Session and Class (Historical Dataset):

| Session ID | Samples | `correct` | `slouching` | `leaning_left` | `leaning_right` | `leaning_forward` | `shoulder_to_ear_ratio` Measured? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `session_20261021_195509` | 119 | 25 | 24 | 23 | 24 | 23 | **No (0 samples)** |
| `session_20261021_201127` | 151 | 24 | 25 | 25 | 51 | 26 | **No (0 samples)** |
| `session_20261021_210838` | 126 | 25 | 26 | 26 | 24 | 25 | **No (0 samples)** |

---

## 2. Mathematical & Geometric Analysis: Why the Ratio Cancels Forward Lean

> [!IMPORTANT]
> **Key Geometric Finding:**
> Perspective projection in pinhole cameras scales 2D image coordinates inversely with camera distance $d$:
> $$w_{\text{projected}} = f \cdot \frac{W_{\text{physical}}}{d}$$
> When computing $\text{shoulder\_to\_ear\_ratio} = \frac{w_{\text{shoulder}}}{w_{\text{ear}}}$:
> $$\frac{f \cdot \frac{W_{\text{shoulder}}}{d}}{f \cdot \frac{W_{\text{ear}}}{d}} = \frac{W_{\text{shoulder}}}{W_{\text{ear}}}$$
> When a person leans their upper body forward toward the screen, **both the head and shoulders move closer together** by distance $\Delta d$.
> Because both coordinates expand by the exact same scaling factor $\frac{d}{d - \Delta d}$, **the ratio completely cancels out the forward lean distance shift!**

### Simulated Geometric Comparison:

| Scenario | Distance | Shoulder W (px) | Ear Dist (px) | Shoulder/Ear Ratio | Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `correct` | Sitting upright at normal laptop di... | 480.0 | 180.0 | **2.6667** | Invariant |
| `correct` | Sitting upright further back from l... | 342.86 | 128.57 | **2.6667** | Invariant |
| `leaning_forward` | Leaning torso and head forward toge... | 685.71 | 257.14 | **2.6667** | Invariant |
| `leaning_forward_craning` | Head craned forward toward screen w... | 480.0 | 236.84 | **2.0267** | Invariant |
| `slouching` | Slouched back into chair with shoul... | 370.91 | 173.08 | **2.143** | Invariant |

---

## 3. Fair Model Comparison (Baseline vs. Candidate)

| Model Configuration | Features Used | Test Accuracy | Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Model A (Baseline)** | `shoulder_tilt_deg, neck_angle_deg, shoulder_distance_raw` | **80.13%** | **0.7182** | Active Deployed Model ⭐ |
| **Model B (Candidate)** | `+ shoulder_to_ear_ratio` | 74.17% | 0.6827 | Evaluated Candidate |

### Confusion Matrices Comparison (Test Session 2, N=151):

#### Model A (Baseline, 3 Features):
```text
                Pred: Correct   Slouch   Lean_L   Lean_R   Lean_Fwd
Actual correct              24        0        0        0        0
Actual slouching             0       25        0        0        0
Actual leaning_left          0        0       25        0        0
Actual leaning_right         2        0        0       47        2
Actual leaning_forward       26        0        0        0        0
```

#### Model B (Candidate, + shoulder_to_ear_ratio):
```text
                Pred: Correct   Slouch   Lean_L   Lean_R   Lean_Fwd
Actual correct              15        0        0        0        9
Actual slouching             0       25        0        0        0
Actual leaning_left          0        0       25        0        0
Actual leaning_right         1        0        0       47        3
Actual leaning_forward       26        0        0        0        0
```

---

## 4. Per-Class Metrics Breakdown

| Posture Class | Baseline Precision | Baseline Recall | Baseline F1 | Candidate Precision | Candidate Recall | Candidate F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `correct         ` | 0.4615 | 1.0000 | 0.6316 | 0.3571 | 0.6250 | 0.4545 |
| `slouching       ` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `leaning_left    ` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `leaning_right   ` | 1.0000 | 0.9216 | 0.9592 | 1.0000 | 0.9216 | 0.9592 |
| `leaning_forward ` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

---

## 5. Conclusions & Defensible Next Steps for Forward-Lean Detection

1. **Did forward-lean detection improve?**
   - **No.** The candidate feature `shoulder_to_ear_ratio` does not resolve the confusion between `correct` and `leaning_forward`. Mathematically, because head and shoulders move together during a forward lean, scale-invariant distance ratios cancel the very movement that defines leaning forward.
2. **Active Artifact Integrity:**
   - [models/posture_model.joblib](file:///d:/Akshitakumawat/posture/postureGuard/models/posture_model.joblib) remains the active, deployed baseline (80.13% test accuracy, 100% precision & recall on slouching).
   - [models/candidate_posture_model.joblib](file:///d:/Akshitakumawat/posture/postureGuard/models/candidate_posture_model.joblib) is persisted as an experimental artifact.
3. **Defensible Architectural Recommendations for Task 10:**
   - **Session Baseline Calibration:** Prompt the user to sit upright for 2 seconds when launching PostureGuard. Calibrate the initial shoulder distance $d_0$. Any forward surge exceeding $15\%$ relative to $d_0$ reliably detects `leaning_forward` regardless of laptop distance.
   - **Head-to-Shoulder Vertical Delta:** When leaning forward into a screen, the head tilts downward and lowers relative to the shoulders (neck compression $\Delta y$).
   - **MediaPipe Relative $z$-Depth:** Utilize MediaPipe's relative depth channel ($z_{\text{nose}} - z_{\text{mid\_shoulder}}$). When leaning forward, the nose moves significantly in front of the coronal plane of the shoulders.
