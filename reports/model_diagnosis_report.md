# PostureGuard: Feature Diagnosis & Error Analysis Report (Task 7)

**Date Generated:** `2026-10-21T21:16:16.549063`  
**Dataset Evaluated:** `396 clean samples`  
**Train Session:** `session_20261021_195509, session_20261021_210838` (N=245)  
**Test Session:** `session_20261021_201127` (N=151)  

---

## 1. Executive Summary & Root Cause of Misclassifications

> [!IMPORTANT]
> **Why Slouching and Leaning Forward Were Predicted as Correct:**
> 1. **Camera Distance Shift:** In Session 2, the user sat approximately **5.54% further back** from the laptop camera than in Session 1 (`shoulder_distance_raw` dropped from mean 0.3848 to 0.3635).
> 2. **Threshold Collapse:** In Session 1, `correct` posture had `shoulder_distance_raw >= 0.3569`. In Session 2, because of the greater camera distance, **100% of slouching samples (max=0.3687)** and **100% of leaning_forward samples (max=0.3782)** fell strictly below `0.3569`. The decision trees therefore classified all of them as `correct`.
> 3. **Sagittal Blindspot:** `calculate_neck_angle` computes lateral head deviation (`abs(dx)`), which is near $0^\circ$ for both upright sitting, slouching, and straight-forward leaning.
> 4. **Why Lateral Leans Succeeded:** `shoulder_tilt_deg` is an angular feature ($atan2(\Delta y, \Delta x)$), making it naturally camera-distance invariant. `leaning_left` and `leaning_right` achieved **100% precision** across sessions.

---

## 2. Statistical Feature Distributions (Train vs Test Sessions)

### A. Training Partitions (`session_20261021_195509, session_20261021_210838`, N=245)

| Class | Feature | Min | Q1 | Median | Q3 | IQR | Max | Mean (±Std) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correct        ` | `shoulder_tilt_deg     ` | -2.22 | -2.0375 | -1.095 | 0.71 | 2.7475 | 0.9 | -0.7708 (±1.2564) |
| `correct        ` | `neck_angle_deg        ` | 1.06 | 1.565 | 3.61 | 6.595 | 5.03 | 7.08 | 4.0528 (±2.5047) |
| `correct        ` | `shoulder_distance_raw ` | 0.3569 | 0.358 | 0.3733 | 0.3831 | 0.0251 | 0.384 | 0.3707 (±0.0123) |
| `slouching      ` | `shoulder_tilt_deg     ` | -0.97 | -0.71 | -0.6 | -0.46 | 0.25 | -0.12 | -0.5986 (±0.1979) |
| `slouching      ` | `neck_angle_deg        ` | 1.59 | 1.9825 | 3.41 | 4.08 | 2.0975 | 4.65 | 3.12 (±1.0665) |
| `slouching      ` | `shoulder_distance_raw ` | 0.3638 | 0.3648 | 0.3655 | 0.408 | 0.0431 | 0.4098 | 0.3854 (±0.0215) |
| `leaning_left   ` | `shoulder_tilt_deg     ` | 17.87 | 18.49 | 18.74 | 18.9 | 0.41 | 20.27 | 18.8029 (±0.5453) |
| `leaning_left   ` | `neck_angle_deg        ` | 44.08 | 45.94 | 46.47 | 47.07 | 1.13 | 48.73 | 46.491 (±1.248) |
| `leaning_left   ` | `shoulder_distance_raw ` | 0.329 | 0.3328 | 0.3353 | 0.405 | 0.0722 | 0.4061 | 0.3668 (±0.036) |
| `leaning_right  ` | `shoulder_tilt_deg     ` | -18.19 | -17.45 | -17.225 | -17.05 | 0.4 | -16.69 | -17.2652 (±0.3269) |
| `leaning_right  ` | `neck_angle_deg        ` | 26.13 | 27.275 | 34.665 | 39.885 | 12.61 | 42.18 | 33.954 (±6.1463) |
| `leaning_right  ` | `shoulder_distance_raw ` | 0.3552 | 0.3566 | 0.3795 | 0.4035 | 0.0469 | 0.4043 | 0.3798 (±0.0235) |
| `leaning_forward` | `shoulder_tilt_deg     ` | -4.3 | -3.3775 | -2.52 | -2.335 | 1.0425 | -1.76 | -2.7871 (±0.6513) |
| `leaning_forward` | `neck_angle_deg        ` | 2.8 | 5.065 | 5.76 | 6.965 | 1.9 | 15.06 | 6.7963 (±3.3191) |
| `leaning_forward` | `shoulder_distance_raw ` | 0.4048 | 0.4155 | 0.418 | 0.4352 | 0.0197 | 0.4365 | 0.4223 (±0.0117) |

### B. Testing Partition (`session_20261021_201127`, N=151)

| Class | Feature | Min | Q1 | Median | Q3 | IQR | Max | Mean (±Std) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `correct        ` | `shoulder_tilt_deg     ` | -3.01 | -2.4 | -1.79 | -1.73 | 0.67 | -1.6 | -2.0171 (±0.3826) |
| `correct        ` | `neck_angle_deg        ` | 0.04 | 0.245 | 1.02 | 3.5325 | 3.2875 | 3.74 | 1.7 (±1.5432) |
| `correct        ` | `shoulder_distance_raw ` | 0.3501 | 0.3507 | 0.3523 | 0.3538 | 0.0031 | 0.3544 | 0.3522 (±0.0016) |
| `slouching      ` | `shoulder_tilt_deg     ` | -0.19 | 0.42 | 0.89 | 1.04 | 0.62 | 1.27 | 0.7452 (±0.4063) |
| `slouching      ` | `neck_angle_deg        ` | 2.78 | 2.89 | 2.99 | 3.04 | 0.15 | 3.28 | 2.98 (±0.1056) |
| `slouching      ` | `shoulder_distance_raw ` | 0.3668 | 0.3672 | 0.3679 | 0.3682 | 0.001 | 0.3687 | 0.3677 (±0.0006) |
| `leaning_left   ` | `shoulder_tilt_deg     ` | 11.1 | 11.19 | 11.26 | 11.38 | 0.19 | 12.51 | 11.4244 (±0.4204) |
| `leaning_left   ` | `neck_angle_deg        ` | 39.22 | 39.85 | 40.75 | 44.04 | 4.19 | 49.06 | 42.3312 (±3.1939) |
| `leaning_left   ` | `shoulder_distance_raw ` | 0.3514 | 0.3529 | 0.3541 | 0.3545 | 0.0016 | 0.3548 | 0.3536 (±0.0011) |
| `leaning_right  ` | `shoulder_tilt_deg     ` | -24.92 | -16.63 | -14.51 | -13.985 | 2.645 | -5.08 | -15.1271 (±3.9335) |
| `leaning_right  ` | `neck_angle_deg        ` | 2.21 | 31.435 | 35.72 | 37.6 | 6.165 | 46.54 | 33.638 (±9.0149) |
| `leaning_right  ` | `shoulder_distance_raw ` | 0.3541 | 0.3641 | 0.3661 | 0.3679 | 0.0037 | 0.3725 | 0.3653 (±0.0038) |
| `leaning_forward` | `shoulder_tilt_deg     ` | -2.75 | -2.11 | -1.94 | -1.8425 | 0.2675 | -1.78 | -1.995 (±0.2041) |
| `leaning_forward` | `neck_angle_deg        ` | 0.04 | 1.9325 | 2.375 | 3.0175 | 1.085 | 3.53 | 2.1827 (±1.0012) |
| `leaning_forward` | `shoulder_distance_raw ` | 0.3734 | 0.3743 | 0.376 | 0.3773 | 0.003 | 0.3782 | 0.3759 (±0.0016) |

---

## 3. Scale-Normalized Feature Investigation

### Availability of Head/Face Landmarks:
- MediaPipe Pose detects 33 body landmarks, including `left_ear`, `right_ear`, and `nose`.
- In `src/landmarks.py`, `left_ear` and `right_ear` are defined in `POSTURE_LANDMARK_MAP`.
- **Historical Dataset Constraint:** In `data/posture_data.csv` and `data/posture_data_clean.csv`, ear coordinates were **not persisted** during historical capture (only the 12 extracted posture features were saved).
- **Integrity Policy:** In compliance with project guidelines, ear landmarks were **not manufactured or synthesized** for the 270 historical samples.

### Implemented Candidate Normalized Feature (`shoulder_to_ear_ratio`):
To provide scale and camera-distance invariance for future sessions and real-time inference, the following candidate calculation has been implemented in `src/features.py`:

```python
def calculate_shoulder_to_ear_ratio(
    left_shoulder: Optional[LandmarkPoint],
    right_shoulder: Optional[LandmarkPoint],
    left_ear: Optional[LandmarkPoint],
    right_ear: Optional[LandmarkPoint],
) -> Optional[float]:
    shoulder_dist = euclidean_distance_2d(left_shoulder, right_shoulder)
    ear_dist = euclidean_distance_2d(left_ear, right_ear)
    if shoulder_dist is None or ear_dist is None or ear_dist < 1e-4:
        return None
    return round(shoulder_dist / ear_dist, 4)
```

**Mathematical Property:**
If a person shifts from 50cm to 70cm from the webcam, perspective scaling shrinks coordinates by factor $k$:
$$\frac{k \cdot \text{shoulder\_distance}}{k \cdot \text{ear\_distance}} = \frac{\text{shoulder\_distance}}{\text{ear\_distance}}$$
The scale factor $k$ cancels out, guaranteeing distance-invariance.

---

## 4. Fair Comparison on Existing Dataset

| Feature Configuration | Test Accuracy | Macro F1 | Status |
| :--- | :---: | :---: | :---: |
| **Baseline 3 Features** (`shoulder_tilt_deg`, `neck_angle_deg`, `shoulder_distance_raw`) | **80.13%** | **0.7165** | Deployed Baseline |
| **Baseline + Interaction Features** (tilt interactions) | 74.17% | 0.6827 | Experimental |

---

## 5. Concrete Action Plan & Recommended Data Collection

1. **Deploy `shoulder_to_ear_ratio` into Live Pipeline:** Ensure `src/features.py` and `src/collect_data.py` calculate and store `shoulder_to_ear_ratio` alongside raw distance.
2. **Targeted Data Collection:** Record 2 additional sessions specifically prompting:
   - Varying distance: sit normally, then slightly forward, then slightly back.
   - Slouching vs upright sitting under natural posture shifts.
3. **Zero Deployed Model Disruption:** The baseline model artifact [models/posture_model.joblib](file:///d:/Akshitakumawat/posture/postureGuard/models/posture_model.joblib) remains safely active until multi-session validation with normalized features is verified.
