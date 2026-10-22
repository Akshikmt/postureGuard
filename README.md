# PostureGuard - Real-time Sitting Posture Detection & Monitoring

A robust, real-time computer vision and machine learning system for monitoring sitting posture in laptop close-up mode.

---

## 📌 Architecture & Design Principles

PostureGuard evaluates sitting posture in real-time from standard laptop webcams without requiring hips or lower-body landmarks to be visible.

### 1. Active 4-Class Posture Taxonomy
To guarantee cross-distance stability and eliminate false forward-lean classifications, the active classifier evaluates four distinct postures:
1. **`correct`** — Upright, ergonomically aligned sitting posture.
2. **`slouching`** — Forward and downward slump (shoulders rounded forward, head drop).
3. **`leaning_left`** — Lateral spinal lean toward the user's left.
4. **`leaning_right`** — Lateral spinal lean toward the user's right.

> **Status of Forward-Lean Detection:** `leaning_forward` detection is temporarily disabled from active classification. Forward-lean detection relies heavily on perspective surge and camera-distance baselines; it will be reintroduced when multi-participant training data with varied camera distances is collected. Raw dataset samples for forward leaning are fully preserved in `data/posture_data.csv`.

### 2. Distance-Invariant Angular Feature Space
The model evaluates distance-invariant angular features rather than raw pixel distances:
- **`shoulder_tilt_deg`** — Lateral angle of the shoulder line relative to horizontal ($atan2(\Delta y, \Delta x)$). Invariant to camera distance.
- **`neck_angle_deg`** — Angle of the head relative to the shoulder midpoint ($atan2(|\Delta x|, -\Delta y)$).
- **`head_to_shoulder_ratio`** — Scale-normalized vertical head height over shoulder width ($(\text{mid\_shoulder}_y - \text{nose}_y) / D_{\text{shoulders}}$).

### 3. Anti-Flicker Temporal Hysteresis & Persistence
To prevent rapid flip-flopping between `correct` and `slouching`:
- **Confidence Threshold:** Requires top probability $\ge 0.55$ and margin over runner-up $\ge 0.10$.
- **Hysteresis Hold:** Boundary noise near decision thresholds ($51\%$ vs $49\%$) does not change the active state.
- **5-Frame Persistence Requirement:** A new posture must persist for at least 5 consecutive frames (~160–200ms at 25–30 FPS) before the display updates.
- **Explicit Uncertain State:** If ambiguous predictions persist for $>15$ frames (~0.5s), the HUD displays `UNCERTAIN` rather than guessing.
- **Zero-Latency Reset:** Immediately clears tracking when the user leaves the frame.

### 4. Upright Posture Calibration & Validation
Personal calibration captures a user-specific upright baseline to detect camera position shifts:
- **Baseline Rejection Rules:** Rejects invalid calibration attempts instead of silently accepting slouched or leaning baselines:
  - $|\text{mean}(\text{tilt})| > 5.0^\circ$ $\implies$ Rejected (uneven shoulders).
  - $\text{mean}(\text{neck}) > 12.0^\circ$ $\implies$ Rejected (neck tilted sideways).
  - $\text{mean}(\text{head\_ratio}) < 0.35$ $\implies$ Rejected (slouched head drop).
  - $\text{std}(D) > 0.030$ $\implies$ Rejected (excessive movement during baseline capture).
- **Camera Shift Alerts:** If the user moves closer or further by $\ge 25\%$ relative to calibrated baseline $D_0$, the HUD displays an alert recommending recalibration (`[c]`).
- **Non-Interference:** Calibration validates stability but does not override ML model predictions.

---

## 📁 Project Structure

```text
postureGuard/
├── data/
│   ├── posture_data.csv            # Raw labeled feature dataset (all 5 classes preserved)
│   ├── posture_data_clean.csv      # Cleaned dataset (396 samples, sessions 1-3)
│   └── session_split.json          # Session-aware partitioning metadata
├── models/
│   ├── posture_model.joblib        # Active 4-class Decision Tree model (97.60% test accuracy)
│   └── candidate_posture_model.joblib # Candidate model artifact evaluated before promotion
├── reports/
│   ├── model_evaluation_report.json # Comprehensive performance metrics & confusion matrix
│   └── model_evaluation_report.md   # Markdown summary of model evaluation
├── src/
│   ├── camera.py                   # Camera feed test utility
│   ├── landmarks.py                # MediaPipe landmark extraction & confidence filtering
│   ├── features.py                 # Geometric angles, scale ratios, and normalization math
│   ├── realtime_posture.py         # Real-time webcam HUD with hysteresis & calibration
│   ├── collect_data.py             # Hands-free dataset collector
│   ├── validate_data.py            # Dataset quality & leakage analyzer
│   ├── prepare_data.py             # Data preparation & session splitter
│   └── train_model.py              # 4-class training & model evaluation pipeline
├── tests/
│   ├── test_features.py            # Unit tests for landmark & feature math
│   ├── test_collect_data.py        # Unit tests for dataset collector
│   ├── test_data_pipeline.py       # Unit tests for data preparation
│   ├── test_model_training.py      # Unit tests for non-leakage training & evaluation
│   └── test_realtime_posture.py    # Unit tests for 4-class inference, hysteresis, calibration
├── requirements.txt                # Dependencies (opencv, mediapipe, scikit-learn, joblib)
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.11)
- Working built-in or external USB webcam

### 2. Environment Setup
```powershell
# Navigate to project root
cd d:\Akshitakumawat\posture\postureGuard

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Real-Time Posture Detection (Webcam)
```powershell
.\venv\Scripts\python.exe src/realtime_posture.py
```

#### Controls:
- <kbd>q</kbd> — Safely quit and release the webcam.
- <kbd>c</kbd> — Trigger 3.0-second upright baseline calibration.
- <kbd>r</kbd> — Reset calibration to default uncalibrated state.

#### Calibration Procedure:
1. Sit comfortably upright facing the webcam at normal working distance.
2. Press <kbd>c</kbd>.
3. Hold still for 3 seconds while the calibration bar fills.
4. If your shoulders are level and posture is upright, the baseline will be established.
5. If you are slouched or leaning during calibration, the system will reject the baseline and guide you to retry.

---

## 🎥 Supported Camera Setup & Operating Conditions

- **Position:** Laptop webcam or external monitor webcam mounted at eye to upper-chest level.
- **Working Distance:** Approximately 45 cm to 75 cm (arms-length viewing distance).
- **Viewing Angle:** Frontal facing ($\pm 15^\circ$ azimuth and elevation).
- **Required Landmarks:** Head (nose/eyes) and shoulders must be clearly visible in frame.
- **Lighting:** Adequate frontal illumination; avoid intense backlighting that silhouettes facial features.

---

## 📊 Model Evaluation (4-Class Session-Aware Split)

Evaluated strictly on held-out Session 2 ($N=125$ test samples, zero leakage from training Sessions 1 & 3):

| Classifier | Train Accuracy | Test Accuracy | Macro F1 | Weighted F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Selected)** | **98.48%** | **97.60%** | **0.9777** | **0.9760** | **Active Deployed** |
| Random Forest | 99.49% | 96.80% | 0.9706 | 0.9680 | Strong candidate |
| Logistic Regression | 94.42% | 94.40% | 0.9472 | 0.9439 | Underfits slouching |
| SVM (RBF) | 98.98% | 88.00% | 0.8876 | 0.8797 | Degraded on session split |

### Decision Tree Confusion Matrix (Held-out Session 2):
```text
                  Predicted:
               correct  slouching  lean_left  lean_right
Actual:
correct           25        0          0          0      (100.0% Recall)
slouching          0       25          0          0      (100.0% Recall)
leaning_left       0        0         23          1      ( 95.8% Recall)
leaning_right      2        0          0         49      ( 96.1% Recall)
```

---

## 👥 Multi-Participant Data Collection Guide

To advance the system toward full cross-person generalization and re-introduce forward-lean detection:

1. **Diverse Participants:**
   - Collect data from at least 5–10 participants with varying body builds, shoulder widths, and heights.
   - Record participant identifier (`participant_id`) and recording conditions (`camera_height`, `user_distance_cm`).

2. **Distance & Angle Variations:**
   - Record at 3 realistic distances: Close (~45 cm), Mid (~60 cm), Far (~80 cm).
   - Record at 2 camera heights: Laptop table level and raised laptop stand level.

3. **Participant-Wise Train/Test Split:**
   - Partition participants so that test set individuals **never appear** in training.
   - Report cross-participant accuracy and F1 score without data leakage.

4. **Running the Guided Collector:**
   ```powershell
   .\venv\Scripts\python.exe src/collect_data.py
   ```
   Follow on-screen audio-visual prompts through each posture routine.

---

## 🧪 Testing

Run the complete automated unit test suite:
```powershell
.\venv\Scripts\python.exe -m unittest discover tests
```
- **Test Suite Status:** 55 tests passing (100% pass rate).
- **Coverage:** Landmarks math, feature normalization, 4-class training, non-leakage partitioning, confidence hysteresis, and calibration rejection.
