# PostureGuard - Real-time Sitting Posture Detection & Monitoring

A robust, real-time computer vision and machine learning system for monitoring sitting posture in laptop close-up mode using MediaPipe Pose and hierarchical biomechanical evaluation.

---

## 📌 Architecture & Design Principles

PostureGuard evaluates sitting posture in real-time from standard laptop webcams without requiring hips or lower-body landmarks to be visible.

### 1. Active 4-Class Posture Taxonomy
The active classifier evaluates four distinct postures:
1. **`correct`** — Upright, ergonomically aligned sitting posture.
2. **`slouching`** — Forward and downward spinal slump (shoulders rounded forward, head drop).
3. **`leaning_left`** — Lateral spinal lean toward the user's left.
4. **`leaning_right`** — Lateral spinal lean toward the user's right.

### 2. Hierarchical Biomechanical Inference
To resolve the fundamental limitation where 2D lateral angles cannot distinguish upright sitting from slouching:
- **Lateral Plane Analysis ($|tilt| > 10^\circ$):** Evaluates lateral shoulder tilt and neck angles trained on the multi-participant MultiPosture dataset (13 subjects). Detects left and right leans with $>95\%$ confidence regardless of camera distance.
- **Sagittal Plane Analysis ($|tilt| \le 10^\circ$):** When shoulders are level, evaluates vertical head elevation using scale-invariant **`head_to_shoulder_ratio`** ($(\text{mid\_shoulder}_y - \text{nose}_y) / D_{\text{shoulders}}$):
  - $\text{ratio} \ge 0.64$ (or $\ge 90\%$ of personal upright baseline) $\implies$ **`correct`**
  - $\text{ratio} < 0.64$ (or drops $>10\%$ below baseline) $\implies$ **`slouching`**

### 3. Anti-Flicker Temporal Hysteresis & Persistence
- **5-Frame Persistence Buffer:** Transitions between postures require 5 consecutive frames (~200ms) to update, eliminating boundary noise and jitter.
- **Immediate Tracking Reset:** Instantly clears tracking when the user leaves the camera frame.

### 4. Upright Posture Calibration
- Press <kbd>c</kbd> while sitting upright to capture a 3-second personal upright baseline.
- Automatically tailors the slouch threshold to **$90\%$ of your exact head elevation**.
- If uncalibrated, a robust default threshold of $0.64$ applies automatically.

---

## 📁 Project Structure

```text
postureGuard/
├── data/
│   └── posture_data.csv              # Sole unified 4-class dataset (2,480 samples: MultiPosture + User Slouching)
├── models/
│   └── posture_model.joblib          # Active production 4-class model bundle
├── src/
│   ├── __init__.py
│   ├── camera.py                     # Webcam video capture utility
│   ├── collect_data.py               # Posture dataset collector (guided collection, 37-column schema)
│   ├── desktop_app.py                # Silent background desktop application & system tray
│   ├── features.py                   # Geometric feature engineering (shoulder tilt, neck angle, head-to-shoulder ratio)
│   ├── landmarks.py                  # MediaPipe landmark extraction & confidence validation
│   ├── pose_detection.py             # MediaPipe Pose solution wrapper
│   ├── realtime_posture.py           # Real-time webcam monitor with HUD, calibration, and hierarchical slouching detection
│   └── train_model.py                # Unified model training & evaluation pipeline
├── requirements.txt                 # Dependencies (opencv, mediapipe, scikit-learn, joblib, pystray, pillow)
└── README.md                        # Documentation
```

---

## 🚀 Quick Start

### 🖥️ Option A: Run Standalone Desktop Executable (.exe)

1. Simply launch the standalone portable executable:
   ```text
   dist/PostureGuard_Desktop.exe
   ```
   *(Self-contained; no Python, libraries, or terminal required).*

2. **Or run from Python source:**
   ```powershell
   .\venv\Scripts\python src/desktop_app.py
   ```

#### Desktop App Features:
- **100% Silent Background Operation:** Runs camera and posture analysis completely in memory; NO camera window or video feed is displayed on screen.
- **Smart Slouch Alert (>10s):** If the user remains in a slouched posture for more than 10 seconds continuously, a modern floating warning popup (`⚠️ Sit Straight!`) appears on top of all windows.
- **Auto-Dismissal after Upright Recovery (5s):** As soon as the user sits upright for 5 continuous seconds, the warning popup vanishes automatically.
- **Windows Startup Integration:** When enabled, PostureGuard runs automatically whenever your PC turns on.
- **System Tray Controls:** Right-click the PostureGuard tray icon to pause/resume monitoring, open the window, or exit.

---

### 📷 Option B: Run Diagnostic Webcam Monitor (With Video HUD)
```powershell
.\venv\Scripts\python src/realtime_posture.py
```
*Displays live camera feed with visual pose skeleton, angle telemetry, and HUD overlay.*

#### In-App Controls:
- <kbd>q</kbd> — Safely quit and close camera window.
- <kbd>c</kbd> — Trigger 3.0-second upright baseline calibration.
- <kbd>r</kbd> — Reset calibration to default uncalibrated state.

---

### ⚙️ Option C: Retrain Posture Model
```powershell
.\venv\Scripts\python src/train_model.py
```
*Trains classifiers on `data/posture_data.csv`, evaluates test metrics, and exports `models/posture_model.joblib`.*

---

## 📊 Unified Model Performance

Evaluated on held-out test partition (428 test samples: MultiPosture unseen subjects + user test frames):

| Posture Class | Precision | Recall | F1-Score | Support | Data Source |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `correct` | **0.93** | **0.93** | **0.93** | 241 | MultiPosture Unseen Participants |
| `slouching` | **0.50** | **1.00** | **0.67** | 7 | User Webcam Verified Frames |
| `leaning_left` | **0.90** | **1.00** | **0.95** | 90 | MultiPosture Unseen Participants |
| `leaning_right` | **1.00** | **0.81** | **0.90** | 90 | MultiPosture Unseen Participants |
| **Overall Model** | **Accuracy: 92.52%** | **Macro F1: 0.8634** | **Weighted F1: 0.92** | **428** | **Balanced 4-Class Test Set** |
