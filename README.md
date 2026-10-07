# PostureGuard - Real-time Sitting Posture Detection & Classification

A Python-based computer vision project for real-time sitting posture detection and classification using webcam feed and machine learning.

---

## 📌 Project Overview (Future Scope)

The ultimate goal of this project is to monitor sitting postures in real time to encourage ergonomic habits. The complete system will:
1. Capture video feed from a webcam.
2. Extract body landmarks (e.g., shoulders, ears, neck, spine).
3. Compute posture-related angles and spatial metrics.
4. Classify postures into categories such as:
   - **Correct sitting**
   - **Slouching**
   - **Leaning left**
   - **Leaning right**
   - **Leaning forward**
5. Provide visual alerts or feedback to help users correct their posture.

---

## 🎯 Current Implementation (Stage 1: Initial Setup & Camera Test)

This first stage sets up the core project structure, virtual environment, and basic webcam acquisition pipeline:
- Minimal directory structure established.
- Isolated Python virtual environment (`venv`).
- Initial OpenCV dependency configured.
- Clean and safe camera test program in `src/camera.py` verifying webcam access and frame display.

> **Note:** Machine learning models, landmark extractors (e.g., MediaPipe), datasets, and UI components will be built in subsequent stages.

---

## 📁 Project Structure

```text
postureGuard/
├── data/              # Raw and processed datasets (future stages)
├── models/            # Saved trained model artifacts (future stages)
├── notebooks/         # Exploratory data analysis & model prototyping (future stages)
├── src/
│   └── camera.py      # Basic camera test & capture utility
├── tests/             # Unit and integration tests
├── venv/              # Python virtual environment
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+** (Detected: Python 3.11)
- Working webcam

### 2. Activate the Virtual Environment

Open a terminal (PowerShell or Command Prompt) and navigate to the project directory:

```powershell
cd postureGuard
```

Activate the virtual environment:

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

- **Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

With the virtual environment activated, install the initial dependency:

```bash
pip install -r requirements.txt
```

### 4. Run the Webcam Test

Test camera connectivity and verify the video stream:

```bash
python src/camera.py
```

- A window titled **"Posture Detection - Camera Feed"** should appear displaying your live video feed.
- Press <kbd>q</kbd> while focusing on the video window to safely exit and release the webcam.
