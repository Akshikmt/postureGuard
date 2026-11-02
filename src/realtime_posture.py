"""
PostureGuard: Real-Time Sitting Posture Detection & Monitoring
--------------------------------------------------------------
Captures live webcam feed, extracts MediaPipe pose landmarks in laptop
close-up mode, executes robust 4-class posture classification using
distance-invariant angular features, prevents prediction flip-flopping
via confidence-based hysteresis and persistence, and validates upright
calibration baselines against slouched or tilted postures.

Active 4-Class Posture Targets:
1. 'correct'        - Balanced upright sitting posture.
2. 'slouching'      - Forward/downward spinal slump (shoulders forward, head drop).
3. 'leaning_left'   - Lateral spinal lean toward the user's left.
4. 'leaning_right'  - Lateral spinal lean toward the user's right.

NOTE: 'leaning_forward' detection is temporarily disabled from active classification
until multi-participant depth and distance training data is collected.

Architectural Principles:
1. Distance Invariance: Active 4-class model uses angular features (shoulder_tilt_deg,
   neck_angle_deg) that are mathematically invariant to camera distance.
2. Strict Validation: Rejects missing, non-finite, or sentinel (-1.0) features before inference.
3. Prediction Hysteresis: Requires minimum confidence (>= 0.55), separation margin (>= 0.10),
   and 5-frame persistence (~160-200ms) to transition between posture states.
4. Calibration Validation: Rejects invalid calibration attempts (lateral tilt > 5.0 deg,
   neck tilt > 12.0 deg, slouched head drop < 0.35 ratio, excessive movement std > 0.03).
5. Distance Shift Alert: Detects when user distance shifts by >= 25% from calibrated baseline.
"""

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_posture_features
from src.landmarks import LandmarkPoint, extract_key_landmarks, get_midpoint


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "posture_model.joblib"

# Active 4-Class Target Labels
FOUR_CLASS_LABELS = [
    "correct",
    "slouching",
    "leaning_left",
    "leaning_right",
]
CLASS_LABELS = FOUR_CLASS_LABELS

# Explicitly documented disabled classes
DISABLED_CLASSES = ["leaning_forward"]

# Color Palette for Display (BGR format)
PALETTE = {
    "correct": (80, 200, 80),          # Green
    "slouching": (60, 100, 240),        # Crimson / Red
    "leaning_left": (220, 160, 50),     # Cyan / Blue
    "leaning_right": (200, 100, 220),   # Purple / Magenta
    "uncertain": (50, 190, 240),        # Amber / Yellow-Orange
    "disabled": (140, 140, 140),        # Dimmed grey
    "warning": (40, 140, 240),          # Warning Orange
    "neutral_card": (25, 25, 25),       # Dark Slate
    "card_border": (70, 70, 70),        # Muted border
    "text_main": (240, 240, 240),       # Off-white
    "text_muted": (160, 160, 160),      # Dimmed
}

# Empirical Training Upright Shoulder Distance Center (Sessions 1 & 3 correct-class mean)
TARGET_UPRIGHT_SHOULDER_DISTANCE = 0.370


def normalize_shoulder_distance(
    raw_distance: float,
    baseline_distance: Optional[float],
    target_center: float = TARGET_UPRIGHT_SHOULDER_DISTANCE
) -> float:
    """
    Normalizes shoulder distance relative to calibrated upright baseline:
        D_norm = D * (target_center / D0)

    Evaluated only when baseline_distance is a valid, positive, finite float.
    If baseline_distance is None, non-numeric, <= 0.0, NaN, or Inf, returns
    raw_distance unchanged to guarantee safe fallback to legacy behavior.
    """
    if baseline_distance is None or not isinstance(baseline_distance, (int, float)):
        return raw_distance
    try:
        d0 = float(baseline_distance)
        if math.isnan(d0) or math.isinf(d0) or d0 <= 0.0:
            return raw_distance
        raw_val = float(raw_distance)
        if math.isnan(raw_val) or math.isinf(raw_val):
            return raw_distance
        return raw_val * (target_center / d0)
    except (TypeError, ValueError):
        return raw_distance


class PosturePredictor:
    """
    Manages posture model loading, strict input feature verification, and
    leak-free inference according to the loaded bundle specification.
    """

    def __init__(self, model_path: Optional[Path] = None):
        target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not target_path.exists():
            # Try resolving relative to PROJECT_ROOT or inside PROJECT_ROOT/models
            if (PROJECT_ROOT / target_path).exists():
                target_path = PROJECT_ROOT / target_path
            elif (PROJECT_ROOT / "models" / target_path.name).exists():
                target_path = PROJECT_ROOT / "models" / target_path.name
            else:
                raise FileNotFoundError(
                    f"Posture model artifact not found at: {target_path} "
                    f"(also checked in {PROJECT_ROOT / target_path} and {PROJECT_ROOT / 'models' / target_path.name})"
                )

        self.model_path = target_path
        artifact = joblib.load(self.model_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(f"Invalid model bundle format in {self.model_path}. Expected dict with 'model'.")

        self.model = artifact["model"]
        self.feature_names: List[str] = artifact.get(
            "feature_names",
            ["shoulder_tilt_deg", "neck_angle_deg"]
        )
        self.classes: List[str] = list(artifact.get("classes", CLASS_LABELS))

    def validate_features(
        self,
        features: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[List[float]], str]:
        """
        Verifies that all features required by the loaded model are present,
        finite, and free of sentinel values (-1.0).

        Returns:
            (is_valid, feature_vector, status_message)
        """
        if not features:
            return False, None, "No features extracted"

        vec: List[float] = []
        for feat in self.feature_names:
            val = features.get(feat)
            if val is None:
                return False, None, f"Missing required landmark: {feat}"
            try:
                num = float(val)
                if math.isnan(num) or math.isinf(num):
                    return False, None, f"Non-finite value in {feat}"
                if math.isclose(num, -1.0, abs_tol=1e-5):
                    return False, None, f"Unobserved landmark sentinel in {feat}"
                vec.append(num)
            except (ValueError, TypeError):
                return False, None, f"Invalid numeric data in {feat}"

        return True, vec, "Valid"

    def predict(
        self,
        features: Optional[Dict[str, Any]],
        baseline_distance: Optional[float] = None
    ) -> Tuple[Optional[str], Dict[str, float], str]:
        """
        Executes model inference on validated features.

        When shoulder_distance_raw is among the required model features and a valid
        upright baseline_distance (D0 > 0) is provided, normalizes shoulder_distance_raw
        before vector assembly. When using distance-invariant angular features, runs
        directly on the validated angles.

        Returns:
            (predicted_label, probability_dict, status_message)
        """
        if not features:
            return None, {}, "No features extracted"

        # Apply calibration normalization if shoulder_distance_raw is in model features
        features_to_use = features
        if (
            "shoulder_distance_raw" in self.feature_names
            and baseline_distance is not None
            and isinstance(baseline_distance, (int, float))
            and not (math.isnan(baseline_distance) or math.isinf(baseline_distance))
            and baseline_distance > 0.0
            and "shoulder_distance_raw" in features
        ):
            raw_d = features["shoulder_distance_raw"]
            if raw_d is not None and isinstance(raw_d, (int, float)):
                norm_d = normalize_shoulder_distance(raw_d, baseline_distance)
                features_to_use = dict(features)
                features_to_use["shoulder_distance_raw"] = norm_d

        is_valid, vec, status_msg = self.validate_features(features_to_use)
        if not is_valid or vec is None:
            return None, {}, status_msg

        X = np.array([vec], dtype=np.float64)
        pred_label = str(self.model.predict(X)[0])

        probabilities: Dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(X)[0]
                model_classes = list(getattr(self.model, "classes_", self.classes))
                for c_name, p in zip(model_classes, probs):
                    probabilities[str(c_name)] = round(float(p), 4)
            except Exception:
                probabilities = {}

        return pred_label, probabilities, "Prediction OK"


class PredictionStabilizer:
    """
    Stabilizes noisy single-frame model predictions using a rolling majority vote.
    Retained for backward compatibility and basic smoothing.
    """

    def __init__(self, window_size: int = 7):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.history: Deque[str] = deque(maxlen=window_size)

    def add_prediction(self, raw_label: Optional[str]) -> Optional[str]:
        """
        Appends a valid raw prediction to the rolling buffer and computes
        the majority vote. Returns None if raw_label is None.
        """
        if raw_label is None:
            return None

        self.history.append(raw_label)
        counts = Counter(self.history)

        # Tie-breaker: if counts are equal, prefer the most recent prediction
        most_common = counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            candidate_labels = [lbl for lbl, cnt in most_common if cnt == most_common[0][1]]
            for lbl in reversed(self.history):
                if lbl in candidate_labels:
                    return lbl
            return candidate_labels[0]

        return most_common[0][0]

    def clear(self) -> None:
        """Clears rolling buffer immediately when tracking is lost."""
        self.history.clear()

    @property
    def is_full(self) -> bool:
        return len(self.history) == self.window_size

    @property
    def current_counts(self) -> Dict[str, int]:
        return dict(Counter(self.history))


class PostureStateHysteresis:
    """
    Confidence-based hysteresis and persistence tracker to eliminate prediction
    flip-flopping between posture classes (especially correct vs slouching).

    Design & Justification:
    1. Confidence Threshold: Requires top_prob >= min_confidence (0.55) and
       margin = (top_prob - runner_up) >= min_margin (0.10) to confirm a label.
    2. Hysteresis Hold: When single-frame predictions flutter near the decision
       boundary (e.g., 51% vs 49%), the tracker holds the active state.
    3. Persistence Requirement: To change state to a different posture, that
       new candidate label must persist for persistence_frames (default 5 frames,
       ~160-200ms at 25-30 FPS).
    4. Explicit Uncertain State: If low-confidence or conflicting predictions
       persist for more than max_uncertain_frames (default 15 frames, ~0.5s),
       the active state switches to 'uncertain'.
    5. Clean Reset: Resets immediately to None when tracking is lost.
    """

    def __init__(
        self,
        persistence_frames: int = 5,
        min_confidence: float = 0.55,
        min_margin: float = 0.10,
        max_uncertain_frames: int = 15
    ):
        if persistence_frames < 1:
            raise ValueError("persistence_frames must be >= 1")
        self.persistence_frames = persistence_frames
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.max_uncertain_frames = max_uncertain_frames

        self.current_state: Optional[str] = None
        self.candidate_state: Optional[str] = None
        self.candidate_counter: int = 0
        self.uncertain_counter: int = 0
        self.history: Deque[str] = deque(maxlen=20)

    def update(
        self,
        raw_label: Optional[str],
        probabilities: Optional[Dict[str, float]]
    ) -> Optional[str]:
        """
        Updates hysteresis tracker with new raw prediction and probability distribution.

        Returns:
            The stabilized posture label ('correct', 'slouching', 'leaning_left',
            'leaning_right', 'uncertain', or None if tracking lost).
        """
        if raw_label is None:
            self.clear()
            return None

        probs = probabilities or {}
        sorted_probs = sorted(probs.values(), reverse=True)
        top_prob = sorted_probs[0] if sorted_probs else 1.0
        runner_up = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = top_prob - runner_up

        is_confident = (top_prob >= self.min_confidence) and (margin >= self.min_margin)

        if not is_confident:
            self.uncertain_counter += 1
            # If candidate was building up, decay it
            if self.candidate_counter > 0:
                self.candidate_counter -= 1

            # If uncertainty persists for a prolonged period, declare 'uncertain'
            if self.uncertain_counter >= self.max_uncertain_frames:
                self.current_state = "uncertain"
                self.history.append("uncertain")
                return self.current_state

            # Otherwise hold previous confirmed state if one exists
            if self.current_state is not None:
                self.history.append(self.current_state)
                return self.current_state
            return "uncertain"

        # Input is confident
        self.uncertain_counter = 0

        # Cold start (no current state established yet)
        if self.current_state is None or self.current_state == "uncertain":
            self.current_state = raw_label
            self.candidate_state = None
            self.candidate_counter = 0
            self.history.append(self.current_state)
            return self.current_state

        # Active state is established
        if raw_label == self.current_state:
            # Re-confirming active state: reset any candidate transition
            self.candidate_state = None
            self.candidate_counter = 0
            self.history.append(self.current_state)
            return self.current_state

        # Proposing a transition to a different posture
        if raw_label == self.candidate_state:
            self.candidate_counter += 1
            if self.candidate_counter >= self.persistence_frames:
                # Transition confirmed!
                self.current_state = raw_label
                self.candidate_state = None
                self.candidate_counter = 0
        else:
            self.candidate_state = raw_label
            self.candidate_counter = 1

        self.history.append(self.current_state)
        return self.current_state

    def clear(self) -> None:
        """Clears all tracking state immediately upon tracking loss."""
        self.current_state = None
        self.candidate_state = None
        self.candidate_counter = 0
        self.uncertain_counter = 0
        self.history.clear()

    @property
    def is_uncertain(self) -> bool:
        return self.current_state == "uncertain"


@dataclass
class CalibrationBaseline:
    """Stores established user-specific upright posture baseline metrics."""
    shoulder_distance: float
    head_to_shoulder_ratio: float
    nose_shoulder_dy: float
    nose_shoulder_dz: float
    sample_count: int
    calibrated_at: str


class UprightPostureCalibrator:
    """
    Upright posture baseline calibration and shift detection mechanism.

    Key Principles (Phase 4):
    1. Validation & Rejection: Rejects invalid calibration attempts instead of silently
       accepting slouched, leaning, or unstable baselines.
       - Lateral tilt > 5.0 deg -> Rejected (uneven shoulders).
       - Neck lateral angle > 12.0 deg -> Rejected (head tilted sideways).
       - Head-to-shoulder ratio < 0.35 -> Rejected (slouched head drop).
       - Distance std > 0.03 -> Rejected (excessive movement).
    2. Camera Shift Detection: Detects if user distance changes by >= 25% from baseline.
    3. Separation of Concerns: Calibration does NOT override or force ML predictions to 'correct'.
    """

    STATE_DISABLED = "DISABLED"
    STATE_READY = "READY"
    STATE_CALIBRATING = "CALIBRATING"
    STATE_CALIBRATED = "CALIBRATED"
    STATE_REJECTED = "REJECTED"

    # Calibration Validation Limits
    MAX_VALID_TILT_DEG = 5.0
    MAX_VALID_NECK_DEG = 12.0
    MIN_VALID_HEAD_RATIO = 0.35
    MAX_VALID_DISTANCE_STD = 0.030
    DISTANCE_SHIFT_THRESHOLD_PCT = 25.0

    def __init__(
        self,
        target_duration_sec: float = 3.0,
        surge_threshold_pct: float = 14.0,
        min_required_samples: int = 15
    ):
        self.target_duration_sec = target_duration_sec
        self.surge_threshold_pct = surge_threshold_pct
        self.min_required_samples = min_required_samples

        self.state = self.STATE_READY
        self.baseline: Optional[CalibrationBaseline] = None

        self._start_time: Optional[float] = None
        self._samples_shoulder_dist: List[float] = []
        self._samples_tilt: List[float] = []
        self._samples_neck: List[float] = []
        self._samples_head_ratio: List[float] = []
        self._samples_dy: List[float] = []
        self._samples_dz: List[float] = []
        self.last_rejection_reason: str = ""

    def start_calibration(self) -> None:
        """Starts or restarts the 3-second upright calibration sequence."""
        self.state = self.STATE_CALIBRATING
        self._start_time = time.time()
        self._samples_shoulder_dist.clear()
        self._samples_tilt.clear()
        self._samples_neck.clear()
        self._samples_head_ratio.clear()
        self._samples_dy.clear()
        self._samples_dz.clear()
        self.last_rejection_reason = ""

    def cancel_or_reset(self) -> None:
        """Resets calibration to READY state."""
        self.state = self.STATE_READY
        self.baseline = None
        self._start_time = None
        self._samples_shoulder_dist.clear()
        self._samples_tilt.clear()
        self._samples_neck.clear()
        self._samples_head_ratio.clear()
        self._samples_dy.clear()
        self._samples_dz.clear()
        self.last_rejection_reason = ""

    def toggle_enable(self) -> None:
        """Toggles calibrator between DISABLED and READY."""
        if self.state == self.STATE_DISABLED:
            self.state = self.STATE_READY
        else:
            self.cancel_or_reset()
            self.state = self.STATE_DISABLED

    def update(
        self,
        features: Optional[Dict[str, Any]],
        landmarks: Optional[Dict[str, Optional[LandmarkPoint]]]
    ) -> Dict[str, Any]:
        """
        Updates calibration state machine and evaluates posture baseline validity.

        Returns:
            Dictionary with current state, progress, validation status, and distance shift.
        """
        info: Dict[str, Any] = {
            "state": self.state,
            "progress_pct": 0.0,
            "remaining_sec": 0.0,
            "is_forward_surge": False,
            "is_distance_shifted": False,
            "distance_surge_pct": 0.0,
            "baseline_distance": self.baseline.shoulder_distance if self.baseline else None,
            "baseline_head_ratio": self.baseline.head_to_shoulder_ratio if self.baseline else None,
            "message": "",
            "forward_lean_disabled": True,
        }

        if self.state == self.STATE_DISABLED:
            info["message"] = "Calibration Disabled"
            return info

        # Extract landmarks and features
        left_sh = landmarks.get("left_shoulder") if landmarks else None
        right_sh = landmarks.get("right_shoulder") if landmarks else None
        nose = landmarks.get("nose") if landmarks else None
        mid_sh = get_midpoint(left_sh, right_sh) if left_sh and right_sh else None

        sh_dist = features.get("shoulder_distance_raw") if features else None
        sh_tilt = features.get("shoulder_tilt_deg") if features else None
        neck_ang = features.get("neck_angle_deg") if features else None
        head_ratio = features.get("head_to_shoulder_ratio") if features else None

        # Fallback ratio calculation from landmarks if not present in features
        if head_ratio is None and mid_sh and nose and sh_dist and sh_dist > 0:
            head_ratio = (mid_sh.y - nose.y) / float(sh_dist)

        has_valid_inputs = (
            sh_dist is not None and isinstance(sh_dist, (int, float)) and sh_dist > 0.0 and
            sh_tilt is not None and isinstance(sh_tilt, (int, float)) and
            neck_ang is not None and isinstance(neck_ang, (int, float)) and
            nose is not None and mid_sh is not None
        )

        if self.state == self.STATE_CALIBRATING:
            if not self._start_time:
                self._start_time = time.time()

            elapsed = time.time() - self._start_time
            remaining = max(0.0, self.target_duration_sec - elapsed)
            progress = min(100.0, (elapsed / self.target_duration_sec) * 100.0)

            info["progress_pct"] = round(progress, 1)
            info["remaining_sec"] = round(remaining, 1)

            if has_valid_inputs:
                dy = float(nose.y - mid_sh.y)
                dz = float(nose.z - mid_sh.z)
                self._samples_shoulder_dist.append(float(sh_dist))
                self._samples_tilt.append(float(sh_tilt))
                self._samples_neck.append(float(neck_ang))
                if head_ratio is not None:
                    self._samples_head_ratio.append(float(head_ratio))
                self._samples_dy.append(dy)
                self._samples_dz.append(dz)

            # Check if duration completed
            if elapsed >= self.target_duration_sec:
                if len(self._samples_shoulder_dist) < self.min_required_samples:
                    self.state = self.STATE_REJECTED
                    self.last_rejection_reason = "Insufficient stable frames. Keep face & shoulders visible."
                    info["state"] = self.STATE_REJECTED
                    info["message"] = f"Calibration failed: {self.last_rejection_reason}"
                    return info

                # Validate upright baseline posture
                mean_tilt = float(np.mean(self._samples_tilt))
                mean_neck = float(np.mean(self._samples_neck))
                mean_ratio = float(np.mean(self._samples_head_ratio)) if self._samples_head_ratio else 0.40
                std_dist = float(np.std(self._samples_shoulder_dist))

                if abs(mean_tilt) > self.MAX_VALID_TILT_DEG:
                    self.state = self.STATE_REJECTED
                    self.last_rejection_reason = f"Uneven shoulders ({mean_tilt:+.1f} deg). Keep level."
                elif mean_neck > self.MAX_VALID_NECK_DEG:
                    self.state = self.STATE_REJECTED
                    self.last_rejection_reason = f"Neck tilted ({mean_neck:.1f} deg). Face forward."
                elif mean_ratio < self.MIN_VALID_HEAD_RATIO:
                    self.state = self.STATE_REJECTED
                    self.last_rejection_reason = f"Slouched baseline (ratio {mean_ratio:.2f} < 0.35). Sit tall."
                elif std_dist > self.MAX_VALID_DISTANCE_STD:
                    self.state = self.STATE_REJECTED
                    self.last_rejection_reason = f"Excessive movement (std {std_dist:.3f}). Stay still."
                else:
                    # Valid baseline accepted!
                    self.baseline = CalibrationBaseline(
                        shoulder_distance=float(np.mean(self._samples_shoulder_dist)),
                        head_to_shoulder_ratio=mean_ratio,
                        nose_shoulder_dy=float(np.mean(self._samples_dy)),
                        nose_shoulder_dz=float(np.mean(self._samples_dz)),
                        sample_count=len(self._samples_shoulder_dist),
                        calibrated_at=datetime.now().strftime("%H:%M:%S")
                    )
                    self.state = self.STATE_CALIBRATED
                    self.last_rejection_reason = ""
                    info["state"] = self.STATE_CALIBRATED
                    info["baseline_distance"] = round(self.baseline.shoulder_distance, 4)
                    info["baseline_head_ratio"] = round(self.baseline.head_to_shoulder_ratio, 3)
                    info["message"] = f"Calibrated Upright (D0={self.baseline.shoulder_distance:.3f})"
                    return info

                # If rejected
                info["state"] = self.STATE_REJECTED
                info["message"] = f"Calibration rejected: {self.last_rejection_reason}"
                return info

            info["message"] = f"Hold upright: {remaining:.1f}s remaining"
            return info

        if self.state == self.STATE_REJECTED:
            info["message"] = f"Rejected: {self.last_rejection_reason} Press 'c' to retry."
            return info

        if self.state == self.STATE_CALIBRATED and self.baseline is not None:
            info["baseline_distance"] = round(self.baseline.shoulder_distance, 4)
            info["baseline_head_ratio"] = round(self.baseline.head_to_shoulder_ratio, 3)
            if has_valid_inputs:
                d0 = self.baseline.shoulder_distance
                surge = ((float(sh_dist) - d0) / d0) * 100.0
                info["distance_surge_pct"] = round(surge, 1)

                # Distance shift warning
                if abs(surge) >= self.DISTANCE_SHIFT_THRESHOLD_PCT:
                    info["is_distance_shifted"] = True
                    info["message"] = f"Distance changed ({surge:+.0f}%). Recalibrate [c]"
                else:
                    info["message"] = f"Calibrated OK (Dist: {surge:+.0f}%)"
            else:
                info["message"] = "Calibrated (tracking lost)"

            return info

        info["message"] = "Press 'c' to calibrate upright posture"
        return info


# Alias ForwardLeanCalibrator to UprightPostureCalibrator for backward compatibility
ForwardLeanCalibrator = UprightPostureCalibrator


def draw_postureguard_hud(
    frame: np.ndarray,
    smoothed_label: Optional[str],
    raw_label: Optional[str],
    probabilities: Dict[str, float],
    features: Optional[Dict[str, Any]],
    status_msg: str,
    calib_info: Dict[str, Any],
    fps: float
) -> None:
    """
    Renders a modern, highly legible 4-class HUD overlay with glassmorphism card styling.
    """
    h, w, _ = frame.shape

    # 1. Main Left Posture Status Card
    card_w, card_h = 370, 245
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), PALETTE["neutral_card"], -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + card_w, 15 + card_h), PALETTE["card_border"], 1)

    # Header
    cv2.putText(frame, "POSTUREGUARD 4-CLASS", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15 + card_w - 85, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PALETTE["text_muted"], 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (30, 52), (15 + card_w - 15, 52), PALETTE["card_border"], 1)

    # Posture Label Display
    if smoothed_label == "uncertain":
        cv2.putText(frame, "UNCERTAIN", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, PALETTE["uncertain"], 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Posture Ambiguous / Borderline (stabilizing...)",
            (30, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            PALETTE["text_muted"],
            1,
            cv2.LINE_AA
        )
    elif smoothed_label:
        color = PALETTE.get(smoothed_label, PALETTE["correct"])
        display_text = smoothed_label.replace("_", " ").upper()
        cv2.putText(frame, display_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        # Confidence
        prob_val = probabilities.get(smoothed_label, 0.0) * 100.0
        raw_str = raw_label or "--"
        cv2.putText(
            frame,
            f"Confidence: {prob_val:.1f}% (raw: {raw_str})",
            (30, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            PALETTE["text_muted"],
            1,
            cv2.LINE_AA
        )
    else:
        cv2.putText(frame, "NO DETECTION", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (80, 80, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, status_msg, (30, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PALETTE["text_muted"], 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (30, 118), (15 + card_w - 15, 118), PALETTE["card_border"], 1)

    # Metrics Display
    sh_tilt = features.get("shoulder_tilt_deg") if features else None
    neck_ang = features.get("neck_angle_deg") if features else None
    head_ratio = features.get("head_to_shoulder_ratio") if features else None
    sh_dist = features.get("shoulder_distance_raw") if features else None

    tilt_str = f"{sh_tilt:+.1f} deg" if sh_tilt is not None else "--"
    neck_str = f"{neck_ang:.1f} deg" if neck_ang is not None else "--"
    ratio_str = f"{head_ratio:.2f}" if head_ratio is not None else "--"
    dist_str = f"{sh_dist:.3f}" if sh_dist is not None else "--"

    metrics_rows = [
        ("Shoulder Tilt  :", tilt_str),
        ("Neck Angle     :", neck_str),
        ("Head Ratio     :", ratio_str),
        ("Shoulder Dist  :", dist_str),
    ]

    y_pos = 138
    for label_str, val_str in metrics_rows:
        cv2.putText(frame, label_str, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PALETTE["text_muted"], 1, cv2.LINE_AA)
        cv2.putText(frame, val_str, (170, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PALETTE["text_main"], 1, cv2.LINE_AA)
        y_pos += 20

    # Instructions row
    cv2.putText(
        frame,
        "[q] Quit  |  [c] Calibrate Upright  |  [r] Reset",
        (30, card_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (130, 130, 130),
        1,
        cv2.LINE_AA
    )

    # 2. Top-Right Upright Calibration Card
    calib_state = calib_info.get("state", UprightPostureCalibrator.STATE_READY)
    calib_w, calib_h = 350, 105
    calib_x = w - calib_w - 15
    calib_overlay = frame.copy()
    cv2.rectangle(calib_overlay, (calib_x, 15), (calib_x + calib_w, 15 + calib_h), PALETTE["neutral_card"], -1)
    cv2.addWeighted(calib_overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (calib_x, 15), (calib_x + calib_w, 15 + calib_h), PALETTE["card_border"], 1)

    cv2.putText(
        frame,
        "UPRIGHT POSTURE CALIBRATION",
        (calib_x + 15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 200, 240),
        1,
        cv2.LINE_AA
    )

    if calib_state == UprightPostureCalibrator.STATE_CALIBRATING:
        rem = calib_info.get("remaining_sec", 0.0)
        prog = calib_info.get("progress_pct", 0.0)
        cv2.putText(
            frame,
            f"Hold Upright: {rem:.1f}s remaining",
            (calib_x + 15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (60, 220, 240),
            1,
            cv2.LINE_AA
        )
        # Progress bar
        bar_x = calib_x + 15
        bar_y = 72
        bar_w = calib_w - 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50, 50, 50), -1)
        fill_w = int((prog / 100.0) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 8), (60, 220, 240), -1)
        cv2.putText(
            frame,
            "Validating posture stability...",
            (calib_x + 15, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            PALETTE["text_muted"],
            1,
            cv2.LINE_AA
        )

    elif calib_state == UprightPostureCalibrator.STATE_CALIBRATED:
        is_shift = calib_info.get("is_distance_shifted", False)
        surge_val = calib_info.get("distance_surge_pct", 0.0)
        d0_val = calib_info.get("baseline_distance", 0.0)

        shift_color = PALETTE["warning"] if is_shift else PALETTE["correct"]
        shift_text = f"Dist Shift ({surge_val:+.0f}%)" if is_shift else f"Dist OK ({surge_val:+.0f}%)"

        cv2.putText(
            frame,
            f"Baseline D0: {d0_val:.3f} | {shift_text}",
            (calib_x + 15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            shift_color,
            1,
            cv2.LINE_AA
        )
        if is_shift:
            cv2.putText(
                frame,
                "Warning: User distance changed >25%. Recalibrate [c]",
                (calib_x + 15, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                PALETTE["warning"],
                1,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame,
                "Baseline active | Dist-invariant model running",
                (calib_x + 15, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                PALETTE["text_muted"],
                1,
                cv2.LINE_AA
            )
        cv2.putText(
            frame,
            "[Forward-lean disabled in 4-class mode]",
            (calib_x + 15, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            PALETTE["disabled"],
            1,
            cv2.LINE_AA
        )

    elif calib_state == UprightPostureCalibrator.STATE_REJECTED:
        rejection_msg = calib_info.get("message", "Calibration rejected")
        cv2.putText(
            frame,
            "CALIBRATION REJECTED",
            (calib_x + 15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            PALETTE["warning"],
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            rejection_msg[:46],
            (calib_x + 15, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            PALETTE["text_main"],
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Sit upright & steady, then press 'c' to retry",
            (calib_x + 15, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            PALETTE["text_muted"],
            1,
            cv2.LINE_AA
        )

    else:
        cv2.putText(
            frame,
            "Press 'c' to capture 3s upright baseline",
            (calib_x + 15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            PALETTE["text_muted"],
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Validates level shoulders and steady posture",
            (calib_x + 15, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (140, 140, 140),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "[Forward-lean disabled in 4-class mode]",
            (calib_x + 15, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            PALETTE["disabled"],
            1,
            cv2.LINE_AA
        )

    # 3. Model Diagnostics Card (Feature & 4-Class Probability Inspection)
    diag_w, diag_h = 440, 135
    diag_x = 15
    diag_y = min(h - diag_h - 15, card_h + 25)

    if h >= 420:
        diag_overlay = frame.copy()
        cv2.rectangle(diag_overlay, (diag_x, diag_y), (diag_x + diag_w, diag_y + diag_h), PALETTE["neutral_card"], -1)
        cv2.addWeighted(diag_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (diag_x, diag_y), (diag_x + diag_w, diag_y + diag_h), (80, 180, 240), 1)

        cv2.putText(
            frame,
            "MODEL DIAGNOSTICS (Distance-Invariant 4-Class)",
            (diag_x + 12, diag_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (80, 200, 255),
            1,
            cv2.LINE_AA
        )

        # Input feature values
        if features and sh_tilt is not None and neck_ang is not None:
            feat_text = f"Features (angles): tilt={sh_tilt:+.1f} deg, neck={neck_ang:.1f} deg [Dist-Invariant]"
        else:
            feat_text = "Input Vector: [Landmarks not fully detected]"

        cv2.putText(
            frame,
            feat_text,
            (diag_x + 12, diag_y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            PALETTE["text_main"],
            1,
            cv2.LINE_AA
        )

        # 4 class probabilities
        p_correct = probabilities.get("correct", 0.0) * 100.0
        p_slouch = probabilities.get("slouching", 0.0) * 100.0
        p_ll = probabilities.get("leaning_left", 0.0) * 100.0
        p_lr = probabilities.get("leaning_right", 0.0) * 100.0

        prob_line1 = f"Correct: {p_correct:4.1f}%  |  Slouch: {p_slouch:4.1f}%"
        prob_line2 = f"Lean_Left: {p_ll:4.1f}%  |  Lean_Right: {p_lr:4.1f}%"

        cv2.putText(frame, prob_line1, (diag_x + 12, diag_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PALETTE["text_main"], 1, cv2.LINE_AA)
        cv2.putText(frame, prob_line2, (diag_x + 12, diag_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, PALETTE["text_main"], 1, cv2.LINE_AA)

        # Calibration & Status Note
        if calib_state == UprightPostureCalibrator.STATE_CALIBRATED:
            d0_val = calib_info.get("baseline_distance", 0.0)
            diag_note = f"Calibration Active: D0={d0_val:.3f} | Dist-shift monitoring active"
            note_color = PALETTE["correct"]
        elif calib_state == UprightPostureCalibrator.STATE_REJECTED:
            diag_note = "Calibration rejected: Slouch or tilt detected during baseline"
            note_color = PALETTE["warning"]
        else:
            diag_note = "Angular classifier active: Distance-independent classification"
            note_color = PALETTE["correct"]

        cv2.putText(frame, diag_note, (diag_x + 12, diag_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, note_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Temporal Hysteresis active: 5-frame persistence prevents flip-flopping.",
            (diag_x + 12, diag_y + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            PALETTE["text_muted"],
            1,
            cv2.LINE_AA
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PostureGuard Real-Time Sitting Posture Detection")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to posture model bundle (.joblib). Defaults to active posture_model.joblib."
    )
    args = parser.parse_args()
    model_path = Path(args.model)

    print("=" * 72)
    print("PostureGuard Real-Time Webcam Detection (Posture Monitoring Mode)")
    print("=" * 72)

    # Step 1: Load Posture Model
    try:
        predictor = PosturePredictor(model_path)
        print(f"Model loaded successfully from: {model_path}")
        print(f"Required features ({len(predictor.feature_names)}): {predictor.feature_names}")
        print(f"Target classes   ({len(predictor.classes)}): {predictor.classes}")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    hysteresis = PostureStateHysteresis(persistence_frames=5, min_confidence=0.55, min_margin=0.10)
    calibrator = UprightPostureCalibrator(target_duration_sec=3.0)

    # Step 2: Open Webcam with DirectShow fallback
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not access webcam. Please check device connectivity.", file=sys.stderr)
        sys.exit(1)

    print("Camera successfully opened! Starting real-time detection.")
    print("Controls: Press 'q' to quit | 'c' to calibrate upright posture | 'r' to reset calibration.\n")

    # Step 3: Initialize MediaPipe Pose Detection
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:

        prev_time = time.time()
        fps = 0.0
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to grab frame. Exiting loop...")
                    break

                # FPS Calculation
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # Color conversion for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose.process(rgb_frame)
                rgb_frame.flags.writeable = True

                is_detected = results.pose_landmarks is not None
                features = None
                landmarks = None
                raw_pred = None
                probs: Dict[str, float] = {}
                status_msg = "No Person Detected"

                if is_detected:
                    # Draw pose skeleton
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    landmarks = extract_key_landmarks(results.pose_landmarks, min_visibility=0.5)
                    features = extract_posture_features(landmarks)

                    # Update calibration state machine first to obtain active baseline
                    calib_info = calibrator.update(features, landmarks)
                    calibrated_d0 = (
                        calibrator.baseline.shoulder_distance
                        if (calibrator.state == UprightPostureCalibrator.STATE_CALIBRATED and calibrator.baseline)
                        else None
                    )

                    # Execute model prediction
                    raw_pred, probs, status_msg = predictor.predict(features, baseline_distance=calibrated_d0)
                    stabilized_pred = hysteresis.update(raw_pred, probs)

                    # Periodic console diagnostic output (every ~1.5s)
                    frame_idx += 1
                    if frame_idx % 30 == 0 and features:
                        sh_tilt = features.get("shoulder_tilt_deg")
                        neck_ang = features.get("neck_angle_deg")
                        sh_dist = features.get("shoulder_distance_raw")
                        p_corr = probs.get("correct", 0.0) * 100
                        p_slouch = probs.get("slouching", 0.0) * 100
                        p_left = probs.get("leaning_left", 0.0) * 100
                        p_right = probs.get("leaning_right", 0.0) * 100
                        print(
                            f"[DIAGNOSTIC] Angles: [tilt={sh_tilt:+.2f} deg, neck={neck_ang:.2f} deg] | "
                            f"Probs: correct={p_corr:.1f}%, slouch={p_slouch:.1f}%, L={p_left:.1f}%, R={p_right:.1f}% | "
                            f"Stabilized: {stabilized_pred} (raw: {raw_pred})"
                        )
                else:
                    # Tracking lost: clear temporal hysteresis state immediately
                    hysteresis.clear()
                    stabilized_pred = None
                    calib_info = calibrator.update(None, None)

                # Render HUD
                draw_postureguard_hud(
                    frame=frame,
                    smoothed_label=stabilized_pred,
                    raw_label=raw_pred,
                    probabilities=probs,
                    features=features,
                    status_msg=status_msg,
                    calib_info=calib_info,
                    fps=fps
                )

                # Display frame
                cv2.imshow("PostureGuard - Real-Time Posture Detection", frame)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exit key 'q' pressed. Shutting down...")
                    break
                elif key == ord('c'):
                    print("Triggering upright posture calibration (3.0 seconds)...")
                    calibrator.start_calibration()
                elif key == ord('r'):
                    print("Resetting calibration.")
                    calibrator.cancel_or_reset()

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam released and OpenCV windows closed.")


if __name__ == "__main__":
    main()
