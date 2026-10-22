"""
Landmark Extraction Module for PostureGuard
-------------------------------------------
This module extracts and structures the key anatomical landmarks needed
for sitting posture detection and ergonomics analysis.

Educational Concepts:
1. What a Landmark is:
   - A landmark is a specific anatomical keypoint (joint or facial feature)
     predicted by MediaPipe Pose's neural network (33 total keypoints).

2. What x, y, z Coordinates Represent:
   - x: Horizontal position normalized to [0.0, 1.0] across image width
        (0.0 = left edge, 1.0 = right edge).
   - y: Vertical position normalized to [0.0, 1.0] across image height
        (0.0 = top edge, 1.0 = bottom edge).
   - z: Relative depth along camera axis (smaller/negative values are closer
        to the camera; rough origin is centered around the hips).
   - visibility: Confidence score [0.0, 1.0] estimating whether the landmark
        is visible in the frame and not occluded.

3. Why Confidence Filtering Matters:
   - When body parts are hidden behind a desk or turned away, low-confidence
     landmark coordinates can jump erratically. Filtering out coordinates
     below a confidence threshold prevents noisy calculations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import mediapipe as mp


# MediaPipe Pose Landmark Enum reference
mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark


@dataclass
class LandmarkPoint:
    """Represents a single 3D anatomical landmark with visibility confidence."""
    x: float
    y: float
    z: float
    visibility: float

    @property
    def xy(self) -> Tuple[float, float]:
        """Return 2D coordinates (x, y)."""
        return self.x, self.y

    @property
    def xyz(self) -> Tuple[float, float, float]:
        """Return 3D coordinates (x, y, z)."""
        return self.x, self.y, self.z


# Target landmarks critical for sitting posture analysis
POSTURE_LANDMARK_MAP = {
    "nose": PoseLandmark.NOSE,
    "left_ear": PoseLandmark.LEFT_EAR,
    "right_ear": PoseLandmark.RIGHT_EAR,
    "left_shoulder": PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
    "left_hip": PoseLandmark.LEFT_HIP,
    "right_hip": PoseLandmark.RIGHT_HIP,
    "left_knee": PoseLandmark.LEFT_KNEE,
    "right_knee": PoseLandmark.RIGHT_KNEE,
    "left_ankle": PoseLandmark.LEFT_ANKLE,
    "right_ankle": PoseLandmark.RIGHT_ANKLE,
}


def extract_key_landmarks(
    results_pose_landmarks: Any,
    min_visibility: float = 0.5
) -> Dict[str, Optional[LandmarkPoint]]:
    """
    Extracts the key landmarks required for sitting posture detection.

    Args:
        results_pose_landmarks: The `results.pose_landmarks` object from MediaPipe.
        min_visibility: Minimum visibility threshold (0.0 to 1.0) to consider
                        a landmark reliably detected.

    Returns:
        Dictionary mapping landmark name (e.g. 'left_shoulder') to a LandmarkPoint,
        or None if the landmark is missing or below the visibility threshold.
    """
    landmarks_dict: Dict[str, Optional[LandmarkPoint]] = {}

    # If MediaPipe did not detect any person at all, return all None
    if not results_pose_landmarks or not hasattr(results_pose_landmarks, "landmark"):
        for name in POSTURE_LANDMARK_MAP:
            landmarks_dict[name] = None
        return landmarks_dict

    raw_landmarks = results_pose_landmarks.landmark

    for name, enum_val in POSTURE_LANDMARK_MAP.items():
        idx = enum_val.value
        if idx < len(raw_landmarks):
            lm = raw_landmarks[idx]
            # Safely handle missing/low-confidence landmarks
            if lm.visibility >= min_visibility:
                landmarks_dict[name] = LandmarkPoint(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(lm.visibility)
                )
            else:
                landmarks_dict[name] = None
        else:
            landmarks_dict[name] = None

    return landmarks_dict


def get_midpoint(
    p1: Optional[LandmarkPoint],
    p2: Optional[LandmarkPoint]
) -> Optional[LandmarkPoint]:
    """
    Computes the midpoint between two landmarks (e.g. mid-shoulder or mid-hip).
    Returns None if either landmark is missing.
    """
    if p1 is None or p2 is None:
        return None

    return LandmarkPoint(
        x=(p1.x + p2.x) / 2.0,
        y=(p1.y + p2.y) / 2.0,
        z=(p1.z + p2.z) / 2.0,
        visibility=min(p1.visibility, p2.visibility)
    )
