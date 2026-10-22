"""
Geometric Feature Extraction Module for PostureGuard
-----------------------------------------------------
This module computes geometric angles, distances, and normalized spatial
metrics from extracted body landmarks for sitting posture analysis.

Educational Concepts:
1. Why We Calculate Angles and Distances:
   - Raw (x, y) coordinates change dramatically whenever a person shifts inside
     the webcam frame. Angles (such as torso tilt, shoulder levelness, and hip
     flexion) describe the relative physical alignment of the body and are
     naturally invariant to where the person is positioned within the frame.

2. Why Normalization is Useful:
   - When a person sits closer to the webcam, distance measurements between
     landmarks appear much larger than when they sit farther away (perspective scaling).
   - By normalizing distances against a stable body metric (such as torso length:
     distance between mid-shoulder and mid-hip), distance features become scale-invariant.

3. Handling Missing Landmarks:
   - If MediaPipe cannot detect a joint with high confidence, these functions
     return None instead of crashing or generating invalid numbers.
"""

import math
from typing import Dict, Optional, Tuple
try:
    from src.landmarks import LandmarkPoint, get_midpoint
except ModuleNotFoundError:
    from landmarks import LandmarkPoint, get_midpoint


def euclidean_distance_2d(
    p1: Optional[LandmarkPoint],
    p2: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Computes Euclidean distance between two 2D points in normalized space.
    Returns None if either point is missing.
    """
    if p1 is None or p2 is None:
        return None
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def euclidean_distance_3d(
    p1: Optional[LandmarkPoint],
    p2: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Computes 3D Euclidean distance (x, y, z) between two points.
    Returns None if either point is missing.
    """
    if p1 is None or p2 is None:
        return None
    return math.sqrt(
        (p2.x - p1.x) ** 2 +
        (p2.y - p1.y) ** 2 +
        (p2.z - p1.z) ** 2
    )


def calculate_ear_distance(
    left_ear: Optional[LandmarkPoint],
    right_ear: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Computes Euclidean distance between left and right ears.
    Returns None if either ear landmark is missing.
    """
    return euclidean_distance_2d(left_ear, right_ear)


def calculate_shoulder_to_ear_ratio(
    left_shoulder: Optional[LandmarkPoint],
    right_shoulder: Optional[LandmarkPoint],
    left_ear: Optional[LandmarkPoint],
    right_ear: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Computes the scale-invariant ratio of shoulder width to ear-to-ear distance:
        ratio = shoulder_width / ear_distance

    Purpose:
    - Provides scale and camera-distance invariance in laptop close-up mode
      when hips and legs are off-screen.
    - If either landmark is missing, or ear distance is near-zero (< 1e-4),
      returns None without crashing.
    """
    shoulder_dist = euclidean_distance_2d(left_shoulder, right_shoulder)
    ear_dist = euclidean_distance_2d(left_ear, right_ear)

    if shoulder_dist is None or ear_dist is None or ear_dist < 1e-4:
        return None

    return round(shoulder_dist / ear_dist, 4)



def calculate_3point_angle(
    a: Optional[LandmarkPoint],
    b: Optional[LandmarkPoint],
    c: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Calculates the 2D joint angle at vertex b formed by rays ba and bc (in degrees).

    Example:
        Hip angle: a=Shoulder, b=Hip, c=Knee
        Knee angle: a=Hip, b=Knee, c=Ankle

    Returns:
        Angle in degrees [0.0, 180.0], or None if any landmark is missing.
    """
    if a is None or b is None or c is None:
        return None

    # Vector BA
    ba_x = a.x - b.x
    ba_y = a.y - b.y

    # Vector BC
    bc_x = c.x - b.x
    bc_y = c.y - b.y

    # Dot product and magnitudes
    dot_product = (ba_x * bc_x) + (ba_y * bc_y)
    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None

    # Clamp cosine to [-1.0, 1.0] to prevent floating point domain errors
    cosine_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    angle_rad = math.acos(cosine_angle)
    return round(math.degrees(angle_rad), 2)


def calculate_shoulder_tilt(
    left_shoulder: Optional[LandmarkPoint],
    right_shoulder: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Calculates the tilt angle of the shoulder line relative to the horizontal plane.

    Returns:
        Signed angle in degrees:
        - 0.0° means shoulders are perfectly level horizontally.
        - Positive/negative degrees indicate leaning to the left or right side.
        Returns None if either shoulder is missing.
    """
    if left_shoulder is None or right_shoulder is None:
        return None

    dx = right_shoulder.x - left_shoulder.x
    dy = right_shoulder.y - left_shoulder.y

    # In image space, dy=0 means perfectly horizontal
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # Normalize to small deviation from horizontal line (around 0 degrees)
    if angle_deg > 90.0:
        angle_deg -= 180.0
    elif angle_deg < -90.0:
        angle_deg += 180.0

    return round(angle_deg, 2)


def calculate_torso_angle(
    mid_shoulder: Optional[LandmarkPoint],
    mid_hip: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Calculates the torso inclination angle relative to the vertical axis.
    Vector originates at mid-hip and points upward to mid-shoulder.

    Returns:
        Angle in degrees from vertical:
        - ~0.0°: Perfectly upright spine.
        - Larger values: Leaning sideways or slouching forward/backward.
        Returns None if either midpoint is missing.
    """
    if mid_shoulder is None or mid_hip is None:
        return None

    dx = mid_shoulder.x - mid_hip.x
    # In image coordinates, shoulder is higher up (smaller y), so (mid_shoulder.y - mid_hip.y) is negative
    dy = mid_shoulder.y - mid_hip.y

    # Calculate angle relative to vertical vector pointing up: (0, -1)
    # Upward vector components: vx = dx, vy = -dy
    upward_dy = -dy
    if abs(upward_dy) < 1e-6 and abs(dx) < 1e-6:
        return 0.0

    angle_rad = math.atan2(abs(dx), upward_dy)
    return round(math.degrees(angle_rad), 2)


def calculate_neck_angle(
    nose: Optional[LandmarkPoint],
    mid_shoulder: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Calculates the neck/head inclination angle from the vertical axis.
    Measures deviation of the head (nose) from the center of the shoulders.

    Returns:
        Angle in degrees relative to vertical (0.0° = head aligned directly above shoulders).
        Returns None if nose or mid-shoulder is missing.
    """
    if nose is None or mid_shoulder is None:
        return None

    dx = nose.x - mid_shoulder.x
    dy = nose.y - mid_shoulder.y
    upward_dy = -dy

    if abs(upward_dy) < 1e-6 and abs(dx) < 1e-6:
        return 0.0

    angle_rad = math.atan2(abs(dx), upward_dy)
    return round(math.degrees(angle_rad), 2)


def calculate_head_to_shoulder_ratio(
    nose: Optional[LandmarkPoint],
    left_shoulder: Optional[LandmarkPoint],
    right_shoulder: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Computes the scale- and camera-distance-invariant ratio of vertical head
    elevation to horizontal shoulder width:
        ratio = (mid_shoulder.y - nose.y) / shoulder_distance

    In standard normalized image space, shoulder y > nose y when head is above
    shoulders. When sitting upright, this ratio is high (~0.45 to ~0.65).
    When slouching back or hunching, head elevation drops (~0.20 to ~0.35).
    Returns None if any required landmark is missing or shoulder width < 1e-4.
    """
    if nose is None or left_shoulder is None or right_shoulder is None:
        return None
    mid_sh = get_midpoint(left_shoulder, right_shoulder)
    if mid_sh is None:
        return None
    sh_dist = euclidean_distance_2d(left_shoulder, right_shoulder)
    if sh_dist is None or sh_dist < 1e-4:
        return None
    head_elevation = mid_sh.y - nose.y
    return round(head_elevation / sh_dist, 4)


def calculate_head_z_distance(
    nose: Optional[LandmarkPoint],
    mid_shoulder: Optional[LandmarkPoint]
) -> Optional[float]:
    """
    Computes MediaPipe relative depth difference between nose and mid-shoulder:
        dz = nose.z - mid_shoulder.z

    Negative values indicate nose is in front of the shoulder coronal plane.
    Returns None if nose or mid_shoulder is missing.
    """
    if nose is None or mid_shoulder is None:
        return None
    return round(float(nose.z - mid_shoulder.z), 4)


def extract_posture_features(
    landmarks: Dict[str, Optional[LandmarkPoint]]
) -> Dict[str, Optional[float]]:
    """
    Extracts all primary geometric posture features from key landmarks.

    Features calculated:
    - shoulder_distance_raw: Euclidean distance between left & right shoulders
    - ear_distance_raw: Euclidean distance between left & right ears
    - shoulder_to_ear_ratio: Ratio of shoulder width to ear distance
    - head_to_shoulder_ratio: Scale-invariant vertical head elevation over shoulder width
    - head_z_distance: Relative depth difference (nose.z - mid_shoulder.z)
    - hip_distance_raw: Euclidean distance between left & right hips
    - torso_length: Distance between mid-shoulder and mid-hip (reference scale)
    - shoulder_distance_norm: Normalized shoulder distance (relative to torso length)
    - hip_distance_norm: Normalized hip distance (relative to torso length)
    - shoulder_tilt_deg: Shoulder line deviation from horizontal
    - torso_angle_deg: Torso spine inclination from vertical
    - neck_angle_deg: Neck/head tilt from vertical
    - left_hip_angle_deg: Angle at left hip (shoulder-hip-knee)
    - right_hip_angle_deg: Angle at right hip (shoulder-hip-knee)
    - left_knee_angle_deg: Angle at left knee (hip-knee-ankle)
    - right_knee_angle_deg: Angle at right knee (hip-knee-ankle)

    Returns:
        Dictionary of feature names mapped to float values (or None if missing).
    """
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")
    left_ear = landmarks.get("left_ear")
    right_ear = landmarks.get("right_ear")
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")
    left_knee = landmarks.get("left_knee")
    right_knee = landmarks.get("right_knee")
    left_ankle = landmarks.get("left_ankle")
    right_ankle = landmarks.get("right_ankle")
    nose = landmarks.get("nose")

    # Midpoints
    mid_shoulder = get_midpoint(left_shoulder, right_shoulder)
    mid_hip = get_midpoint(left_hip, right_hip)

    # 1. Distances (Raw)
    shoulder_dist_raw = euclidean_distance_2d(left_shoulder, right_shoulder)
    ear_dist_raw = calculate_ear_distance(left_ear, right_ear)
    shoulder_to_ear_ratio = calculate_shoulder_to_ear_ratio(left_shoulder, right_shoulder, left_ear, right_ear)
    head_to_shoulder_ratio = calculate_head_to_shoulder_ratio(nose, left_shoulder, right_shoulder)
    head_z_dist = calculate_head_z_distance(nose, mid_shoulder)
    hip_dist_raw = euclidean_distance_2d(left_hip, right_hip)
    torso_len = euclidean_distance_2d(mid_shoulder, mid_hip)

    # 2. Normalized Distances (divided by torso length for scale invariance)
    shoulder_dist_norm = None
    hip_dist_norm = None
    if torso_len and torso_len > 1e-4:
        if shoulder_dist_raw is not None:
            shoulder_dist_norm = round(shoulder_dist_raw / torso_len, 4)
        if hip_dist_raw is not None:
            hip_dist_norm = round(hip_dist_raw / torso_len, 4)

    # 3. Angles (Degrees)
    shoulder_tilt = calculate_shoulder_tilt(left_shoulder, right_shoulder)
    torso_angle = calculate_torso_angle(mid_shoulder, mid_hip)
    neck_angle = calculate_neck_angle(nose, mid_shoulder)

    # Hip Angles (Shoulder - Hip - Knee)
    left_hip_angle = calculate_3point_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_3point_angle(right_shoulder, right_hip, right_knee)

    # Knee Angles (Hip - Knee - Ankle)
    left_knee_angle = calculate_3point_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_3point_angle(right_hip, right_knee, right_ankle)

    return {
        "shoulder_distance_raw": round(shoulder_dist_raw, 4) if shoulder_dist_raw else None,
        "ear_distance_raw": round(ear_dist_raw, 4) if ear_dist_raw else None,
        "shoulder_to_ear_ratio": shoulder_to_ear_ratio,
        "head_to_shoulder_ratio": head_to_shoulder_ratio,
        "head_z_distance": head_z_dist,
        "hip_distance_raw": round(hip_dist_raw, 4) if hip_dist_raw else None,
        "torso_length": round(torso_len, 4) if torso_len else None,
        "shoulder_distance_norm": shoulder_dist_norm,
        "hip_distance_norm": hip_dist_norm,
        "shoulder_tilt_deg": shoulder_tilt,
        "torso_angle_deg": torso_angle,
        "neck_angle_deg": neck_angle,
        "left_hip_angle_deg": left_hip_angle,
        "right_hip_angle_deg": right_hip_angle,
        "left_knee_angle_deg": left_knee_angle,
        "right_knee_angle_deg": right_knee_angle,
    }
