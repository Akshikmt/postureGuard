"""
Unit tests for landmark extraction and geometric feature calculations.
"""

import unittest
from src.landmarks import LandmarkPoint, extract_key_landmarks, get_midpoint
from src.features import (
    calculate_3point_angle,
    calculate_shoulder_tilt,
    calculate_torso_angle,
    calculate_neck_angle,
    calculate_ear_distance,
    calculate_shoulder_to_ear_ratio,
    calculate_head_to_shoulder_ratio,
    calculate_head_z_distance,
    extract_posture_features,
    euclidean_distance_2d,
)


class TestLandmarksAndFeatures(unittest.TestCase):

    def setUp(self):
        # Simulated upright sitting person (normalized coordinates)
        # Head at top center
        self.nose = LandmarkPoint(x=0.50, y=0.20, z=0.0, visibility=0.95)
        self.left_ear = LandmarkPoint(x=0.48, y=0.19, z=0.0, visibility=0.90)
        self.right_ear = LandmarkPoint(x=0.52, y=0.19, z=0.0, visibility=0.90)

        # Shoulders level horizontally
        self.left_shoulder = LandmarkPoint(x=0.40, y=0.35, z=0.0, visibility=0.95)
        self.right_shoulder = LandmarkPoint(x=0.60, y=0.35, z=0.0, visibility=0.95)

        # Hips level horizontally below shoulders
        self.left_hip = LandmarkPoint(x=0.42, y=0.70, z=0.0, visibility=0.90)
        self.right_hip = LandmarkPoint(x=0.58, y=0.70, z=0.0, visibility=0.90)

        # Knees forward / down
        self.left_knee = LandmarkPoint(x=0.42, y=0.90, z=-0.2, visibility=0.85)
        self.right_knee = LandmarkPoint(x=0.58, y=0.90, z=-0.2, visibility=0.85)

        # Ankles
        self.left_ankle = LandmarkPoint(x=0.42, y=1.00, z=-0.2, visibility=0.80)
        self.right_ankle = LandmarkPoint(x=0.58, y=1.00, z=-0.2, visibility=0.80)

        self.upright_landmarks = {
            "nose": self.nose,
            "left_ear": self.left_ear,
            "right_ear": self.right_ear,
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_hip": self.left_hip,
            "right_hip": self.right_hip,
            "left_knee": self.left_knee,
            "right_knee": self.right_knee,
            "left_ankle": self.left_ankle,
            "right_ankle": self.right_ankle,
        }

    def test_upright_posture_angles(self):
        """In an upright posture, torso angle and shoulder tilt should be near 0 degrees."""
        feats = extract_posture_features(self.upright_landmarks)

        # Shoulder tilt should be 0.0 deg (both shoulders at y = 0.35)
        self.assertAlmostEqual(feats["shoulder_tilt_deg"], 0.0, places=1)

        # Torso angle should be 0.0 deg (mid-shoulder and mid-hip both have x = 0.50)
        self.assertAlmostEqual(feats["torso_angle_deg"], 0.0, places=1)

        # Neck angle should be 0.0 deg (nose and mid-shoulder both at x = 0.50)
        self.assertAlmostEqual(feats["neck_angle_deg"], 0.0, places=1)

    def test_shoulder_tilt_when_leaning_side(self):
        """When leaning to the right/left, shoulder tilt and torso angle change significantly."""
        # Person leans to one side: right shoulder dips down (larger y)
        leaning_landmarks = self.upright_landmarks.copy()
        leaning_landmarks["left_shoulder"] = LandmarkPoint(x=0.40, y=0.30, z=0.0, visibility=0.95)
        leaning_landmarks["right_shoulder"] = LandmarkPoint(x=0.60, y=0.40, z=0.0, visibility=0.95)

        feats = extract_posture_features(leaning_landmarks)
        # Shoulder tilt should now be distinctly non-zero
        self.assertNotEqual(feats["shoulder_tilt_deg"], 0.0)
        self.assertGreater(abs(feats["shoulder_tilt_deg"]), 10.0)

    def test_torso_angle_when_leaning_torso(self):
        """When torso shifts laterally, torso inclination angle should increase."""
        leaning_torso = self.upright_landmarks.copy()
        # Shoulders shift rightwards relative to hips
        leaning_torso["left_shoulder"] = LandmarkPoint(x=0.55, y=0.35, z=0.0, visibility=0.95)
        leaning_torso["right_shoulder"] = LandmarkPoint(x=0.75, y=0.35, z=0.0, visibility=0.95)

        feats = extract_posture_features(leaning_torso)
        # Torso angle should be substantially > 0
        self.assertGreater(feats["torso_angle_deg"], 15.0)

    def test_scale_normalization_invariance(self):
        """Test that normalized shoulder distance remains identical when person moves closer to camera."""
        feats_far = extract_posture_features(self.upright_landmarks)

        # Person moves 2x closer: coordinates scale by 2 around center
        scaled_landmarks = {}
        for name, pt in self.upright_landmarks.items():
            if pt is not None:
                scaled_landmarks[name] = LandmarkPoint(
                    x=0.50 + (pt.x - 0.50) * 2.0,
                    y=0.50 + (pt.y - 0.50) * 2.0,
                    z=pt.z,
                    visibility=pt.visibility
                )
            else:
                scaled_landmarks[name] = None

        feats_close = extract_posture_features(scaled_landmarks)

        # Raw distance doubled
        self.assertAlmostEqual(feats_close["shoulder_distance_raw"], feats_far["shoulder_distance_raw"] * 2.0, places=2)
        # BUT normalized distance relative to torso length remains invariant!
        self.assertAlmostEqual(feats_close["shoulder_distance_norm"], feats_far["shoulder_distance_norm"], places=2)

    def test_safe_handling_of_missing_landmarks(self):
        """Test that missing or low confidence landmarks do not cause exceptions."""
        empty_landmarks = {k: None for k in self.upright_landmarks}
        feats = extract_posture_features(empty_landmarks)

        # All values should gracefully be None, zero crashes
        for key, val in feats.items():
            self.assertIsNone(val, f"Expected None for missing feature {key}")

    def test_3point_joint_angles(self):
        """Test 90 degree right angle calculation."""
        # 3 points forming a 90 degree angle at vertex b
        a = LandmarkPoint(x=0.0, y=1.0, z=0.0, visibility=1.0)
        b = LandmarkPoint(x=0.0, y=0.0, z=0.0, visibility=1.0)
        c = LandmarkPoint(x=1.0, y=0.0, z=0.0, visibility=1.0)

        angle = calculate_3point_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_ear_distance_and_shoulder_to_ear_ratio(self):
        """Test calculation of ear distance and shoulder-to-ear ratio."""
        feats = extract_posture_features(self.upright_landmarks)
        # In upright_landmarks: shoulder width = 0.20, ear width = 0.04
        # Ratio = 0.20 / 0.04 = 5.0
        self.assertAlmostEqual(feats["ear_distance_raw"], 0.04, places=3)
        self.assertAlmostEqual(feats["shoulder_to_ear_ratio"], 5.0, places=2)

    def test_shoulder_to_ear_ratio_scale_invariance(self):
        """Test that shoulder_to_ear_ratio remains invariant when person moves closer to camera."""
        feats_far = extract_posture_features(self.upright_landmarks)

        # Scale by 2x closer to camera
        scaled_landmarks = {}
        for name, pt in self.upright_landmarks.items():
            if pt is not None:
                scaled_landmarks[name] = LandmarkPoint(
                    x=0.50 + (pt.x - 0.50) * 2.0,
                    y=0.50 + (pt.y - 0.50) * 2.0,
                    z=pt.z,
                    visibility=pt.visibility
                )
            else:
                scaled_landmarks[name] = None

        feats_close = extract_posture_features(scaled_landmarks)

        # Raw distances double
        self.assertAlmostEqual(feats_close["shoulder_distance_raw"], feats_far["shoulder_distance_raw"] * 2.0, places=2)
        self.assertAlmostEqual(feats_close["ear_distance_raw"], feats_far["ear_distance_raw"] * 2.0, places=2)
        # BUT shoulder_to_ear_ratio remains identical (scale-invariant)!
        self.assertAlmostEqual(feats_close["shoulder_to_ear_ratio"], feats_far["shoulder_to_ear_ratio"], places=2)

    def test_shoulder_to_ear_ratio_missing_and_zero_division(self):
        """Test that missing ears or zero ear distance returns None gracefully."""
        self.assertIsNone(calculate_shoulder_to_ear_ratio(self.left_shoulder, self.right_shoulder, None, self.right_ear))
        self.assertIsNone(calculate_shoulder_to_ear_ratio(None, self.right_shoulder, self.left_ear, self.right_ear))

        # Both ears at exact same point (zero ear distance)
        same_ear = LandmarkPoint(x=0.5, y=0.5, z=0.0, visibility=1.0)
        self.assertIsNone(calculate_shoulder_to_ear_ratio(self.left_shoulder, self.right_shoulder, same_ear, same_ear))

    def test_head_to_shoulder_ratio(self):
        """Test vertical head elevation to shoulder width ratio calculation."""
        # Upright: mid_shoulder y = 0.35, nose y = 0.20 -> elevation = 0.15
        # shoulder width = 0.60 - 0.40 = 0.20 -> ratio = 0.15 / 0.20 = 0.75
        ratio = calculate_head_to_shoulder_ratio(self.nose, self.left_shoulder, self.right_shoulder)
        self.assertAlmostEqual(ratio, 0.75, places=2)

        # Slouched: nose drops down to y = 0.29 (elevation = 0.06) -> ratio = 0.06 / 0.20 = 0.30
        slouched_nose = LandmarkPoint(x=0.50, y=0.29, z=0.0, visibility=0.95)
        slouched_ratio = calculate_head_to_shoulder_ratio(slouched_nose, self.left_shoulder, self.right_shoulder)
        self.assertAlmostEqual(slouched_ratio, 0.30, places=2)
        self.assertLess(slouched_ratio, ratio)

        # Scale invariance across camera distance
        # Person sitting closer: coordinates scale by 2.0x from center (0.5, 0.5)
        sh_l_close = LandmarkPoint(x=0.30, y=0.20, z=0.0, visibility=0.95)
        sh_r_close = LandmarkPoint(x=0.70, y=0.20, z=0.0, visibility=0.95)
        nose_close = LandmarkPoint(x=0.50, y=-0.10, z=0.0, visibility=0.95)
        # width = 0.40, mid_sh.y = 0.20, nose.y = -0.10 -> elevation = 0.30 -> ratio = 0.30 / 0.40 = 0.75
        close_ratio = calculate_head_to_shoulder_ratio(nose_close, sh_l_close, sh_r_close)
        self.assertAlmostEqual(close_ratio, 0.75, places=2)

        # Missing landmarks
        self.assertIsNone(calculate_head_to_shoulder_ratio(None, self.left_shoulder, self.right_shoulder))
        self.assertIsNone(calculate_head_to_shoulder_ratio(self.nose, None, self.right_shoulder))

    def test_head_z_distance(self):
        """Test MediaPipe relative depth measurement between nose and mid-shoulder."""
        mid_sh = get_midpoint(self.left_shoulder, self.right_shoulder)
        # Normal head aligned with shoulders (dz = 0.0)
        self.assertAlmostEqual(calculate_head_z_distance(self.nose, mid_sh), 0.0, places=2)

        # Head leaning forward in z-axis (dz = -0.12)
        fwd_nose = LandmarkPoint(x=0.50, y=0.20, z=-0.12, visibility=0.95)
        self.assertAlmostEqual(calculate_head_z_distance(fwd_nose, mid_sh), -0.12, places=2)

        # Missing
        self.assertIsNone(calculate_head_z_distance(None, mid_sh))
        self.assertIsNone(calculate_head_z_distance(self.nose, None))


if __name__ == "__main__":
    unittest.main()


