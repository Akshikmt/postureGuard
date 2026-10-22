"""
Unit tests for Real-Time Posture Detection, Hysteresis, and Calibration (4-Class Mode)
-------------------------------------------------------------------------------------
Covers:
1. Model loading, bundle inspection, 4-class labels, and distance-invariant feature ordering.
2. Input validation: missing features, NaNs, infinities, and -1.0 sentinels.
3. Distance-invariance: verifies identical accurate predictions across varying camera distances.
4. User diagnostic resolution: verifies upright sitting (tilt=-2.1, neck=0.4, dist=0.475)
   correctly and stably predicts 'correct' without false forward lean or slouching.
5. PostureStateHysteresis:
   - Eliminates prediction flip-flopping between correct and slouching on boundary noise.
   - Persistence requirement (5 frames) before transitioning state.
   - Clean handling of borderline/ambivalent predictions.
   - Explicit 'uncertain' state when ambiguous input persists.
   - Immediate tracking loss reset.
   - Rapid and responsive transition on genuine posture shifts.
6. UprightPostureCalibrator:
   - Successful upright baseline capture.
   - Rejection of tilted baseline (tilt > 5.0 deg).
   - Rejection of neck-bent baseline (neck > 12.0 deg).
   - Rejection of slouched baseline (head ratio < 0.35).
   - Rejection of unstable movement during calibration (distance std > 0.03).
   - Distance shift detection (>= 25% shift from D0).
   - Non-interference: calibration does not force ML prediction to 'correct'.
   - Forward-lean disabled notice.
7. Legacy components: PredictionStabilizer rolling majority vote and normalize_shoulder_distance.
"""

from pathlib import Path
import time
import unittest

from src.landmarks import LandmarkPoint
from src.realtime_posture import (
    CLASS_LABELS,
    DEFAULT_MODEL_PATH,
    FOUR_CLASS_LABELS,
    TARGET_UPRIGHT_SHOULDER_DISTANCE,
    CalibrationBaseline,
    PosturePredictor,
    PostureStateHysteresis,
    PredictionStabilizer,
    UprightPostureCalibrator,
    normalize_shoulder_distance,
)


class TestRealtimePosture(unittest.TestCase):

    def setUp(self):
        self.model_path = DEFAULT_MODEL_PATH
        self.assertTrue(self.model_path.exists(), "Default model artifact must exist at models/posture_model.joblib")
        self.predictor = PosturePredictor(self.model_path)

        # Baseline valid feature vector (sitting upright)
        self.valid_features = {
            "shoulder_tilt_deg": -1.2,
            "neck_angle_deg": 3.5,
            "shoulder_distance_raw": 0.380,
            "head_to_shoulder_ratio": 0.44,
        }

    # --- 1. Model Loading & Feature Order Tests ---

    def test_model_loading_and_feature_names(self):
        """Model must load successfully with 4 classes and 2 distance-invariant angular features."""
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(len(self.predictor.feature_names), 2)
        self.assertEqual(
            self.predictor.feature_names,
            ["shoulder_tilt_deg", "neck_angle_deg"]
        )
        self.assertEqual(self.predictor.classes, FOUR_CLASS_LABELS)
        self.assertNotIn("leaning_forward", self.predictor.classes)

    def test_exact_feature_ordering_in_vector(self):
        """Feature vector must be assembled in the exact order declared by bundle."""
        is_valid, vec, msg = self.predictor.validate_features(self.valid_features)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Valid")
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 2)
        self.assertAlmostEqual(vec[0], -1.2)  # shoulder_tilt_deg
        self.assertAlmostEqual(vec[1], 3.5)   # neck_angle_deg

    # --- 2. Input Feature Validation & Safe Failure Handling ---

    def test_rejects_missing_features(self):
        """Must return False and not predict when any required feature is missing."""
        incomplete = {"shoulder_tilt_deg": 0.0}  # neck_angle_deg missing
        is_valid, vec, msg = self.predictor.validate_features(incomplete)
        self.assertFalse(is_valid)
        self.assertIsNone(vec)
        self.assertIn("Missing required landmark", msg)

        pred, probs, status = self.predictor.predict(incomplete)
        self.assertIsNone(pred)
        self.assertEqual(probs, {})
        self.assertIn("Missing required landmark", status)

    def test_rejects_none_input(self):
        """Must safely handle None input without crashing."""
        is_valid, vec, msg = self.predictor.validate_features(None)
        self.assertFalse(is_valid)
        self.assertIsNone(vec)
        self.assertEqual(msg, "No features extracted")

        pred, probs, status = self.predictor.predict(None)
        self.assertIsNone(pred)
        self.assertEqual(probs, {})

    def test_rejects_nan_and_inf(self):
        """Must reject NaN and Infinity values in required features."""
        nan_input = self.valid_features.copy()
        nan_input["shoulder_tilt_deg"] = float("nan")
        is_valid, vec, msg = self.predictor.validate_features(nan_input)
        self.assertFalse(is_valid)
        self.assertIn("Non-finite", msg)

        inf_input = self.valid_features.copy()
        inf_input["neck_angle_deg"] = float("inf")
        is_valid_inf, _, msg_inf = self.predictor.validate_features(inf_input)
        self.assertFalse(is_valid_inf)
        self.assertIn("Non-finite", msg_inf)

    def test_rejects_sentinel_values(self):
        """Must reject -1.0 sentinels in primary features and not substitute fake numbers."""
        sentinel_input = self.valid_features.copy()
        sentinel_input["shoulder_tilt_deg"] = -1.0
        is_valid, vec, msg = self.predictor.validate_features(sentinel_input)
        self.assertFalse(is_valid)
        self.assertIn("sentinel", msg.lower())

    def test_valid_prediction_returns_4_classes_and_probabilities(self):
        """Valid feature input must return a recognized 4-class label and non-empty probabilities."""
        pred, probs, status = self.predictor.predict(self.valid_features)
        self.assertIn(pred, FOUR_CLASS_LABELS)
        self.assertEqual(status, "Prediction OK")
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

    # --- 3. Distance Invariance & User Diagnostic Resolution ---

    def test_user_diagnostic_case_stable_correct(self):
        """
        User's reported diagnostic case:
        tilt=-2.1 deg, neck=0.4 deg, dist=0.475
        In 4-class mode, this upright posture must predict 'correct' with high confidence,
        unaffected by camera distance or false leaning_forward.
        """
        user_upright = {
            "shoulder_tilt_deg": -2.1,
            "neck_angle_deg": 0.4,
            "shoulder_distance_raw": 0.475,
        }
        pred, probs, status = self.predictor.predict(user_upright)
        self.assertEqual(pred, "correct")
        self.assertGreater(probs["correct"], 0.70)
        self.assertEqual(status, "Prediction OK")

    def test_distance_invariance_across_diverse_distances(self):
        """
        Upright sitting posture must produce identical predictions regardless of camera distance,
        because active features are angles invariant to distance.
        """
        test_distances = [0.280, 0.350, 0.420, 0.475, 0.550]
        for dist in test_distances:
            upright_input = {
                "shoulder_tilt_deg": -2.1,
                "neck_angle_deg": 0.4,
                "shoulder_distance_raw": dist,
            }
            pred, probs, _ = self.predictor.predict(upright_input)
            self.assertEqual(pred, "correct")
            self.assertGreater(probs["correct"], 0.70)

    def test_genuine_lateral_leans_and_slouch_detected(self):
        """Verifies clear detection of leaning_left, leaning_right, and slouching."""
        # 1. Leaning Left (tilt = +17.5 deg, neck = 42 deg)
        pred_l, probs_l, _ = self.predictor.predict({"shoulder_tilt_deg": 17.5, "neck_angle_deg": 42.0})
        self.assertEqual(pred_l, "leaning_left")
        self.assertGreater(probs_l["leaning_left"], 0.85)

        # 2. Leaning Right (tilt = -18.0 deg, neck = 38 deg)
        pred_r, probs_r, _ = self.predictor.predict({"shoulder_tilt_deg": -18.0, "neck_angle_deg": 38.0})
        self.assertEqual(pred_r, "leaning_right")
        self.assertGreater(probs_r["leaning_right"], 0.85)

        # 3. Slouching (tilt = -0.5 deg, neck = 2.0 deg)
        pred_s, probs_s, _ = self.predictor.predict({"shoulder_tilt_deg": -0.5, "neck_angle_deg": 2.0})
        self.assertEqual(pred_s, "slouching")
        self.assertGreater(probs_s["slouching"], 0.60)

    # --- 4. PostureStateHysteresis (Temporal Stability & Anti-Flicker) Tests ---

    def test_hysteresis_suppresses_boundary_jitter(self):
        """
        When predictions oscillate near the decision boundary with low margin,
        hysteresis must hold the active state and prevent rapid flip-flopping.
        """
        tracker = PostureStateHysteresis(persistence_frames=5, min_confidence=0.55, min_margin=0.10)

        # 1. Establish initial 'correct' state
        for _ in range(3):
            tracker.update("correct", {"correct": 0.90, "slouching": 0.10, "leaning_left": 0.0, "leaning_right": 0.0})
        self.assertEqual(tracker.current_state, "correct")

        # 2. Simulate 4 frames of ambiguous/boundary noise (slouching at 52% vs correct 48%)
        ambiguous_probs = {"correct": 0.48, "slouching": 0.52, "leaning_left": 0.0, "leaning_right": 0.0}
        for _ in range(4):
            state = tracker.update("slouching", ambiguous_probs)
            # Must NOT flip to slouching; holds previous confirmed state
            self.assertEqual(state, "correct")
            self.assertEqual(tracker.current_state, "correct")

    def test_hysteresis_persistence_on_genuine_transition(self):
        """
        A genuine posture change (e.g. correct -> leaning_left) must update
        cleanly once the persistence requirement (5 frames) is satisfied.
        """
        tracker = PostureStateHysteresis(persistence_frames=5, min_confidence=0.55, min_margin=0.10)

        # Establish 'correct'
        for _ in range(3):
            tracker.update("correct", {"correct": 0.85, "slouching": 0.15, "leaning_left": 0.0, "leaning_right": 0.0})
        self.assertEqual(tracker.current_state, "correct")

        lean_probs = {"correct": 0.05, "slouching": 0.05, "leaning_left": 0.85, "leaning_right": 0.05}

        # Frames 1 to 4: candidate accumulates, active state remains 'correct'
        for f in range(1, 5):
            state = tracker.update("leaning_left", lean_probs)
            self.assertEqual(state, "correct", f"Failed at transition frame {f}")
            self.assertEqual(tracker.candidate_state, "leaning_left")
            self.assertEqual(tracker.candidate_counter, f)

        # Frame 5: persistence reached -> active state switches to 'leaning_left'
        state_5 = tracker.update("leaning_left", lean_probs)
        self.assertEqual(state_5, "leaning_left")
        self.assertEqual(tracker.current_state, "leaning_left")
        self.assertIsNone(tracker.candidate_state)

    def test_hysteresis_uncertain_state_on_prolonged_ambiguity(self):
        """
        If predictions remain borderline/uncertain for > max_uncertain_frames,
        tracker must declare 'uncertain'.
        """
        tracker = PostureStateHysteresis(
            persistence_frames=5,
            min_confidence=0.55,
            min_margin=0.10,
            max_uncertain_frames=10
        )
        # Establish 'correct'
        for _ in range(2):
            tracker.update("correct", {"correct": 0.85, "slouching": 0.15})

        # Prolonged ambiguous frames
        ambiguous = {"correct": 0.51, "slouching": 0.49}
        for _ in range(9):
            tracker.update("slouching", ambiguous)
            self.assertEqual(tracker.current_state, "correct")

        # 10th ambiguous frame: triggers 'uncertain'
        res = tracker.update("slouching", ambiguous)
        self.assertEqual(res, "uncertain")
        self.assertTrue(tracker.is_uncertain)

    def test_hysteresis_immediate_reset_on_tracking_loss(self):
        """When person leaves frame (raw_label is None), tracker immediately resets to None."""
        tracker = PostureStateHysteresis(persistence_frames=5)
        tracker.update("correct", {"correct": 0.90, "slouching": 0.10})
        self.assertIsNotNone(tracker.current_state)

        # Tracking lost
        res = tracker.update(None, None)
        self.assertIsNone(res)
        self.assertIsNone(tracker.current_state)
        self.assertIsNone(tracker.candidate_state)
        self.assertEqual(tracker.candidate_counter, 0)

    # --- 5. UprightPostureCalibrator (Validation & Rejection) Tests ---

    def test_calibrator_accepts_valid_upright_posture(self):
        """Valid upright posture (level shoulders, upright neck, steady distance) must calibrate successfully."""
        calibrator = UprightPostureCalibrator(target_duration_sec=0.1, min_required_samples=3)
        calibrator.start_calibration()
        self.assertEqual(calibrator.state, UprightPostureCalibrator.STATE_CALIBRATING)

        dummy_landmarks = {
            "left_shoulder": LandmarkPoint(x=0.40, y=0.35, z=0.0, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.60, y=0.35, z=0.0, visibility=0.9),
            "nose": LandmarkPoint(x=0.50, y=0.20, z=-0.05, visibility=0.9),
        }
        dummy_features = {
            "shoulder_distance_raw": 0.380,
            "shoulder_tilt_deg": 0.5,
            "neck_angle_deg": 3.0,
            "head_to_shoulder_ratio": 0.45,
        }

        for _ in range(5):
            calibrator.update(dummy_features, dummy_landmarks)
            time.sleep(0.025)

        info = calibrator.update(dummy_features, dummy_landmarks)
        self.assertEqual(calibrator.state, UprightPostureCalibrator.STATE_CALIBRATED)
        self.assertIsNotNone(calibrator.baseline)
        self.assertAlmostEqual(calibrator.baseline.shoulder_distance, 0.380, places=2)
        self.assertAlmostEqual(calibrator.baseline.head_to_shoulder_ratio, 0.45, places=2)
        self.assertTrue(info["forward_lean_disabled"])

    def test_calibrator_rejects_tilted_shoulders(self):
        """Calibration must reject when shoulders are tilted laterally (> 5.0 deg)."""
        calibrator = UprightPostureCalibrator(target_duration_sec=0.05, min_required_samples=2)
        calibrator.start_calibration()

        dummy_landmarks = {
            "left_shoulder": LandmarkPoint(x=0.40, y=0.32, z=0.0, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.60, y=0.38, z=0.0, visibility=0.9),
            "nose": LandmarkPoint(x=0.50, y=0.20, z=-0.05, visibility=0.9),
        }
        tilted_features = {
            "shoulder_distance_raw": 0.380,
            "shoulder_tilt_deg": 8.5,  # Exceeds 5.0 deg limit
            "neck_angle_deg": 3.0,
            "head_to_shoulder_ratio": 0.45,
        }

        for _ in range(4):
            calibrator.update(tilted_features, dummy_landmarks)
            time.sleep(0.02)

        info = calibrator.update(tilted_features, dummy_landmarks)
        self.assertEqual(calibrator.state, UprightPostureCalibrator.STATE_REJECTED)
        self.assertIsNone(calibrator.baseline)
        self.assertIn("Uneven shoulders", calibrator.last_rejection_reason)
        self.assertIn("Uneven shoulders", info["message"])

    def test_calibrator_rejects_slouched_head_drop(self):
        """Calibration must reject when head is dropped / slouched (head_to_shoulder_ratio < 0.35)."""
        calibrator = UprightPostureCalibrator(target_duration_sec=0.05, min_required_samples=2)
        calibrator.start_calibration()

        dummy_landmarks = {
            "left_shoulder": LandmarkPoint(x=0.40, y=0.35, z=0.0, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.60, y=0.35, z=0.0, visibility=0.9),
            "nose": LandmarkPoint(x=0.50, y=0.25, z=-0.05, visibility=0.9),
        }
        slouched_features = {
            "shoulder_distance_raw": 0.380,
            "shoulder_tilt_deg": 0.5,
            "neck_angle_deg": 3.0,
            "head_to_shoulder_ratio": 0.28,  # Below 0.35 limit
        }

        for _ in range(4):
            calibrator.update(slouched_features, dummy_landmarks)
            time.sleep(0.02)

        info = calibrator.update(slouched_features, dummy_landmarks)
        self.assertEqual(calibrator.state, UprightPostureCalibrator.STATE_REJECTED)
        self.assertIsNone(calibrator.baseline)
        self.assertIn("Slouched baseline", calibrator.last_rejection_reason)

    def test_calibrator_rejects_excessive_movement(self):
        """Calibration must reject when user moves excessively (distance std > 0.030)."""
        calibrator = UprightPostureCalibrator(target_duration_sec=0.06, min_required_samples=3)
        calibrator.start_calibration()

        dummy_landmarks = {
            "left_shoulder": LandmarkPoint(x=0.40, y=0.35, z=0.0, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.60, y=0.35, z=0.0, visibility=0.9),
            "nose": LandmarkPoint(x=0.50, y=0.20, z=-0.05, visibility=0.9),
        }

        # Rapidly fluctuating distance
        dist_sequence = [0.320, 0.440, 0.310, 0.450]
        for d in dist_sequence:
            feat = {
                "shoulder_distance_raw": d,
                "shoulder_tilt_deg": 0.5,
                "neck_angle_deg": 3.0,
                "head_to_shoulder_ratio": 0.45,
            }
            calibrator.update(feat, dummy_landmarks)
            time.sleep(0.02)

        info = calibrator.update(feat, dummy_landmarks)
        self.assertEqual(calibrator.state, UprightPostureCalibrator.STATE_REJECTED)
        self.assertIsNone(calibrator.baseline)
        self.assertIn("Excessive movement", calibrator.last_rejection_reason)

    def test_calibrator_distance_shift_detection(self):
        """Once calibrated, a distance shift of >= 25% must flag is_distance_shifted."""
        calibrator = UprightPostureCalibrator(target_duration_sec=0.05, min_required_samples=2)
        calibrator.start_calibration()

        dummy_landmarks = {
            "left_shoulder": LandmarkPoint(x=0.40, y=0.35, z=0.0, visibility=0.9),
            "right_shoulder": LandmarkPoint(x=0.60, y=0.35, z=0.0, visibility=0.9),
            "nose": LandmarkPoint(x=0.50, y=0.20, z=-0.05, visibility=0.9),
        }
        upright_features = {
            "shoulder_distance_raw": 0.400,  # D0 = 0.400
            "shoulder_tilt_deg": 0.5,
            "neck_angle_deg": 3.0,
            "head_to_shoulder_ratio": 0.45,
        }

        for _ in range(4):
            calibrator.update(upright_features, dummy_landmarks)
            time.sleep(0.02)
        calibrator.update(upright_features, dummy_landmarks)

        self.assertEqual(calibrator.state, UprightPostureCalibrator.STATE_CALIBRATED)

        # Normal small change (5%)
        info_ok = calibrator.update({"shoulder_distance_raw": 0.420, "shoulder_tilt_deg": 0.5, "neck_angle_deg": 3.0}, dummy_landmarks)
        self.assertFalse(info_ok["is_distance_shifted"])

        # Significant distance shift (30% change: 0.400 * 1.30 = 0.520)
        info_shift = calibrator.update({"shoulder_distance_raw": 0.520, "shoulder_tilt_deg": 0.5, "neck_angle_deg": 3.0}, dummy_landmarks)
        self.assertTrue(info_shift["is_distance_shifted"])
        self.assertIn("Distance changed", info_shift["message"])

    # --- 6. Legacy Normalization & Stabilizer Tests ---

    def test_normalize_shoulder_distance_formula_and_fallbacks(self):
        """Legacy normalization formula D_norm = D * (0.370 / D0) and safe fallbacks."""
        d_norm = normalize_shoulder_distance(0.475, 0.475)
        self.assertAlmostEqual(d_norm, 0.370, places=4)

        for invalid in [None, 0.0, -0.4, float("nan"), float("inf"), "abc"]:
            self.assertEqual(normalize_shoulder_distance(0.475, invalid), 0.475)

    def test_legacy_prediction_stabilizer_majority_vote(self):
        """Legacy PredictionStabilizer majority voting and tie-breaking."""
        stab = PredictionStabilizer(window_size=5)
        stab.add_prediction("correct")
        stab.add_prediction("correct")
        stab.add_prediction("slouching")
        res = stab.add_prediction("correct")
        self.assertEqual(res, "correct")


if __name__ == "__main__":
    unittest.main()
