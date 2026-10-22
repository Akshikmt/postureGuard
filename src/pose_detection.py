"""
Real-Time Pose & Feature Detection Script (PostureGuard)
-------------------------------------------------------
This script captures live video from your webcam, detects human pose
landmarks using Google MediaPipe Pose, extracts geometric features
(angles and normalized distances), and overlays the real-time calculations
on the video feed for posture monitoring.

Educational Concepts:
1. What a Landmark is:
   - Specific anatomical keypoint on the human body with 3D normalized
     coordinates (x, y, z) and a visibility confidence score.

2. What (x, y, z) coordinates represent:
   - x, y: Position normalized to [0.0, 1.0] across width and height.
   - z: Relative depth along the camera axis.

3. Why we calculate angles and distances:
   - Angles (torso inclination, shoulder tilt, hip angle) describe body
     alignment independent of where the person sits in front of the camera.

4. Why normalization is useful:
   - Distances shrink or expand based on camera distance; dividing by torso
     length provides scale invariance.
"""

import sys
from pathlib import Path
import cv2
import mediapipe as mp

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.landmarks import extract_key_landmarks
    from src.features import extract_posture_features
except ModuleNotFoundError:
    from landmarks import extract_key_landmarks
    from features import extract_posture_features


def draw_debug_overlay(frame, features, is_detected: bool):
    """
    Draws a clean semi-transparent HUD overlay on the video frame
    displaying real-time calculated posture metrics.
    """
    h, w, _ = frame.shape

    # Draw semi-transparent background card on the top-left
    overlay = frame.copy()
    box_w, box_h = 320, 160
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (25, 25, 25), -1)
    # Blend with original frame for glassmorphism effect (alpha = 0.65)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    # Border
    cv2.rectangle(frame, (15, 15), (15 + box_w, 15 + box_h), (80, 80, 80), 1)

    if not is_detected or features is None:
        cv2.putText(
            frame,
            "Status: No Person Detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 50, 240),  # Red
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Please sit in front of camera",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )
        return

    # Status header
    is_upper_body = features.get("shoulder_tilt_deg") is not None and features.get("neck_angle_deg") is not None
    status_title = "Status: Upper Body Detected" if is_upper_body else "Status: Pose Detected"
    cv2.putText(
        frame,
        status_title,
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (60, 220, 60),  # Green
        2,
        cv2.LINE_AA
    )

    # Format angle strings safely (handles close-up mode where hips are off-screen)
    shoulder_val = features.get("shoulder_tilt_deg")
    neck_val = features.get("neck_angle_deg")
    shoulder_dist = features.get("shoulder_distance_raw")
    torso_val = features.get("torso_angle_deg")

    shoulder_str = f"{shoulder_val:+.1f} deg" if shoulder_val is not None else "--"
    neck_str = f"{neck_val:.1f} deg" if neck_val is not None else "--"
    dist_str = f"{shoulder_dist:.3f}" if shoulder_dist is not None else "--"
    torso_str = f"{torso_val:.1f} deg" if torso_val is not None else "Off-screen (Close-up)"

    # Display calculated metrics on screen
    lines = [
        (f"Shoulder Tilt: {shoulder_str}", (220, 220, 220)),
        (f"Neck Angle:    {neck_str}", (220, 220, 220)),
        (f"Shoulder Width:{dist_str}", (220, 220, 220)),
        (f"Torso (Spine): {torso_str}", (180, 180, 180)),
    ]

    y_offset = 68
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA
        )
        y_offset += 23


def main():
    # Step 1: Initialize MediaPipe Pose and Drawing Utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Step 2: Open the Default Webcam using OpenCV
    camera_index = 0
    # Open Webcam with robust DirectShow backend on Windows (avoids MSMF/obsensor index errors)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    # Handle camera access failure gracefully
    if not cap.isOpened():
        print("=" * 60)
        print("Error: Could not access the webcam.")
        print("Troubleshooting tips:")
        print(" 1. Ensure no other app (Zoom, Teams, Camera) is using the webcam.")
        print(" 2. Check Windows privacy settings: allow desktop apps to access camera.")
        print(" 3. Verify that your webcam is properly connected.")
        print("=" * 60)
        sys.exit(1)

    print("Camera successfully opened!")
    print("Press 'q' in the display window to exit.")

    # Step 3: Configure MediaPipe Pose Detection Model
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:

        try:
            while cap.isOpened():
                # Read a frame from the webcam
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to grab frame from camera. Exiting loop...")
                    break

                # Step 4: Color Conversion (BGR -> RGB)
                # MediaPipe requires RGB images, while OpenCV reads frames as BGR.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Performance optimization: mark image as non-writeable before inference
                rgb_frame.flags.writeable = False

                # Step 5: Run MediaPipe Pose Detection
                results = pose.process(rgb_frame)

                # Re-enable writeable flag to allow drawing annotations on frame
                rgb_frame.flags.writeable = True

                is_detected = results.pose_landmarks is not None
                features = None

                # Step 6: Process Landmarks & Calculate Geometric Features
                if is_detected:
                    # Draw detected landmark dots and skeleton connecting lines
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # Extract target key landmarks with confidence threshold
                    landmarks = extract_key_landmarks(results.pose_landmarks, min_visibility=0.5)

                    # Calculate posture features (angles, distances, normalizations)
                    features = extract_posture_features(landmarks)

                # Step 7: Draw Debug Overlay displaying real-time angle metrics
                draw_debug_overlay(frame, features, is_detected)

                # Step 8: Display the Processed Video Frame
                cv2.imshow("PostureGuard - Pose Detection", frame)

                # Step 9: Safe Exit on Key Press ('q')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key 'q' pressed. Shutting down...")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Cleaning up...")

        finally:
            # Step 10: Proper Resource Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam released and OpenCV display windows closed properly.")


if __name__ == "__main__":
    main()
