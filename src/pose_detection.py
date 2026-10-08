"""
Real-Time Pose Detection Script (PostureGuard)
----------------------------------------------
This script captures live video from your webcam and detects human pose
landmarks in real time using Google MediaPipe Pose and OpenCV.

Key Concepts Explained:
1. Why OpenCV is used:
   - OpenCV handles hardware-level webcam access, reading video frames as
     numpy arrays, displaying windows, and capturing keyboard inputs (e.g., 'q' to quit).

2. Why color format conversion (BGR -> RGB) is needed:
   - OpenCV captures frames in BGR (Blue-Green-Red) channel order by default.
   - MediaPipe's deep learning models are trained on standard RGB (Red-Green-Blue)
     images. Passing BGR directly would degrade detection accuracy.

3. What MediaPipe Pose does:
   - It is an AI/ML pipeline optimized for real-time inference on CPUs/GPUs
     that identifies human body pose keypoints from an image frame.

4. What pose landmarks are:
   - 33 standardized body points (such as nose, ears, shoulders, elbows,
     wrists, and hips) with normalized coordinates (x, y, z) and visibility scores.

5. What the landmark drawing step does:
   - It uses `mp_drawing.draw_landmarks` to superimpose landmark circles
     and connecting skeleton lines on top of the video frame for visual verification.
"""

import sys
import cv2
import mediapipe as mp


def main():
    # Step 1: Initialize MediaPipe Pose and Drawing Utilities
    # mp.solutions.pose provides the pre-trained pose estimation model.
    mp_pose = mp.solutions.pose

    # mp.solutions.drawing_utils provides helper functions to render landmarks and connections.
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Step 2: Open the Default Webcam using OpenCV
    camera_index = 0
    print(f"Attempting to open camera (index {camera_index})...")
    cap = cv2.VideoCapture(camera_index)

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
    # - min_detection_confidence: Confidence threshold to first detect a person (0.0 to 1.0).
    # - min_tracking_confidence: Confidence threshold to keep tracking the person across frames.
    # - model_complexity: 1 provides a good balance between speed and accuracy.
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

                # Performance optimization: mark image as non-writeable before passing to MediaPipe
                rgb_frame.flags.writeable = False

                # Step 5: Run MediaPipe Pose Detection
                results = pose.process(rgb_frame)

                # Re-enable writeable flag to allow drawing annotations on frame
                rgb_frame.flags.writeable = True

                # Step 6: Draw Pose Landmarks and Skeleton Connections
                # Handle cases where no person is in front of the camera safely.
                if results.pose_landmarks:
                    # Draw detected landmark dots and skeleton connecting lines onto the BGR frame
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    status_text = "Pose Detected: 33 Landmarks"
                    text_color = (0, 255, 0)  # Green
                else:
                    # Gracefully handle frames where no person is detected
                    status_text = "No Person Detected"
                    text_color = (0, 0, 255)  # Red

                # Overlay status text on the frame
                cv2.putText(
                    frame,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    text_color,
                    2,
                    cv2.LINE_AA
                )

                # Step 7: Display the Processed Video Frame
                cv2.imshow("PostureGuard - Pose Detection", frame)

                # Step 8: Safe Exit on Key Press ('q')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key 'q' pressed. Shutting down...")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Cleaning up...")

        finally:
            # Step 9: Proper Resource Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam released and OpenCV display windows closed properly.")


if __name__ == "__main__":
    main()
