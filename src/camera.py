"""
Webcam Feed Test Script
-----------------------
This script tests the camera connection using OpenCV.
It opens the default webcam, displays the live video feed in a window,
and safely releases the camera when 'q' is pressed.
"""

import sys
import cv2


def main():
    # 0 is typically the default built-in or primary connected webcam
    camera_index = 0
    print(f"Attempting to open camera (index {camera_index})...")

    # Initialize video capture with robust DirectShow backend on Windows (avoids MSMF/obsensor index errors)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    # Check if the camera was successfully opened
    if not cap.isOpened():
        print("=" * 60)
        print("Error: Could not access the webcam.")
        print("Troubleshooting tips:")
        print(" 1. Check if another application (Zoom, Teams, Chrome, Camera) is using the camera.")
        print(" 2. Ensure camera privacy settings in Windows allow desktop apps to access the camera.")
        print(" 3. Verify that the camera is properly plugged in.")
        print("=" * 60)
        sys.exit(1)

    print("Camera opened successfully!")
    print("Press 'q' in the camera window to exit.")

    try:
        while True:
            # Read a single frame from the camera feed
            ret, frame = cap.read()

            # If the frame could not be read, exit the loop
            if not ret or frame is None:
                print("Error: Unable to read frame from webcam.")
                break

            # Display the current frame in an OpenCV window
            cv2.imshow("PostureGuard - Camera Feed", frame)

            # Wait 1ms for key event; exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit key 'q' pressed. Closing camera feed...")
                break

    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C in terminal
        print("\nInterrupted by user (Ctrl+C). Closing...")

    finally:
        # Release the webcam hardware resource
        cap.release()
        # Close all OpenCV display windows
        cv2.destroyAllWindows()
        print("Camera released and windows closed properly.")


if __name__ == "__main__":
    main()
