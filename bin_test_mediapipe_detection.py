#!/usr/bin/env python3
"""
MediaPipe Face Detection Test

Test application to visualize MediaPipe face landmarks detection.

Usage:
    python bin_test_mediapipe_detection.py             # Use default camera
    python bin_test_mediapipe_detection.py --camera 1  # Use camera device 1

Controls:
    - 'b': Toggle background on/off
    - 'q': Quit
"""

from time import time
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


def detect_camera_resolution(camera_id: int) -> tuple[int, int]:
    """
    Detect the actual resolution of the specified camera.
    
    Args:
        camera_id: Camera device ID
        
    Returns:
        Tuple of (width, height) in pixels
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {camera_id}, using default resolution")
        return (640, 480)
    
    # Get the actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Detected camera resolution: {width}x{height}")
    return (width, height)


def run_face_detection(camera_id: int, model_path: str = './models/face_landmarker.task'):
    """
    Run MediaPipe face detection and visualization.
    
    Args:
        camera_id: Camera device ID
        model_path: Path to MediaPipe face landmarker model
    """
    # Detect camera resolution
    video_size = detect_camera_resolution(camera_id)
    
    # Setup MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )

    print(f"Starting camera {camera_id}...")
    print("Press 'b' to toggle background, 'q' to quit")
    
    # State
    show_background = True

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
        start_time = time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                break

            # Calculate the timestamp (in milliseconds) from the start
            # VIDEO mode requires a strictly increasing timestamp
            frame_timestamp_ms = int((time() - start_time) * 1000)

            # Process frame with MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            face_landmarks_list = face_landmarker_result.face_landmarks
            
            # Use original frame or black background based on toggle
            if show_background:
                annotated_image = np.copy(frame)
            else:
                annotated_image = np.zeros_like(frame)

            # Loop through the detected faces to visualize
            for idx in range(len(face_landmarks_list)):
                face_landmarks = face_landmarks_list[idx]

                # Draw face mesh tesselation
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
                
                # Draw face contours
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles
                    .get_default_face_mesh_contours_style()
                )
                
                # Draw left iris
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles
                    .get_default_face_mesh_iris_connections_style()
                )
                
                # Draw right iris
                drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles
                    .get_default_face_mesh_iris_connections_style()
                )

            cv2.imshow('MediaPipe Face Landmarker', annotated_image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                show_background = not show_background
                status = "ON" if show_background else "OFF"
                print(f"Background: {status}")

        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MediaPipe Face Detection Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  'b' - Toggle background on/off
  'q' - Quit the application

Examples:
  %(prog)s                  # Use default camera
  %(prog)s --camera 1       # Use camera device 1
        """,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/face_landmarker.task",
        help="Path to MediaPipe face landmarker model (default: ./models/face_landmarker.task)",
    )
    args = parser.parse_args()

    run_face_detection(args.camera, args.model)


if __name__ == "__main__":
    main()
