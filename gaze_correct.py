#!/usr/bin/env python3
"""
Single Window Gaze Correction Application

A simplified gaze correction implementation using a single window.

Usage:
    python gaze_correct.py                      # Use dlib backend
    python gaze_correct.py --backend mediapipe  # Use mediapipe backend
    python gaze_correct.py --camera 1           # Use camera device 1

Controls:
    - 'g': Toggle gaze correction on/off
    - 'c': Toggle calibration mode
    - 'q': Quit
"""

import sys
import cv2
from displayers.dis_single_window import SingleWindowGazeCorrector, DisplayConfig
from displayers.face_predictor import create_face_predictor
from utils.camera import detect_camera_resolution, list_cameras, select_camera  # noqa: F401 (re-exported for virtual_cam)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Single Window Gaze Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  'g' - Toggle gaze correction on/off
  'c' - Toggle calibration mode
  'q' - Quit the application

Examples:
  %(prog)s                         # Use default dlib backend
  %(prog)s --backend mediapipe     # Use MediaPipe for face detection
  %(prog)s --camera 1              # Use camera device 1
        """,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="dlib",
        choices=["dlib", "mediapipe"],
        help="Face detection backend (default: dlib)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID (omit to select interactively)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./model_managers/gaze_corrector_v1_01.yaml",
        help="Path to gaze corrector config file (default: ./model_managers/gaze_corrector_v1_01.yaml)",
    )
    parser.add_argument(
        "--eye-scale",
        type=float,
        default=0.85,
        dest="eye_scale",
        help="Scale factor for corrected eye patch (default: 0.85). "
             "Reduces the warp model's tendency to make eyes look larger. "
             "Use 1.0 to disable, lower values for more reduction.",
    )
    args = parser.parse_args()

    # Resolve camera: interactive picker if --camera not provided
    camera_id = args.camera if args.camera is not None else select_camera()

    # Detect camera resolution
    video_size = detect_camera_resolution(camera_id)
    
    # Calculate appropriate face detection size (half resolution)
    face_detect_size = (video_size[0] // 2, video_size[1] // 2)

    # Create display config with detected resolution
    display_config = DisplayConfig(
        video_size=video_size,
        face_detect_size=face_detect_size,
    )

    print(f"Video size: {video_size}, Face detection size: {face_detect_size}")

    # Create face predictor based on selected backend
    predictor = create_face_predictor(args.backend)

    # Create and run the corrector
    corrector = SingleWindowGazeCorrector(
        face_predictor=predictor,
        display_config=display_config,
        camera_id=camera_id,
        config_path=args.config,
        eye_scale=args.eye_scale,
    )
    corrector.run()


if __name__ == "__main__":
    main()
