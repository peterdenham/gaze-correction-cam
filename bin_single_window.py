#!/usr/bin/env python3
"""
Single Window Gaze Correction Application

A simplified gaze correction implementation using a single window.

Usage:
    python bin_single_window.py                      # Use dlib backend
    python bin_single_window.py --backend mediapipe  # Use mediapipe backend
    python bin_single_window.py --camera 1           # Use camera device 1

Controls:
    - 'g': Toggle gaze correction on/off
    - 'c': Toggle calibration mode
    - 'q': Quit
"""

import os
import sys
import cv2
from displayers.dis_single_window import SingleWindowGazeCorrector, DisplayConfig
from displayers.face_predictor import create_face_predictor


def list_cameras(max_id: int = 9) -> list[tuple[int, int, int]]:
    """
    Probe camera device IDs and return available cameras.

    Returns:
        List of (device_id, width, height) for each working camera.
    """
    cameras = []
    consecutive_misses = 0
    # Suppress OpenCV stderr noise when probing non-existent device IDs
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        probe_results = []
        for i in range(max_id + 1):
            cap = cv2.VideoCapture(i)
            opened = cap.isOpened()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
            cap.release()
            probe_results.append((i, opened, w, h))
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull_fd)

    for i, opened, w, h in probe_results:
        if opened:
            cameras.append((i, w, h))
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if cameras and consecutive_misses >= 2:
                break
            if not cameras and i >= 2:
                break
    return cameras


def select_camera() -> int:
    """
    Interactively prompt the user to select a camera from detected devices.

    Returns:
        The chosen camera device ID.
    """
    print("Scanning for available cameras...")
    cameras = list_cameras()

    if not cameras:
        print("No cameras detected. Defaulting to device 0.")
        return 0

    if len(cameras) == 1:
        cam_id, w, h = cameras[0]
        print(f"Found 1 camera: device {cam_id} ({w}x{h}) — using it automatically.")
        return cam_id

    print(f"\nFound {len(cameras)} cameras:")
    for idx, (cam_id, w, h) in enumerate(cameras):
        print(f"  [{idx}] Device {cam_id}  ({w}x{h})")

    while True:
        try:
            choice = input(f"\nSelect camera [0-{len(cameras) - 1}]: ").strip()
            n = int(choice)
            if 0 <= n < len(cameras):
                return cameras[n][0]
            print(f"  Please enter a number between 0 and {len(cameras) - 1}.")
        except (ValueError, EOFError):
            print("  Invalid input. Please enter a number.")


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
    )
    corrector.run()


if __name__ == "__main__":
    main()
