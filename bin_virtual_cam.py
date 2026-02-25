#!/usr/bin/env python3
"""
Virtual Camera Gaze Correction

Runs gaze correction headlessly and outputs corrected frames to a virtual
camera device, making it available as a source in OBS and video conferencing
apps.

Usage:
    python bin_virtual_cam.py                      # Use dlib backend
    python bin_virtual_cam.py --backend mediapipe  # Use mediapipe backend
    python bin_virtual_cam.py --camera 1           # Use camera device 1
    python bin_virtual_cam.py --passthrough        # Raw camera, no correction (debug)

In OBS:
    Add Source → Video Capture Device → select "OBS Virtual Camera"

Requirements:
    - OBS installed (provides the virtual camera driver on macOS)
    - pyvirtualcam: already in pyproject.toml

Press Ctrl+C to stop.
"""

import argparse
import sys

import cv2
import numpy as np
import pyvirtualcam

from bin_single_window import detect_camera_resolution, select_camera
from displayers.dis_single_window import DisplayConfig, SingleWindowGazeCorrector
from displayers.face_predictor import create_face_predictor


def main():
    parser = argparse.ArgumentParser(
        description="Gaze Correction Virtual Camera — streams corrected video to OBS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
In OBS: Add Source → Video Capture Device → select "OBS Virtual Camera"

Examples:
  %(prog)s                         # dlib backend, interactive camera picker
  %(prog)s --backend mediapipe     # MediaPipe face detection
  %(prog)s --camera 1 --fps 30     # explicit camera and FPS
  %(prog)s --passthrough           # raw camera feed, no correction (use to debug black screen)
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
        help="Path to gaze corrector config file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS for virtual camera output (default: 30)",
    )
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help="Send raw camera frames without gaze correction (useful for debugging black screen)",
    )
    args = parser.parse_args()

    camera_id = args.camera if args.camera is not None else select_camera()
    width, height = detect_camera_resolution(camera_id)

    corrector = None
    if not args.passthrough:
        display_config = DisplayConfig(
            video_size=(width, height),
            face_detect_size=(width // 2, height // 2),
        )
        predictor = create_face_predictor(args.backend)
        corrector = SingleWindowGazeCorrector(
            face_predictor=predictor,
            display_config=display_config,
            camera_id=camera_id,
            config_path=args.config,
        )

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        sys.exit(1)

    mode = "PASSTHROUGH (no correction)" if args.passthrough else f"GAZE CORRECTION ({args.backend})"
    print(f"Starting virtual camera ({width}x{height} @ {args.fps} fps) — {mode}")
    print("In OBS: Add Source → Video Capture Device → 'OBS Virtual Camera'")
    print("Press Ctrl+C to stop.\n")

    try:
        with pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        ) as cam:
            print(f"Virtual camera active: {cam.device}")
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                if args.passthrough:
                    out = frame
                else:
                    out = corrector.process_frame(frame)

                cam.send(out)
                cam.sleep_until_next_frame()

                frame_count += 1
                if frame_count % 60 == 0:
                    brightness = np.mean(out)
                    print(f"  frame {frame_count:5d} | brightness: {brightness:.1f}/255")

    except RuntimeError as e:
        print(f"\nFailed to start virtual camera: {e}")
        print("\nTo fix:")
        print("  1. Open OBS")
        print("  2. Click 'Start Virtual Camera' in the Controls panel")
        print("  3. Approve the System Extension in System Settings → Privacy & Security")
        print("  4. Restart if prompted, then run this script again")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        if corrector is not None:
            corrector.gaze_corrector.close()
        print("Virtual camera stopped.")


if __name__ == "__main__":
    main()
