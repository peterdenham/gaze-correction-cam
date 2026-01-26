#!/usr/bin/env python3
"""
Single Window Gaze Correction Application

A simplified gaze correction implementation using a single window.

Usage:
    python bin_single_window.py                    # Use dlib backend
    python bin_single_window.py --backend mediapipe  # Use mediapipe backend
    python bin_single_window.py --camera 1         # Use camera device 1

Controls:
    - 'g': Toggle gaze correction on/off
    - 'q': Quit
"""

from displayers.single_window import SingleWindowGazeCorrector
from displayers.face_predictor import create_face_predictor


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Single Window Gaze Correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  'g' - Toggle gaze correction on/off
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
        default=0,
        help="Camera device ID (default: 0)",
    )
    args = parser.parse_args()

    # Create face predictor based on selected backend
    predictor = create_face_predictor(args.backend)

    # Create and run the corrector
    corrector = SingleWindowGazeCorrector(
        face_predictor=predictor,
        camera_id=args.camera,
    )
    corrector.run()


if __name__ == "__main__":
    main()
