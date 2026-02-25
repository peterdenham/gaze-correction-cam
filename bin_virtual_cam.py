#!/usr/bin/env python3
"""
Virtual Camera Gaze Correction

Runs gaze correction headlessly and outputs corrected frames to a virtual
camera device, making it available as a source in OBS and video conferencing
apps.

Usage:
    python bin_virtual_cam.py                      # Use dlib backend
    python bin_virtual_cam.py --backend mediapipe  # Use mediapipe backend
    python bin_virtual_cam.py --camera 4           # Use camera device 4
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
import threading
import time

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
  %(prog)s --camera 4                  # recommended camera
  %(prog)s --camera 4 --backend mediapipe  # faster face detection on Apple Silicon
  %(prog)s --passthrough               # raw camera feed (debug black screen)
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
        help="Send raw camera frames without gaze correction (debug)",
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

    mode = "PASSTHROUGH" if args.passthrough else f"GAZE CORRECTION ({args.backend})"
    print(f"Starting virtual camera ({width}x{height} @ {args.fps} fps) — {mode}")
    print("In OBS: Add Source → Video Capture Device → 'OBS Virtual Camera'")
    print("Press Ctrl+C to stop.\n")

    # --- Threaded pipeline ---
    # capture_thread  → latest_raw    (always newest camera frame)
    # process_thread  → latest_corrected (always newest corrected frame)
    # main loop       → sends latest_corrected to OBS at target FPS

    latest_raw: list = [None]        # [frame] — shared mutable slot
    latest_corrected: list = [None]  # [frame] — shared mutable slot
    lock_raw = threading.Lock()
    lock_corrected = threading.Lock()
    running = threading.Event()
    running.set()

    proc_fps_counter: list = [0.0]   # [fps] updated by process_thread

    def capture_thread():
        while running.is_set():
            ret, frame = cap.read()
            if ret:
                with lock_raw:
                    latest_raw[0] = frame

    def process_thread():
        frames_done = 0
        t_start = time.monotonic()
        while running.is_set():
            with lock_raw:
                frame = latest_raw[0]
            if frame is None:
                time.sleep(0.001)
                continue

            if args.passthrough:
                out = frame
            else:
                out = corrector.process_frame(frame)

            with lock_corrected:
                latest_corrected[0] = out

            frames_done += 1
            elapsed = time.monotonic() - t_start
            if elapsed >= 2.0:
                proc_fps_counter[0] = frames_done / elapsed
                frames_done = 0
                t_start = time.monotonic()

    t_capture = threading.Thread(target=capture_thread, daemon=True)
    t_process = threading.Thread(target=process_thread, daemon=True)

    try:
        with pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        ) as cam:
            print(f"Virtual camera active: {cam.device}")
            t_capture.start()
            t_process.start()

            out_frame_count = 0
            t_stats = time.monotonic()

            while True:
                with lock_corrected:
                    frame = latest_corrected[0]

                if frame is not None:
                    cam.send(frame)
                    out_frame_count += 1
                    cam.sleep_until_next_frame()
                else:
                    # Processing thread not ready yet — spin until first frame
                    time.sleep(1 / args.fps)

                elapsed = time.monotonic() - t_stats
                if elapsed >= 2.0:
                    out_fps = out_frame_count / elapsed
                    print(
                        f"  proc: {proc_fps_counter[0]:.1f} fps  |"
                        f"  out: {out_fps:.1f} fps  |"
                        f"  brightness: {np.mean(frame):.0f}/255"
                        if frame is not None else ""
                    )
                    out_frame_count = 0
                    t_stats = time.monotonic()

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
        running.clear()
        cap.release()
        if corrector is not None:
            corrector.gaze_corrector.close()
        print("Virtual camera stopped.")


if __name__ == "__main__":
    main()
