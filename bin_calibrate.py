#!/usr/bin/env python3
"""
Guided Camera Geometry Calibration

Solves for accurate focal_length and camera_offset from physical measurements
plus a brief face-capture session.  Run once per screen/camera setup.

Usage:
    python bin_calibrate.py                      # dlib backend, camera 0
    python bin_calibrate.py --backend mediapipe  # MediaPipe face detector
    python bin_calibrate.py --camera 1           # Use camera device 1
"""

import math
import os
import statistics
import sys
import time

import cv2
import numpy as np

# AVFoundation camera name lookup (macOS only — falls back to empty dict)
try:
    import AVFoundation as _AVF

    def _camera_name_map() -> dict[int, str]:
        devices = _AVF.AVCaptureDevice.devicesWithMediaType_(_AVF.AVMediaTypeVideo)
        return {i: d.localizedName() for i, d in enumerate(devices)}
except Exception:
    def _camera_name_map() -> dict[int, str]:
        return {}


def _list_cameras(max_id: int = 9) -> list[tuple[int, int, int]]:
    """Probe camera device IDs and return (device_id, width, height) for each found."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        results = []
        for i in range(max_id + 1):
            cap = cv2.VideoCapture(i)
            opened = cap.isOpened()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
            cap.release()
            results.append((i, opened, w, h))
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull_fd)

    cameras: list[tuple[int, int, int]] = []
    consecutive_misses = 0
    for i, opened, w, h in results:
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


def _select_camera() -> int:
    """Interactively prompt the user to pick a camera. Returns the device ID."""
    print("  Scanning for cameras...")
    cameras = _list_cameras()
    names = _camera_name_map()

    def label(cam_id: int, w: int, h: int) -> str:
        name = names.get(cam_id, f"Device {cam_id}")
        return f"{name}  ({w}×{h})"

    if not cameras:
        print("  No cameras detected — defaulting to device 0.")
        return 0

    if len(cameras) == 1:
        cam_id, w, h = cameras[0]
        print(f"  Found 1 camera: {label(cam_id, w, h)} — using it automatically.")
        return cam_id

    print(f"  Found {len(cameras)} cameras:")
    for idx, (cam_id, w, h) in enumerate(cameras):
        print(f"    [{idx}] {label(cam_id, w, h)}")

    while True:
        try:
            choice = input(f"  Select camera [0-{len(cameras) - 1}]: ").strip()
            n = int(choice)
            if 0 <= n < len(cameras):
                return cameras[n][0]
            print(f"  Enter a number between 0 and {len(cameras) - 1}.")
        except (ValueError, EOFError):
            print("  Invalid input.")


from displayers.face_predictor import (
    EyeExtractionConfig,
    create_face_predictor,
)
from model_managers.gaze_corrector_v1 import CameraUserSetting
from model_managers.user_settings_db import UserSettingsDB


# ── Screen and camera presets ─────────────────────────────────────────────────

def _screen_height_cm(diagonal_in: float, aspect_w: float = 16, aspect_h: float = 9) -> float:
    """Height in cm for a screen of given diagonal and aspect ratio."""
    ratio = aspect_h / math.sqrt(aspect_w ** 2 + aspect_h ** 2)
    return diagonal_in * ratio * 2.54


SCREEN_PRESETS: list[tuple[str, float | None]] = [
    ("13-inch laptop (16:9)",    _screen_height_cm(13)),
    ("14-inch laptop (16:9)",    _screen_height_cm(14)),
    ("15.6-inch laptop (16:9)",  _screen_height_cm(15.6)),
    ("16-inch laptop (16:9)",    _screen_height_cm(16)),
    ("24-inch monitor (16:9)",   _screen_height_cm(24)),
    ("27-inch monitor (16:9)",   _screen_height_cm(27)),
    ("34-inch ultrawide (21:9)", _screen_height_cm(34, 21, 9)),
    ("Custom",                   None),
]

CAMERA_PRESETS: list[tuple[str, float | None]] = [
    ("Laptop built-in (~0.5 cm above screen)",   0.5),
    ("Webcam clipped to monitor (~1.5 cm)",       1.5),
    ("Webcam on stand (~3 cm above monitor top)", 3.0),
    ("Custom",                                    None),
]


# ── Tiny CLI helpers ──────────────────────────────────────────────────────────

def _pick(prompt: str, options: list[tuple]) -> float | None:
    """Print a numbered menu and return the numeric value (index 1) of the chosen item."""
    print(f"\n{prompt}")
    for i, (label, val) in enumerate(options):
        suffix = f"  (~{val:.1f} cm)" if val is not None else ""
        print(f"  [{i}] {label}{suffix}")
    while True:
        try:
            n = int(input(f"Select [0-{len(options) - 1}]: ").strip())
            if 0 <= n < len(options):
                return options[n][1]
            print(f"  Please enter a number between 0 and {len(options) - 1}.")
        except (ValueError, EOFError):
            print("  Invalid input.")


def _ask_float(prompt: str, default: float, lo: float, hi: float) -> float:
    """Prompt for a float with a default and range validation."""
    while True:
        try:
            raw = input(f"{prompt} [{default}]: ").strip()
            val = float(raw) if raw else default
            if lo <= val <= hi:
                return val
            print(f"  Value must be between {lo} and {hi}.")
        except (ValueError, EOFError):
            print("  Invalid input.")


# ── Face capture ──────────────────────────────────────────────────────────────

def _capture_ipd_pixels(
    camera_id: int,
    backend: str,
    duration_s: float = 5.0,
) -> float | None:
    """
    Open camera, run face detection for `duration_s` seconds, return median
    inter-pupillary distance in pixels.  Returns None on failure.
    """
    # Suppress OpenCV stderr noise on open
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        cap = cv2.VideoCapture(camera_id)
        opened = cap.isOpened()
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        os.close(devnull_fd)

    if not opened:
        print(f"  Error: cannot open camera {camera_id}.")
        return None

    predictor = create_face_predictor(backend)
    config = EyeExtractionConfig()

    samples: list[float] = []
    deadline = time.monotonic() + duration_s
    bar_total = 30

    print(f"  Look directly at the camera lens. Hold still for {duration_s:.0f} s...")
    print("  Press 'q' to cancel.\n")

    while time.monotonic() < deadline:
        ret, frame = cap.read()
        if not ret:
            break

        remaining = max(0.0, deadline - time.monotonic())
        filled = int(bar_total * (1.0 - remaining / duration_s))
        bar = "█" * filled + "░" * (bar_total - filled)

        # Detect face and measure IPD
        face_data_list = predictor.list_eye_data(frame, config)
        detected = False
        for face in face_data_list:
            if face.left_eye is None or face.right_eye is None:
                continue
            le = face.left_eye.center
            re = face.right_eye.center
            ipd_px = math.sqrt((le[0] - re[0]) ** 2 + (le[1] - re[1]) ** 2)
            if ipd_px > 10:
                samples.append(ipd_px)
            # Draw eye dots
            cv2.circle(frame, (int(le[0]), int(le[1])), 5, (50, 200, 50), -1)
            cv2.circle(frame, (int(re[0]), int(re[1])), 5, (50, 200, 50), -1)
            detected = True

        status_color = (0, 220, 0) if detected else (0, 100, 220)
        label = f"[{bar}] {remaining:.1f}s  samples={len(samples)}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.imshow("Calibration — look at camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("  Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < 5:
        print(f"  Only {len(samples)} valid detections — try again in better lighting.")
        return None

    median_ipd = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(f"  Captured {len(samples)} samples.  Median IPD: {median_ipd:.1f} px  (±{stdev:.1f})")
    return median_ipd


# ── Calibration math ──────────────────────────────────────────────────────────

def compute_settings(
    screen_height_cm: float,
    cam_above_top_cm: float,
    ipd_mm: float,
    distance_cm: float,
    measured_ipd_px: float,
) -> CameraUserSetting:
    """
    Compute calibrated camera settings.

    focal_length (pixels):
        From the pinhole camera model's similar-triangle relationship:
            f = ipd_px * distance_cm / ipd_cm
        where ipd_px was measured from the captured frame and distance_cm is
        the user's sitting distance from the screen.

    camera_offset[1] (cm, signed, screen-center coordinates):
        The camera sits above the screen.  In estimate_gaze_angle the
        positive-Y axis points DOWN from screen centre toward the user,
        so the camera's Y offset from screen centre is negative:
            cam_y = -(screen_height_cm / 2 + cam_above_top_cm)
        Example: 13" screen (16.3 cm) + 0.5 cm above → cam_y ≈ -8.7 cm
        vs. the hard-coded default of -21 cm, which is far too large for a
        laptop and causes significant over-correction.
    """
    ipd_cm = ipd_mm / 10.0
    focal_length = measured_ipd_px * distance_cm / ipd_cm
    cam_y = -(screen_height_cm / 2.0 + cam_above_top_cm)

    return CameraUserSetting(
        focal_length=round(focal_length, 1),
        ipd=round(ipd_cm, 2),
        camera_offset=(0.0, round(cam_y, 2), -1.0),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_offset(offset: tuple[float, float, float]) -> str:
    return f"({offset[0]:.1f}, {offset[1]:.2f}, {offset[2]:.1f}) cm"


def _load_existing(db_path: str, setting_name: str) -> CameraUserSetting:
    """Load existing settings for before/after comparison. Returns defaults on any error."""
    try:
        db = UserSettingsDB(db_path)
        saved = db.get_setting(setting_name)
        if saved:
            return CameraUserSetting.from_dict(saved)
    except Exception:
        pass
    return CameraUserSetting()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Guided camera geometry calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Solves for focal_length and camera_offset from physical measurements.
Run once when setting up on a new screen or changing your camera position.
""",
    )
    parser.add_argument(
        "--backend", default="dlib", choices=["dlib", "mediapipe"],
        help="Face detection backend (default: dlib)",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device ID (omit to select interactively)",
    )
    parser.add_argument(
        "--db", default="./user_settings.db",
        help="Path to settings database (default: ./user_settings.db)",
    )
    parser.add_argument(
        "--setting", default="camera_default",
        help="Setting name to save (default: camera_default)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Face-capture duration in seconds (default: 5)",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Gaze Correction — Camera Geometry Calibration")
    print("=" * 60)
    print()
    print("This wizard measures your screen and camera geometry once")
    print("so the gaze correction can compute accurate redirect angles.")
    print()

    # ── Step 1: Screen height ─────────────────────────────────────────────────
    print("── Step 1 of 4: Screen size ─────────────────────────────────")
    screen_height = _pick("Select your screen:", SCREEN_PRESETS)
    if screen_height is None:
        screen_height = _ask_float(
            "Screen height in cm (measured edge to edge)", default=18.0, lo=8.0, hi=100.0
        )
    print(f"  → Screen height: {screen_height:.1f} cm")

    # ── Step 2: Camera position ───────────────────────────────────────────────
    print()
    print("── Step 2 of 4: Camera position ─────────────────────────────")
    print("How far is the camera lens above the TOP EDGE of the screen?")
    cam_above = _pick("Select camera position:", CAMERA_PRESETS)
    if cam_above is None:
        cam_above = _ask_float(
            "Distance from top screen edge to camera lens (cm)", default=1.0, lo=0.0, hi=30.0
        )
    print(f"  → Camera above top edge: {cam_above:.1f} cm")

    # ── Step 3: Physical measurements ────────────────────────────────────────
    print()
    print("── Step 3 of 4: Your measurements ───────────────────────────")
    print("Inter-pupillary distance (IPD) = distance between your pupils.")
    print("Average adult: 63 mm  (typical range 54–72 mm).")
    print("Tip: your optometrist has this, or measure with a ruler and mirror.")
    print()
    ipd_mm = _ask_float("Your IPD in mm", default=63.0, lo=45.0, hi=80.0)
    print()
    print("How far do you normally sit from the screen while working?")
    distance_cm = _ask_float("Sitting distance in cm", default=60.0, lo=20.0, hi=200.0)

    # ── Step 4: Face capture ─────────────────────────────────────────────────
    print()
    print("── Step 4 of 4: Face capture ────────────────────────────────")
    camera_id = args.camera if args.camera is not None else _select_camera()
    print(f"  → Using camera device {camera_id}")
    print()
    print("Sit at your normal working position.")
    print("When the camera window opens, look directly at the camera lens.")
    input("\nPress Enter to start face capture...")
    print()

    measured_ipd_px = _capture_ipd_pixels(camera_id, args.backend, args.duration)
    if measured_ipd_px is None:
        print("\nCalibration failed — could not obtain IPD measurement.")
        sys.exit(1)

    # ── Compute new settings ─────────────────────────────────────────────────
    new_settings = compute_settings(
        screen_height_cm=screen_height,
        cam_above_top_cm=cam_above,
        ipd_mm=ipd_mm,
        distance_cm=distance_cm,
        measured_ipd_px=measured_ipd_px,
    )
    old_settings = _load_existing(args.db, args.setting)

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  focal_length   : {new_settings.focal_length:>7.1f} px   (was {old_settings.focal_length:.1f})")
    print(f"  IPD            : {new_settings.ipd:>7.2f} cm   (was {old_settings.ipd:.2f})")
    print(f"  camera_offset  :  {_fmt_offset(new_settings.camera_offset)}")
    print(f"    (was          :  {_fmt_offset(old_settings.camera_offset)})")
    print()
    cam_y_change = new_settings.camera_offset[1] - old_settings.camera_offset[1]
    if abs(cam_y_change) > 1.0:
        direction = "smaller" if cam_y_change > 0 else "larger"
        print(f"  Note: camera_offset[1] shifted by {cam_y_change:+.1f} cm ({direction}).")
        print("  This is the most impactful parameter — expect noticeably")
        print("  different correction with the new value.")
        print()

    # ── Save ──────────────────────────────────────────────────────────────────
    confirm = input("Save these settings? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        db = UserSettingsDB(args.db)
        db.save_setting(args.setting, new_settings.to_dict())
        print(f"\n✓  Saved to '{args.db}'  (setting: '{args.setting}').")
        print("   Run bin_single_window.py (or bin_virtual_cam.py) to use the")
        print("   new calibration.")
    else:
        print("  Settings not saved.")


if __name__ == "__main__":
    main()
