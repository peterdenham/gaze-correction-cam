"""
Camera enumeration and interactive selection utilities.

Shared by gaze_correct.py, virtual_cam.py, and bin_calibrate.py.

Why the preview step exists
----------------------------
On macOS, virtual cameras (OBS, mmhmm, Elgato, Immersed, ...) can cause the
AVFoundation device index reported by pyobjc to differ from the device index
that OpenCV uses internally — they sometimes intercept the enumeration and
insert themselves at different positions in each call site.  Rather than
relying on a name→index mapping that can be silently wrong, we show a live
preview after each selection so the user can visually confirm the camera is
correct before continuing.
"""

import os
import time

import cv2


# ── Camera probing ─────────────────────────────────────────────────────────────

def list_cameras(max_id: int = 9) -> list[tuple[int, int, int]]:
    """
    Probe camera device IDs and return (device_id, width, height) for each
    that opens successfully.
    """
    # Suppress OpenCV's stderr noise when probing non-existent device IDs
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

    cameras: list[tuple[int, int, int]] = []
    consecutive_misses = 0
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


def get_camera_name_map() -> dict[int, str]:
    """
    Return {device_id: name} using AVFoundation (macOS only).

    WARNING: The index returned by AVFoundation's devicesWithMediaType may not
    match OpenCV's device numbering when virtual cameras are installed.  Treat
    these names as hints only — the preview step in select_camera() is the
    reliable confirmation.
    """
    try:
        import AVFoundation
        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(
            AVFoundation.AVMediaTypeVideo
        )
        return {i: d.localizedName() for i, d in enumerate(devices)}
    except Exception:
        return {}


# ── Preview confirmation ───────────────────────────────────────────────────────

def _preview_confirm(camera_id: int, label: str, timeout_s: float = 3.0) -> bool:
    """
    Open camera_id and show a live preview for up to timeout_s seconds.

    Returns:
        True  — user confirmed (SPACE / ENTER) or timeout elapsed
        False — user pressed R to re-pick
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return True  # Can't open for preview; assume OK

    win = f"Camera {camera_id}: {label}  — SPACE/ENTER=confirm   R=re-pick"
    deadline = time.monotonic() + timeout_s
    confirmed = True

    while time.monotonic() < deadline:
        ret, frame = cap.read()
        if not ret:
            break

        remaining = max(0.0, deadline - time.monotonic())
        hint = f"SPACE/ENTER = confirm   R = re-pick   (auto in {remaining:.1f}s)"
        cv2.putText(frame, hint, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)

        key = cv2.waitKey(50) & 0xFF
        if key in (ord(" "), 13):   # SPACE or ENTER
            confirmed = True
            break
        elif key in (ord("r"), ord("R")):
            confirmed = False
            break

    cap.release()
    cv2.destroyWindow(win)
    return confirmed


# ── Interactive picker ────────────────────────────────────────────────────────

def select_camera() -> int:
    """
    Scan for available cameras, present a numbered menu, show a live preview
    for confirmation, and return the chosen device ID.
    """
    print("Scanning for available cameras...")
    cameras = list_cameras()
    names = get_camera_name_map()

    def label(cam_id: int, w: int, h: int) -> str:
        name = names.get(cam_id, f"Device {cam_id}")
        return f"{name}  ({w}×{h})"

    if not cameras:
        print("No cameras detected — defaulting to device 0.")
        return 0

    if len(cameras) == 1:
        cam_id, w, h = cameras[0]
        print(f"Found 1 camera: {label(cam_id, w, h)} — using it automatically.")
        return cam_id

    # Multiple cameras: interactive loop that allows re-picking after preview
    while True:
        print(f"\nFound {len(cameras)} cameras:")
        for idx, (cam_id, w, h) in enumerate(cameras):
            print(f"  [{idx}] {label(cam_id, w, h)}")
        print("  (names are hints — a preview will open to confirm)")

        # Get numeric choice
        chosen_id = None
        while chosen_id is None:
            try:
                n = int(input(f"\nSelect camera [0-{len(cameras) - 1}]: ").strip())
                if 0 <= n < len(cameras):
                    chosen_id = cameras[n][0]
                else:
                    print(f"  Please enter a number between 0 and {len(cameras) - 1}.")
            except (ValueError, EOFError):
                print("  Invalid input.")

        cam_label = label(chosen_id, *next(
            (w, h) for cid, w, h in cameras if cid == chosen_id
        ))
        print(f"  Opening camera {chosen_id} ({cam_label}) for preview...")
        if _preview_confirm(chosen_id, cam_label):
            return chosen_id
        print("  Re-picking...")


def detect_camera_resolution(camera_id: int) -> tuple[int, int]:
    """Return (width, height) of the given camera device."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Warning: could not open camera {camera_id}, using default resolution.")
        return (640, 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Detected camera resolution: {w}×{h}")
    return (w, h)
