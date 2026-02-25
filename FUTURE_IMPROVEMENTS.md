# Future Improvements

Improvements identified during code review (2026-02-25) and architectural analysis.
Ordered roughly by impact. The two highest-impact items (#1 and #2) have already been shipped.

---

## Already Shipped

- **Temporal smoothing (EMA)** — `8d2fe55` — smoothed correction angles, removed int() truncation
- **Feathered eye blending** — `bcd1d9d` — Gaussian elliptical mask replaces hard pixel_cut paste
- **Eye bbox clamping** — `39465f6` — prevents NumPy index-wrap on faces near frame edges
- **Delete `utils/config.py`** — `39465f6` — removed `type=eval` code injection vector

---

## Remaining — Code Review Findings

### Medium Priority

**#3 — Silent exception swallow in `process_frame`**
`displayers/dis_single_window.py:341–343`
`except Exception as e` catches everything including OOM and TF errors, then silently continues
with the uncorrected frame. Should catch specific recoverable exceptions only, or at minimum
log the full traceback so failures are diagnosable.

**#4 — Division by near-zero in angle calculation**
`model_managers/gaze_corrector_v1.py:423`
If `ipd_pixels` is very small (tiny or distant face), `eye_z` grows very large and the angle
calculation can produce extreme values. Add clamping: `ipd_pixels = max(ipd_pixels, 1.0)` and
clamp the final angles to a sane range (e.g. ±30°).

**#6 — TOCTOU race in `UserSettingsDB.save_setting`**
`model_managers/user_settings_db.py:76–94`
SELECT then UPDATE/INSERT is not atomic. Replace with a single `INSERT OR REPLACE` (upsert).
Doesn't matter in normal single-process use but would crash if two processes shared the DB.

**#7 — SQLite write on every calibration keypress**
`model_managers/gaze_corrector_v1.py:295, 313, 349`
`adjust_camera_offset()` and `adjust_focal_length()` call `save_camera_settings()` on every
invocation, which opens and closes a DB connection. With key-repeat this is ~30 writes/second.
Save on calibration mode exit or app quit instead.

**#10 — Arrow key codes non-functional on macOS**
`displayers/dis_single_window.py:81–89`
`KEY_UP/DOWN/LEFT/RIGHT = 0/1/2/3` are placeholder values. On macOS, `cv2.waitKey() & 0xFF`
returns 0xFF (255) for all arrow keys — they can't be distinguished this way. Use
`cv2.waitKey(1)` without the `& 0xFF` mask and map the full multi-byte keycodes, or replace
arrow key calibration with number-key adjustments.

### Lower Priority

**#8 — Unvectorized anchor map loop**
`displayers/face_predictor.py:241–260` (same in mediapipe backend)
6 landmarks × `np.expand_dims + np.tile + np.concatenate` per eye per frame in a Python loop.
Replace with a single vectorized NumPy operation using broadcasting across the landmark axis.
Measurable but not dramatic improvement (~1ms/frame saved).

**#9 — Unnecessary frame copy in single-window mode**
`displayers/dis_single_window.py:329`
`display_frame = frame.copy()` — the frame is freshly read from `cap.read()` and not shared
with any other thread in the single-window path. The copy is only needed in `bin_virtual_cam.py`
where `process_thread` holds the raw frame slot. Can remove the copy in `run()`.

---

## Architectural Improvements (from rebuild analysis)

**A — Single batched TF inference for both eyes**
`model_managers/gaze_corrector_v1.py:100–153`
Two separate TF1.x Sessions (one per eye) means two `sess.run()` dispatches per frame.
Rebuild with a single graph accepting a batch of `[left, right]` eye images to halve overhead.

**B — TF Lite / CoreML conversion for Apple Silicon**
TF1.x compat mode has no Metal GPU delegation — the warping model runs on CPU.
The DeepWarp network is small (48×64 input, ~5 conv layers); exporting to TF Lite or CoreML
would enable GPU/ANE acceleration on M-series chips. Estimated 3–5× inference speedup.
Steps: export checkpoint → SavedModel → `tf.lite.TFLiteConverter` → validate output matches.

**C — Face detection every N frames + optical flow tracking**
`displayers/face_predictor.py:131`
dlib HOG runs full-frame every frame. Standard approach: run detection every 3–5 frames,
propagate landmarks between detections with Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`).
Expected ~2–3× speedup on the detection step.

**D — Guided one-time camera geometry calibration**
`model_managers/gaze_corrector_v1.py:375`
Default `focal_length=650`, `ipd=6.3cm`, `camera_offset=(0,-21,-1)` are guesses and wrong for
most setups. The math to solve for actual values exists in `estimate_gaze_angle`. A guided
procedure (stare at 4 screen corners, capture frames) could solve for `f` and `camera_offset`
from the collected eye position data.
