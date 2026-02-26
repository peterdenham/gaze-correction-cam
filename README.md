# Gaze Correction Camera

## Overview

This project implements a gaze correction system for video communication that uses computer vision and deep learning techniques to adjust eye gaze direction in real-time, providing a more natural eye contact experience during video calls. ([study more](./docs/orignal_doc.md))

## Demo

<!--
Source - Adapted from Stack Overflow
Retrieved 2026-01-27
-->

<div style="position: relative; display: inline-block;">
  <!-- Video Thumbnail -->
  <a href="https://www.youtube.com/watch?v=tOobANsNzOQ" target="_blank">
    <img src="https://img.youtube.com/vi/tOobANsNzOQ/0.jpg" style="width: 320px; display: block;">
  </a>
</div>

## Prerequisites

Environment:

```text
ProductName:            macOS
ProductVersion:         15.2
BuildVersion:           24C101
```

The following dependencies are required to run this application:

- [Python 3.12+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/) for dependency management
- [CMake](https://cmake.org/download/) (required for building dlib)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) (required for certain dependencies)

## Installation

1. Install system dependencies:

   ```bash
   brew install pkg-config
   brew install cmake
   ```

2. Install Python dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Download pretrained model files:

   Download the following files from [GitHub Releases](https://github.com/WangWilly/gaze-correction-cam/releases) and place them in the appropriate directories:
   - **Face landmark detector**: `shape_predictor_68_face_landmarks.dat`
     - Place in: `lm_feat/shape_predictor_68_face_landmarks.dat`
   - **Gaze correction model weights**: FLX model (Left and Right eye models)
     - Place in: `weights/warping_model/flx/12/L/` and `weights/warping_model/flx/12/R/`
     - Required files per directory: `checkpoint`, `L.data-00000-of-00001` / `R.data-00000-of-00001`, `L.index` / `R.index`, `L.meta` / `R.meta`

   - **(Optional) MediaPipe model**: `face_landmarker.task` (for MediaPipe backend)
     - Place in: `models/face_landmarker.task`
     - Download from [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)

## Usage

### Gaze Correction — Single Window (Recommended)

```bash
# Default dlib backend — interactive camera picker on first run
poetry run python gaze_correct.py

# MediaPipe backend (requires models/face_landmarker.task)
poetry run python gaze_correct.py --backend mediapipe

# Skip the picker and specify a camera directly
poetry run python gaze_correct.py --camera 0

# Adjust eye-patch scale to reduce magnification (default: 0.85)
poetry run python gaze_correct.py --eye-scale 0.85
```

**Camera picker:** Omitting `--camera` opens an interactive list of all detected cameras with a live preview window so you can confirm the right device before starting.

#### Controls

| Key | Action                        |
| --- | ----------------------------- |
| `g` | Toggle gaze correction on/off |
| `c` | Toggle calibration mode       |
| `q` | Quit application              |

#### Calibration Mode Controls

When calibration mode is enabled (press `c`):

| Key                          | Action                                 |
| ---------------------------- | -------------------------------------- |
| Arrow keys (`↑` `↓` `←` `→`) | Adjust camera offset X/Y (±0.5 cm)     |
| `+` / `-`                    | Adjust camera offset Z depth (±0.5 cm) |
| `[` / `]`                    | Adjust focal length (±10 pixels)       |
| `e` / `E`                    | Adjust eye-patch scale (±0.05)         |
| `r`                          | Reset to default values                |

The calibration overlay displays:

- Current camera offset (X, Y, Z in cm)
- Estimated eye position (X, Y, Z in cm)
- Current focal length (in pixels)
- Current eye-patch scale
- Top-view diagram showing camera, screen, and eye positions

---

### Virtual Camera — OBS Output

Stream gaze-corrected video to a virtual camera device (OBS, video conferencing apps):

```bash
# Start virtual camera with gaze correction
poetry run python virtual_cam.py

# Specify backend and camera
poetry run python virtual_cam.py --backend mediapipe --camera 4

# Pass raw frames without correction (debug)
poetry run python virtual_cam.py --passthrough
```

In OBS: **Add Source → Video Capture Device → "OBS Virtual Camera"**

> **Prerequisite:** OBS must be installed and Virtual Camera must be started from OBS Controls before running.

---

### Geometry Calibration Wizard

Solve accurate `focal_length` and `camera_offset` from your physical setup — run once per screen/camera configuration:

```bash
poetry run python bin_calibrate.py
```

The wizard walks through four steps:

1. **Screen size** — pick a preset or enter exact height in cm
2. **Camera position** — how far the lens sits above the top screen edge
3. **Your measurements** — inter-pupillary distance (IPD) and sitting distance
4. **Face capture** — 5-second automated IPD measurement via live camera

Settings are saved to `user_settings.db` and loaded automatically by `gaze_correct.py` and `virtual_cam.py`.

## System Requirements

- macOS with camera access permissions
- Sufficient GPU resources for real-time processing
- Webcam or video capture device

## Architecture & Module Documentation

### System Overview

This is a **real-time gaze correction system** that redirects eye gaze in video streams to create natural eye contact during video calls. The system uses face detection, facial landmarks, and deep learning models to warp eye regions.

### File Structure & Module Organization

#### 1. Entry Points

##### gaze_correct.py ⭐ Main Application

- **Purpose**: Single-window gaze correction app with real-time controls
- **Features**:
  - Interactive camera picker with live preview (`--camera` optional)
  - Auto-detects camera resolution
  - Toggle gaze correction on/off (`g` key)
  - Calibration mode for camera offset and eye-scale adjustment (`c` key)
  - `--eye-scale` flag to reduce warp magnification (default: 0.85)
  - Supports multiple backends (dlib/MediaPipe)
- **Flow**: `Camera Input → FacePredictor → GazeCorrector → Display Output`

##### virtual_cam.py

- **Purpose**: Headless pipeline that outputs corrected frames to a virtual camera device
- **Use case**: OBS, Zoom, Teams — any app that reads from a camera
- **Features**: Same backends and `--eye-scale` as `gaze_correct.py`, plus `--passthrough` for debug

##### bin_calibrate.py

- **Purpose**: Guided geometry calibration wizard — solves `focal_length` and `camera_offset` from physical measurements + a 5-second face capture
- **Output**: Saves to `user_settings.db`, loaded automatically by the other entry points

##### bin_focal_length_calibration.py

- Legacy standalone tool for camera focal length calibration

##### bin_test_mediapipe_detection.py

- Test utility for MediaPipe face detection

#### 2. Core Modules (displayers/)

The `displayers/` directory contains the main business logic components:

##### face_predictor.py - Face Detection & Landmark Extraction

**Purpose**: Abstract interface for face detection backends

**Key Classes**:

- `FacePredictor` (ABC): Interface for face detection
- `DlibFacePredictor`: Implementation using dlib (68 landmarks)
- `MediaPipeFacePredictor`: Implementation using Google MediaPipe
- Data classes: `FaceData`, `EyeData`, `EyeLandmarks`

**Process**: `Input Frame → Face Detection → Landmark Prediction → Eye Extraction → EyeData`

**Output**: `FaceData` containing:

- Left/right eye images (normalized 48×64)
- Anchor maps (feature point maps for spatial guidance)
- Eye center coordinates
- Original positions in frame

##### gaze_corrector.py - Gaze Correction Model

**Purpose**: Wraps TensorFlow models for eye gaze correction

**Key Classes**:

- `GazeModel`: TensorFlow model wrapper (loads L/R eye models)
- `GazeCorrector`: High-level interface for gaze correction
- `CameraConfig`: Camera geometry (focal length, IPD, camera offset)

**Process**:

```
EyeData + Camera Geometry → TF Model Inference → Warped Eye Image
                          ↓
                    Angle Calculation (3D geometry)
```

**Components**:

1. **Model Loading**: Loads separate L/R eye TensorFlow models from `weights/`
2. **Angle Calculation**: Computes gaze redirection angle based on:
   - Eye position in 3D space
   - Camera position relative to screen
   - Target gaze direction (toward camera)
3. **Eye Warping**: Applies learned transformation to redirect gaze

**Camera Geometry**:

- `focal_length`: Camera focal length (pixels)
- `ipd`: Inter-pupillary distance (cm)
- `camera_offset`: Camera position (X, Y, Z) relative to screen center

##### dis_single_window.py - Application Orchestrator

**Purpose**: Main application logic coordinating all components

**Key Class**: `SingleWindowGazeCorrector`

**Responsibilities**:

1. Camera capture and frame processing
2. FacePredictor → GazeCorrector pipeline
3. Real-time toggle controls
4. Calibration mode UI
5. Composite frame rendering

**Pipeline**:

```
Camera Frame
    ↓
Resize for Face Detection (320×240)
    ↓
FacePredictor.list_eye_data()
    ↓
For each eye:
    - If gaze_enabled: GazeCorrector.correct_eye()
    - Else: Use original eye image
    ↓
Composite corrected eyes onto original frame
    ↓
Draw status overlay
    ↓
Display in window
```

#### 3. TensorFlow Models (tf_models/)

##### flx.py - FLX Model Architecture

**Purpose**: Defines the neural network architecture for gaze correction

**Key Components**:

- `encoder()`: Encodes gaze angle into spatial feature map
- `trans_module()`: Transformation module with skip connections
- `apply_lcm()`: Light color modulation for realistic rendering
- `inference()`: Main forward pass combining all components

**Architecture**:

```
Eye Image + Anchor Map + Angle
    ↓
[Feature Extraction CNN]
    ↓
[Angle Encoder] → Spatial Feature Map
    ↓
[Transformation Module (Dense CNN)]
    ↓
[Flow Field Generation]
    ↓
[Spatial Transformer] → Warped Image
    ↓
[Light Color Modulation]
    ↓
Corrected Eye Image
```

##### transformation.py - Spatial Transformer

**Purpose**: Implements differentiable image warping

**Key Functions**:

- `meshgrid()`: Generates coordinate grid
- `interpolate()`: Bilinear interpolation for smooth warping
- `apply_transformation()`: Applies flow field to warp image

**Used for**: Applying learned pixel displacement fields to eye images

##### tf_utils.py

- Common TensorFlow utilities
- CNN/DNN blocks with batch normalization

#### 4. Utilities (utils/)

##### config.py - Configuration Management

**Purpose**: Centralized configuration using argparse

**Parameters**:

- Model dimensions (height=48, width=64, ef_dim=12)
- Camera parameters (focal length, IPD, camera offset)
- Network settings (IP, ports for multi-process mode)

##### logger.py - Logging Utility

**Purpose**: Formatted logging with timestamps and thread IDs

**Format**: `2026-01-27 10:30:45.123 Python[12345:67890] +[ClassName]: Message`

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN APPLICATION                         │
│                       (gaze_correct.py)                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
         ┌─────────────────────────────┐
         │   Camera Capture (OpenCV)   │
         │   Original: 640×480         │
         └─────────────┬───────────────┘
                       │
                       ↓
         ┌─────────────────────────────┐
         │  Resize for Detection       │
         │  Downscaled: 320×240        │
         └─────────────┬───────────────┘
                       │
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│                    FACE DETECTION LAYER                          │
│                  (displayers/face_predictor.py)                  │
├──────────────────────────────────────────────────────────────────┤
│  • Detect face(s) in frame                                       │
│  • Predict 68 facial landmarks (dlib) OR                         │
│  • Predict 478 landmarks (MediaPipe)                             │
│  • Extract eye regions (6 points per eye)                        │
│  • Resize eye images to 48×64                                    │
│  • Generate anchor maps (landmark feature maps)                  │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ↓ Output: List[FaceData]
              │
┌─────────────────────────────────────────────────────────────────┐
│  FaceData {                                                     │
│    left_eye: EyeData {                                          │
│      image: 48×64×3 (normalized)                                │
│      anchor_map: 48×64×12 (feature points)                      │
│      center: (x, y)                                             │
│      top_left: (row, col)                                       │
│    }                                                            │
│    right_eye: EyeData {...}                                     │
│  }                                                              │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ↓
┌──────────────────────────────────────────────────────────────────┐
│                   GAZE CORRECTION LAYER                          │
│                 (displayers/gaze_corrector.py)                   │
├──────────────────────────────────────────────────────────────────┤
│  For each eye:                                                   │
│    1. Calculate 3D eye position from landmarks                   │
│    2. Compute gaze redirection angle (toward camera)             │
│    3. Feed to TensorFlow model:                                  │
│       • Eye image (48×64×3)                                      │
│       • Anchor map (48×64×12)                                    │
│       • Gaze angle (θx, θy)                                      │
│    4. Model outputs warped eye image                             │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ↓
┌──────────────────────────────────────────────────────────────────┐
│                      TensorFlow MODEL                            │
│                     (tf_models/flx.py)                           │
├──────────────────────────────────────────────────────────────────┤
│  [Encoder] → Angle to spatial feature map                        │
│  [CNN Feature Extraction] → Image features                       │
│  [Transformation Module] → Flow field prediction                 │
│  [Spatial Transformer] → Apply warping                           │
│  [Light Color Module] → Adjust lighting                          │
└─────────────┬────────────────────────────────────────────────────┘
              │
              ↓ Corrected Eye Image (48×64×3)
              │
┌──────────────────────────────────────────────────────────────────┐
│                    COMPOSITE & DISPLAY                           │
│                (dis_single_window.py)                            │
├──────────────────────────────────────────────────────────────────┤
│  1. Resize corrected eyes back to original size                  │
│  2. Paste onto original 640×480 frame at eye positions           │
│  3. Draw status overlay (GAZE ON/OFF)                            │
│  4. Draw calibration overlay (if enabled)                        │
│  5. Display in OpenCV window                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

#### 1. Dependency Injection

- `FacePredictor` is injectable → easy to swap backends (dlib ↔ MediaPipe)
- `GazeCorrector` is injectable → testable and modular

#### 2. Abstract Interface

- `FacePredictor` is abstract base class
- Implementations: `DlibFacePredictor`, `MediaPipeFacePredictor`

#### 3. Configuration Objects

- Dataclasses for configuration (immutable, type-safe)
- `DisplayConfig`, `CameraConfig`, `GazeModelConfig`, etc.

#### 4. Separation of Concerns

- Face detection ≠ Gaze correction
- Display logic ≠ Model inference
- Configuration ≠ Business logic

### Module Responsibilities

| Module                | Input                      | Output           | Responsibility                    |
| --------------------- | -------------------------- | ---------------- | --------------------------------- |
| **face_predictor**    | Frame (BGR)                | `List[FaceData]` | Detect faces, extract eye regions |
| **gaze_corrector**    | `FaceData` + Camera Config | Corrected frame  | Apply gaze correction model       |
| **flx.py**            | Eye image + Anchor + Angle | Warped eye       | Neural network inference          |
| **transformation.py** | Flow field + Image         | Warped image     | Spatial transformation            |
| **dis_single_window** | Camera stream              | Display window   | Orchestrate pipeline, UI          |

### How It Works (High-Level)

1. **Capture** video frame from webcam
2. **Detect** face and extract 68 facial landmarks
3. **Extract** left/right eye regions (48×64 each)
4. **Calculate** 3D eye position and required gaze angle
5. **Inference** through trained CNN to generate warping flow field
6. **Warp** eye image using spatial transformer
7. **Composite** corrected eyes back onto original frame
8. **Display** result in real-time

The key innovation is the **learned warping transformation** that realistically redirects gaze while preserving eye appearance, lighting, and texture.

## References

The implementation is based on research in gaze correction techniques using warping-based convolutional neural networks.
