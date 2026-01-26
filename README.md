# Gaze Correction Camera

## Overview

This project implements a gaze correction system for video communication that uses computer vision and deep learning techniques to adjust eye gaze direction in real-time, providing a more natural eye contact experience during video calls. ([study more](./docs/orignal_doc.md))

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

### Single Window Application (Recommended)

Run the simplified single-window gaze correction application:

```bash
# Using default dlib backend
poetry run python bin_single_window.py

# Using MediaPipe backend (requires face_landmarker.task)
poetry run python bin_single_window.py --backend mediapipe

# Specify camera device
poetry run python bin_single_window.py --camera 0
```

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
| `r`                          | Reset to default values                |

The calibration overlay displays:

- Current camera offset (X, Y, Z in cm)
- Estimated eye position (X, Y, Z in cm)
- Current focal length (in pixels)
- Top-view diagram showing camera, screen, and eye positions

### Legacy Multi-Process Application (Deprecated)

The original multi-process application with socket communication is still available but deprecated:

```bash
poetry run python bin_regz_socket_MP_FD.py
```

**Note**: This requires additional configuration in `config.py` and is less user-friendly. Use `bin_single_window.py` instead.

## System Requirements

- macOS with camera access permissions
- Sufficient GPU resources for real-time processing
- Webcam or video capture device

## References

The implementation is based on research in gaze correction techniques using warping-based convolutional neural networks.
