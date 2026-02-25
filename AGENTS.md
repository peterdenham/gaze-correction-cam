# Gaze Correction Camera - Agent Guidelines

## Project Overview
This is a Python 3.12+ macOS-only gaze correction application using TensorFlow (DeepWarp CNN), OpenCV, and dlib/MediaPipe for face detection. It provides real-time eye gaze redirection for video communication.

## Build & Run Commands
- Install dependencies: `poetry install`
- Run main application: `poetry run python bin_single_window.py`
- Run with MediaPipe backend: `poetry run python bin_single_window.py --backend mediapipe`
- Run virtual camera: `poetry run python bin_virtual_cam.py`
- Format code: `poetry run black .` (uses black formatter)
- Test scripts: `bin_test_dlib_detection.py` and `bin_test_mediapipe_detection.py` (visual tests only)

## Code Style Guidelines

### Formatting
- Uses black formatter (dev dependency) with default settings
- Line length: 88 characters
- Double quotes for strings
- Trailing commas in multi-line lists/args

### Imports
- Standard library → Third-party → Local modules
- Group imports with blank lines between groups
- Use `import X` for top-level modules, `from X import Y` for specific items
- Lazy imports for heavy optional dependencies (dlib, mediapipe)
- Standard aliases: `np`, `tf`, `cv2`

### Naming Conventions
- Classes: `PascalCase` (`GazeCorrector`, `FacePredictor`)
- Functions/Methods: `snake_case` (`list_cameras`, `apply_correction`)
- Constants: `UPPER_SNAKE_CASE` (`BORDER_CROP_PIXELS`, `LEFT_EYE_INDICES`)
- Private members: `_leading_underscore` (`_load_models`, `_detect_faces`)
- Module-level variables: `snake_case` (`model_config`)

### Type Annotations
- Pervasive use of type hints throughout
- Modern Python 3.12+ syntax: `list[...]`, `tuple[...]`, `dict[...]`
- Use `typing.Optional` for nullable types
- `np.ndarray`, `tf.Tensor` for array/tensor annotations
- Forward references as strings for circular dependencies

### Error Handling
- Return `None` or empty collections for expected failures (no exceptions)
- Use `Optional` return types for potentially missing data
- `try/except` only at pipeline boundaries and external system interactions
- Graceful degradation with fallback defaults
- No custom exception classes defined

### Docstrings
- Google-style docstrings for all public classes and methods
- Include Args, Returns sections
- Example and Notes sections when appropriate
- Module-level docstrings explaining purpose
- Entry-point scripts include Usage/Controls documentation

### File Structure & Patterns
- Each file has module docstring at top
- Logical sections separated by 80-character `#` banners
- Configuration dataclasses using `@dataclass` decorator
- Abstract Base Class pattern with factory function creation
- Constructor injection for dependencies
- Resource cleanup via `close()` methods
- Backward compatibility aliases with `"""Deprecated: Use X instead."""` docstrings

### Architecture Patterns
- Dependency injection for face predictors and gaze correctors
- Factory functions for creating backend implementations
- Dataclass-based configuration objects
- Separation of concerns (face detection, model inference, display orchestration)
- TensorFlow 1.x-style graph construction with `tf.compat.v1` APIs
- Parallel execution using `ThreadPoolExecutor`
- SQLite-backed user settings persistence

## Key Conventions and Gotchas
- Uses TensorFlow 1.x compatibility API (`tf.compat.v1.Session`, placeholders)
- Custom Logger class instead of Python's standard logging module
- All entry points use `if __name__ == "__main__": main()` pattern
- No automated tests - only manual visual test scripts exist
- Requires macOS 15.2+ with PyObjC and OBS virtual camera driver
- External model files (dlib/MediaPipe/TensorFlow) must be downloaded manually