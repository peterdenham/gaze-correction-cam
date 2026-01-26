"""
Single Window Gaze Corrector

A simplified gaze correction application that:
- Uses a single window (no socket communication)
- Allows real-time toggle of gaze correction with 'g' key
- Supports injectable face predictor backends
- Decoupled architecture: FacePredictor -> GazeCorrector
"""

import cv2
from dataclasses import dataclass
from typing import Optional

from utils.logger import Logger
from displayers.face_predictor import (
    FacePredictor,
    EyeExtractionConfig,
    create_face_predictor,
)
from displayers.gaze_corrector import (
    GazeCorrector,
    GazeModelConfig,
    CameraConfig,
)


################################################################################
# Configuration
################################################################################


@dataclass
class DisplayConfig:
    """Configuration for the display application."""

    video_size: tuple[int, int] = (640, 480)
    face_detect_size: tuple[int, int] = (320, 240)
    window_name: str = "Gaze Correction"

    @property
    def x_ratio(self) -> float:
        return self.video_size[0] / self.face_detect_size[0]

    @property
    def y_ratio(self) -> float:
        return self.video_size[1] / self.face_detect_size[1]


################################################################################
# Single Window Gaze Corrector
################################################################################


class SingleWindowGazeCorrector:
    """
    Single-window gaze correction application with real-time toggle.

    This class orchestrates:
    - FacePredictor: Detects faces and extracts eye data
    - GazeCorrector: Applies gaze correction model

    Controls:
        - 'g': Toggle gaze correction on/off
        - 'q': Quit
    """

    def __init__(
        self,
        face_predictor: Optional[FacePredictor] = None,
        gaze_corrector: Optional[GazeCorrector] = None,
        display_config: Optional[DisplayConfig] = None,
        camera_id: int = 0,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.display_cfg = display_config or DisplayConfig()

        # Initialize face predictor (injectable)
        self.face_predictor = face_predictor or create_face_predictor("dlib")
        self.logger.log(f"Using face predictor: {self.face_predictor.get_name()}")

        # Initialize gaze corrector (injectable)
        self.gaze_corrector = gaze_corrector or GazeCorrector()

        # Eye extraction config (matches model requirements)
        self.eye_config = EyeExtractionConfig()

        # State
        self.gaze_correction_enabled = True
        self.camera_id = camera_id

    def draw_status(self, frame) -> None:
        """Draw status overlay on frame."""
        status = "GAZE ON" if self.gaze_correction_enabled else "GAZE OFF"
        color = (0, 255, 0) if self.gaze_correction_enabled else (0, 0, 255)

        cv2.rectangle(frame, (10, 10), (150, 60), (0, 0, 0), -1)
        cv2.putText(
            frame, status, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"[{self.face_predictor.get_name()}]", (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA
        )

    def process_frame(self, frame):
        """
        Process a single frame with gaze correction.

        Args:
            frame: BGR video frame

        Returns:
            Processed frame with gaze correction applied
        """
        display_frame = frame.copy()

        # For MediaPipe, process the frame first
        if hasattr(self.face_predictor, "process_frame"):
            self.face_predictor.process_frame(frame)

        # Detect faces on downscaled grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect_gray = cv2.resize(gray, self.display_cfg.face_detect_size)
        faces = self.face_predictor.detect_faces(detect_gray)

        # Process first detected face
        for face in faces:
            try:
                # Get landmarks
                landmarks = self.face_predictor.predict_landmarks(
                    gray, face, (self.display_cfg.x_ratio, self.display_cfg.y_ratio)
                )
                if landmarks is None:
                    continue

                # Extract eye data
                face_data = self.face_predictor.extract_eye_data(
                    display_frame, landmarks, self.eye_config
                )

                # Apply gaze correction
                display_frame = self.gaze_corrector.apply_correction(
                    display_frame, face_data
                )
            except Exception as e:
                self.logger.log(f"Error: {e}")
            break  # Only process first face

        return display_frame

    def run(self):
        """Main application loop."""
        self.logger.log(f"Starting camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_cfg.video_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_cfg.video_size[1])

        cv2.namedWindow(self.display_cfg.window_name)
        self.logger.log("Press 'g' to toggle gaze correction, 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.log("Failed to read frame")
                break

            if self.gaze_correction_enabled:
                display_frame = self.process_frame(frame)
            else:
                display_frame = frame.copy()

            self.draw_status(display_frame)
            cv2.imshow(self.display_cfg.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                self.gaze_correction_enabled = not self.gaze_correction_enabled
                self.logger.log(
                    f"Gaze correction: {'enabled' if self.gaze_correction_enabled else 'disabled'}"
                )

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.gaze_corrector.close()
        self.logger.log("Shutdown complete")


################################################################################
# Entry Point
################################################################################


def main():
    """Run the single window gaze corrector."""
    import argparse

    parser = argparse.ArgumentParser(description="Single Window Gaze Correction")
    parser.add_argument(
        "--backend",
        type=str,
        default="dlib",
        choices=["dlib", "mediapipe"],
        help="Face detection backend",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID",
    )
    args = parser.parse_args()

    # Create injectable components
    predictor = create_face_predictor(args.backend)
    corrector = GazeCorrector()

    # Run application
    app = SingleWindowGazeCorrector(
        face_predictor=predictor,
        gaze_corrector=corrector,
        camera_id=args.camera,
    )
    app.run()


if __name__ == "__main__":
    main()
