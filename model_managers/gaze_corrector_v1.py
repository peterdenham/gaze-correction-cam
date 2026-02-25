"""
Gaze Corrector V1 Module

This module provides the gaze correction model wrapper and inference logic,
with YAML-based configuration and database-backed user settings.
"""

import math
import yaml
import numpy as np
import tensorflow as tf
import cv2
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from tf_models.gaze_corrector_v1 import gaze_warp_model
from utils.logger import Logger
from model_managers.user_settings_db import UserSettingsDB


################################################################################
# Configuration Classes
################################################################################


@dataclass
class GazeWarpModelConfig:
    """Hyperparameters for the gaze warp model."""
    height: int = 48
    width: int = 64
    encoded_angle_dim: int = 16


@dataclass
class GazeModelConfig:
    """Configuration for the gaze correction model."""
    
    model_dir: str = "./weights/warping_model/flx/12/"
    eye_input_size: tuple[int, int] = (48, 64)  # (height, width)
    ef_dim: int = 12
    channel: int = 3
    gaze_warp_model: GazeWarpModelConfig = None
    
    def __post_init__(self):
        if self.gaze_warp_model is None:
            self.gaze_warp_model = GazeWarpModelConfig()
        elif isinstance(self.gaze_warp_model, dict):
            self.gaze_warp_model = GazeWarpModelConfig(**self.gaze_warp_model)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GazeModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert eye_input_size from list to tuple
        if 'eye_input_size' in data and isinstance(data['eye_input_size'], list):
            data['eye_input_size'] = tuple(data['eye_input_size'])
        
        return cls(**data)


@dataclass
class CameraUserSetting:
    """User-adjustable camera and screen geometry settings."""
    
    focal_length: float = 650.0
    ipd: float = 6.3  # Inter-pupillary distance in cm
    camera_offset: tuple[float, float, float] = (0, -21, -1)  # relative to screen center
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            'focal_length': self.focal_length,
            'ipd': self.ipd,
            'camera_offset': list(self.camera_offset),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CameraUserSetting":
        """Load from dictionary."""
        if 'camera_offset' in data and isinstance(data['camera_offset'], list):
            data['camera_offset'] = tuple(data['camera_offset'])
        return cls(**data)


################################################################################
# Gaze Correction Model
################################################################################


class GazeModel:
    """TensorFlow model wrapper for eye gaze correction."""

    def __init__(self, config: GazeModelConfig):
        self.cfg = config
        self.logger = Logger("GazeModel")
        self._load_models()

    def _load_models(self):
        """Load left and right eye models."""
        # Build ModelConfig for gaze_warp_model
        model_cfg = gaze_warp_model.ModelConfig(
            height=self.cfg.gaze_warp_model.height,
            width=self.cfg.gaze_warp_model.width,
            encoded_angle_dim=self.cfg.gaze_warp_model.encoded_angle_dim,
        )

        # Left eye model
        self.logger.log("Loading left eye model...")
        with tf.Graph().as_default() as g_left:
            with tf.name_scope("inputs"):
                self.le_img = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.channel],
                )
                self.le_fp = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.ef_dim],
                )
                self.le_ang = tf.compat.v1.placeholder(tf.float32, [None, 2])

            self.le_pred, _, _ = gaze_warp_model.build_inference_graph(
                self.le_img, self.le_fp, self.le_ang, False, model_cfg
            )
            self.l_sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
                graph=g_left,
            )
            self._restore_checkpoint(self.l_sess, self.cfg.model_dir + "L/")

        # Right eye model
        self.logger.log("Loading right eye model...")
        with tf.Graph().as_default() as g_right:
            with tf.name_scope("inputs"):
                self.re_img = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.channel],
                )
                self.re_fp = tf.compat.v1.placeholder(
                    tf.float32,
                    [None, self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], self.cfg.ef_dim],
                )
                self.re_ang = tf.compat.v1.placeholder(tf.float32, [None, 2])

            self.re_pred, _, _ = gaze_warp_model.build_inference_graph(
                self.re_img, self.re_fp, self.re_ang, False, model_cfg
            )
            self.r_sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
                graph=g_right,
            )
            self._restore_checkpoint(self.r_sess, self.cfg.model_dir + "R/")

        self.logger.log("Models loaded successfully")

    def _restore_checkpoint(self, sess, model_dir: str):
        """Restore model from checkpoint."""
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        ckpt = tf.compat.v1.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            self.logger.log(f"Warning: No checkpoint found in {model_dir}")

    def infer_eye(
        self, eye: str, img: np.ndarray, anchor_map: np.ndarray, angle: list
    ) -> np.ndarray:
        """
        Run inference for a single eye.

        Args:
            eye: "L" or "R"
            img: Eye image normalized to [0, 1], shape (H, W, 3)
            anchor_map: Feature point map, shape (H, W, ef_dim)
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image, shape (H, W, 3)
        """
        if eye == "L":
            result = self.l_sess.run(
                self.le_pred,
                feed_dict={
                    self.le_img: np.expand_dims(img, axis=0),
                    self.le_fp: np.expand_dims(anchor_map, axis=0),
                    self.le_ang: np.expand_dims(angle, axis=0),
                },
            )
        else:
            result = self.r_sess.run(
                self.re_pred,
                feed_dict={
                    self.re_img: np.expand_dims(img, axis=0),
                    self.re_fp: np.expand_dims(anchor_map, axis=0),
                    self.re_ang: np.expand_dims(angle, axis=0),
                },
            )
        return result.reshape(self.cfg.eye_input_size[0], self.cfg.eye_input_size[1], 3)

    def close(self):
        """Close TensorFlow sessions."""
        self.l_sess.close()
        self.r_sess.close()


################################################################################
# Gaze Corrector
################################################################################


class GazeCorrector:
    """
    High-level gaze correction interface with database-backed user settings.
    
    Takes FaceData from a FacePredictor and applies gaze correction.
    video_size is passed from outside via apply_correction.
    """

    def __init__(
        self,
        config_path: str = "./model_managers/gaze_corrector_v1_01.yaml",
        db_path: str = "./user_settings.db",
        setting_name: str = "camera_default",
        smooth_alpha: float = 0.4,
        eye_scale: float = 0.92,
    ):
        """
        Initialize gaze corrector.
        
        Args:
            config_path: Path to YAML configuration file
            db_path: Path to SQLite database
            setting_name: Name of camera setting to load from database
        """
        self.logger = Logger("GazeCorrector")
        
        # Load model configuration from YAML
        self.model_cfg = GazeModelConfig.from_yaml(config_path)
        self.logger.log(f"Loaded model config from: {config_path}")
        
        # Initialize database
        self.db = UserSettingsDB(db_path)
        self.setting_name = setting_name
        
        # Load camera settings from database or use defaults
        self.camera_settings = self._load_camera_settings()
        
        # Initialize model
        self.model = GazeModel(self.model_cfg)

        # Last estimated eye position (for visualization)
        self.last_eye_position: list[float] = [0, 0, -60]

        # EMA smoothing for correction angles — reduces frame-to-frame jitter.
        # alpha=1.0 disables smoothing (raw angles); lower values smooth more.
        self.smooth_alpha: float = smooth_alpha
        self._smoothed_angles: list[float] | None = None

        # Scale factor applied to the corrected eye patch before blending.
        # The spatial transformer warp tends to magnify the eye region slightly
        # (it "pulls" pixels outward to redirect gaze), making eyes look bigger.
        # Values < 1.0 shrink the corrected patch centered in the bbox so the
        # apparent eye size matches the uncorrected frame.  0.92 compensates
        # for roughly 5-15° of gaze correction.  Use 1.0 to disable.
        self.eye_scale: float = eye_scale

        # Reusable thread pool for parallel eye correction (avoids per-frame churn)
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _load_camera_settings(self) -> CameraUserSetting:
        """Load camera settings from database or return defaults."""
        saved = self.db.get_setting(self.setting_name)
        if saved:
            self.logger.log(f"Loaded camera settings from database: {self.setting_name}")
            return CameraUserSetting.from_dict(saved)
        else:
            self.logger.log("Using default camera settings")
            settings = CameraUserSetting()
            # Save defaults to database
            self.db.save_setting(self.setting_name, settings.to_dict())
            return settings

    def save_camera_settings(self):
        """Save current camera settings to database."""
        self.db.save_setting(self.setting_name, self.camera_settings.to_dict())
        self.logger.log(f"Saved camera settings to database: {self.setting_name}")

    ############################################################################
    # Camera Offset Adjustment API
    ############################################################################

    def get_camera_offset(self) -> tuple[float, float, float]:
        """Get current camera offset (x, y, z) in cm."""
        return self.camera_settings.camera_offset

    def set_camera_offset(self, x: float, y: float, z: float) -> None:
        """
        Set camera offset relative to screen center.

        Args:
            x: Horizontal offset in cm (positive = right)
            y: Vertical offset in cm (positive = down)
            z: Depth offset in cm (negative = behind screen)
        """
        self.camera_settings.camera_offset = (x, y, z)
        self.save_camera_settings()
        self.logger.log(f"Camera offset set to: ({x:.1f}, {y:.1f}, {z:.1f})")

    def adjust_camera_offset(self, dx: float = 0, dy: float = 0, dz: float = 0) -> tuple[float, float, float]:
        """
        Adjust camera offset by delta values.

        Args:
            dx: Change in X (horizontal)
            dy: Change in Y (vertical)
            dz: Change in Z (depth)

        Returns:
            New camera offset tuple
        """
        x, y, z = self.camera_settings.camera_offset
        self.camera_settings.camera_offset = (x + dx, y + dy, z + dz)
        self.save_camera_settings()
        return self.camera_settings.camera_offset

    def get_last_eye_position(self) -> list[float]:
        """Get the last estimated eye position [x, y, z] in cm."""
        return self.last_eye_position

    ############################################################################
    # Focal Length Adjustment API
    ############################################################################

    def get_focal_length(self) -> float:
        """Get current focal length in pixels."""
        return self.camera_settings.focal_length

    def set_focal_length(self, focal_length: float) -> None:
        """
        Set focal length.

        Args:
            focal_length: Focal length in pixels (typically 500-1000)
        """
        self.camera_settings.focal_length = focal_length
        self.save_camera_settings()
        self.logger.log(f"Focal length set to: {focal_length:.1f}")

    def adjust_focal_length(self, delta: float) -> float:
        """
        Adjust focal length by delta value.

        Args:
            delta: Change in focal length (pixels)

        Returns:
            New focal length
        """
        self.camera_settings.focal_length += delta
        self.save_camera_settings()
        return self.camera_settings.focal_length

    ############################################################################
    # IPD Adjustment API
    ############################################################################

    def get_ipd(self) -> float:
        """Get inter-pupillary distance in cm."""
        return self.camera_settings.ipd

    def set_ipd(self, ipd: float) -> None:
        """
        Set inter-pupillary distance.

        Args:
            ipd: IPD in cm (typically 5.5-7.0)
        """
        self.camera_settings.ipd = ipd
        self.save_camera_settings()
        self.logger.log(f"IPD set to: {ipd:.1f} cm")

    ############################################################################
    # Eye Scale Adjustment API
    ############################################################################

    def get_eye_scale(self) -> float:
        """Get current eye scale factor."""
        return self.eye_scale

    def adjust_eye_scale(self, delta: float) -> float:
        """
        Adjust eye scale by delta, clamped to [0.5, 1.0].

        Args:
            delta: Change in scale (positive = larger, negative = smaller)

        Returns:
            New eye scale value
        """
        self.eye_scale = max(0.5, min(1.0, self.eye_scale + delta))
        return self.eye_scale

    ############################################################################
    # Gaze Estimation and Correction
    ############################################################################

    def estimate_gaze_angle(
        self, 
        le_center: tuple[float, float], 
        re_center: tuple[float, float],
        video_size: tuple[int, int],
    ) -> tuple[list[int], list[float]]:
        """
        Estimate gaze redirection angles based on eye positions.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels
            video_size: (width, height) of video frame

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        settings = self.camera_settings

        # Estimate eye depth from inter-pupillary distance
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(settings.focal_length * settings.ipd) / ipd_pixels

        # Estimate eye position in 3D (camera coordinates, cm)
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - video_size[0])
            / (2 * settings.focal_length)
            + settings.camera_offset[0]
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - video_size[1])
            / (2 * settings.focal_length)
            + settings.camera_offset[1]
        )

        eye_position = [eye_x, eye_y, eye_z]

        # Store for visualization
        self.last_eye_position = eye_position

        # Target gaze point (looking at camera)
        target = (0, 0, 0)

        # Calculate angles
        a_v = math.degrees(math.atan((target[1] - eye_y) / (target[2] - eye_z)))
        a_h = math.degrees(math.atan((target[0] - eye_x) / (target[2] - eye_z)))

        # Add camera offset angles
        a_v += math.degrees(
            math.atan((eye_y - settings.camera_offset[1]) / (settings.camera_offset[2] - eye_z))
        )
        a_h += math.degrees(
            math.atan((eye_x - settings.camera_offset[0]) / (settings.camera_offset[2] - eye_z))
        )

        return [a_v, a_h], eye_position

    def _blend_eye(self, frame: np.ndarray, corrected: np.ndarray, eye_data) -> None:
        """
        Blend a corrected eye patch into frame using a soft elliptical mask.

        Rather than a hard cut-paste (which leaves seams when the corrected
        patch differs in brightness/tone), this builds a Gaussian-blurred
        elliptical mask centred on the eye and alpha-composites the result.
        The mask fades to 0 at the boundary, so the output always transitions
        smoothly back into the surrounding skin.

        Args:
            frame: BGR frame to modify in-place
            corrected: Corrected eye image in [0, 1] float, shape (H, W, 3)
            eye_data: EyeData with top_left and original_size
        """
        top, left = eye_data.top_left
        orig_h, orig_w = eye_data.original_size

        # Guard against clamped-to-zero slices
        fh, fw = frame.shape[:2]
        bottom = min(top + orig_h, fh)
        right = min(left + orig_w, fw)
        actual_h = bottom - top
        actual_w = right - left
        if actual_h <= 0 or actual_w <= 0:
            return

        # Build soft elliptical mask — filled ellipse blurred to soft edges.
        # Inset the ellipse by ~1/6 of each half-axis so the Gaussian blur has
        # room to decay to zero before reaching the corners.  Using
        # BORDER_CONSTANT (zero-pad) prevents BORDER_REFLECT_101 from folding
        # the high-value ellipse interior back into the corners, which would
        # otherwise multiply against the model's black out-of-bounds border and
        # produce visible dark tick-mark artifacts at each corner.
        mask = np.zeros((actual_h, actual_w), dtype=np.float32)
        cx, cy = actual_w // 2, actual_h // 2
        inset_x = max(2, cx // 6)
        inset_y = max(2, cy // 6)
        cv2.ellipse(mask, (cx, cy), (max(1, cx - inset_x), max(1, cy - inset_y)), 0, 0, 360, 1.0, -1)

        # Blur radius: ~1/4 of shorter dimension, forced odd
        blur_r = max(3, min(actual_h, actual_w) // 4)
        blur_r = blur_r if blur_r % 2 == 1 else blur_r + 1
        mask = cv2.GaussianBlur(mask, (blur_r, blur_r), 0, borderType=cv2.BORDER_CONSTANT)[..., np.newaxis]  # (H, W, 1)

        roi = frame[top:bottom, left:right].astype(np.float32)
        corrected_255 = np.clip(corrected[:actual_h, :actual_w] * 255.0, 0.0, 255.0)

        # Counteract the spatial-transformer's tendency to magnify the eye.
        # Shrink the corrected patch to eye_scale × bbox size, center it, and
        # fill the surrounding area with the original frame pixels so the
        # transition is seamless before the Gaussian mask is applied.
        if self.eye_scale < 1.0:
            sh = max(1, int(actual_h * self.eye_scale))
            sw = max(1, int(actual_w * self.eye_scale))
            shrunk = cv2.resize(corrected_255, (sw, sh))
            scaled = roi.copy()
            y0 = (actual_h - sh) // 2
            x0 = (actual_w - sw) // 2
            scaled[y0:y0 + sh, x0:x0 + sw] = shrunk
            corrected_255 = scaled

        blended = corrected_255 * mask + roi * (1.0 - mask)
        frame[top:bottom, left:right] = blended.astype(np.uint8)

    def correct_eye(
        self, eye_data, eye_side: str, angle: list[float]
    ) -> np.ndarray:
        """
        Apply gaze correction to a single eye.

        Args:
            eye_data: Eye extraction data (EyeData from face_predictor)
            eye_side: "L" or "R"
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image resized to original size
        """
        result = self.model.infer_eye(
            eye_side, eye_data.image, eye_data.anchor_map, angle
        )
        # Crop the model's ~3px dark border before resizing.  The spatial
        # transformer outputs zero (black) wherever it samples outside the
        # source image bounds, leaving a dark fringe along all four edges.
        # The soft elliptical mask in _blend_eye still has significant weight
        # close to the bounding-box edge (it can't decay to zero over just a
        # few pixels), so without this crop those black pixels composite
        # visibly into the output frame.  48→42 / 64→58 retains >87% of the
        # model's prediction area and the resize back to original size fills
        # the full eye region cleanly.
        B = 3
        result = result[B:-B, B:-B]
        return cv2.resize(result, (eye_data.original_size[1], eye_data.original_size[0]))

    def apply_correction(self, frame: np.ndarray, face_data, video_size: tuple[int, int]) -> np.ndarray:
        """
        Apply gaze correction to a frame using extracted face data.

        Args:
            frame: BGR video frame to modify
            face_data: Extracted face/eye data from FacePredictor (FaceData)
            video_size: (width, height) of video frame

        Returns:
            Frame with corrected gaze
        """
        if face_data.left_eye is None or face_data.right_eye is None:
            return frame

        le = face_data.left_eye
        re = face_data.right_eye

        # Estimate gaze angle (video_size passed from outside)
        alpha, _ = self.estimate_gaze_angle(le.center, re.center, video_size)

        # EMA smoothing: blend new angle with history to reduce jitter.
        # On first detection, seed with the raw angle (no lag on startup).
        if self._smoothed_angles is None:
            self._smoothed_angles = list(alpha)
        else:
            a = self.smooth_alpha
            self._smoothed_angles[0] = a * alpha[0] + (1 - a) * self._smoothed_angles[0]
            self._smoothed_angles[1] = a * alpha[1] + (1 - a) * self._smoothed_angles[1]
        alpha = self._smoothed_angles

        # Correct both eyes in parallel (independent TF graphs/sessions)
        fut_l = self._pool.submit(self.correct_eye, le, "L", alpha)
        fut_r = self._pool.submit(self.correct_eye, re, "R", alpha)
        le_corrected = fut_l.result()
        re_corrected = fut_r.result()

        # Blend corrected eye patches into frame with soft elliptical masks
        self._blend_eye(frame, le_corrected, le)
        self._blend_eye(frame, re_corrected, re)

        return frame

    def close(self):
        """Release model resources."""
        self._pool.shutdown(wait=True)
        self.model.close()
