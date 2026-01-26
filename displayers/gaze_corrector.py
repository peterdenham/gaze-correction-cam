"""
Gaze Corrector Module

This module provides the gaze correction model wrapper and inference logic,
decoupled from face detection and eye extraction.
"""

import math
import numpy as np
import tensorflow as tf
import cv2
from dataclasses import dataclass
from typing import Optional

import tf_models.flx as flx_model
from utils.config import get_config
from utils.logger import Logger
from displayers.face_predictor import EyeData, FaceData


################################################################################
# Configuration
################################################################################


@dataclass
class GazeModelConfig:
    """Configuration for the gaze correction model."""

    model_dir: str = "./weights/warping_model/flx/12/"
    eye_input_size: tuple[int, int] = (48, 64)  # (height, width)
    ef_dim: int = 12
    channel: int = 3


@dataclass
class CameraConfig:
    """Camera and screen geometry configuration."""

    focal_length: float = 650.0
    ipd: float = 6.3  # Inter-pupillary distance in cm
    camera_offset: tuple[float, float, float] = (0, -21, -1)  # relative to screen center
    video_size: tuple[int, int] = (640, 480)


################################################################################
# Gaze Correction Model
################################################################################


class GazeModel:
    """TensorFlow model wrapper for eye gaze correction."""

    def __init__(self, config: Optional[GazeModelConfig] = None):
        self.cfg = config or GazeModelConfig()
        self.logger = Logger("GazeModel")
        self._load_models()

    def _load_models(self):
        """Load left and right eye models."""
        general_cfg, _ = get_config()
        model_cfg = flx_model.ModelConfig.parse_from(general_cfg)

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

            self.le_pred, _, _ = flx_model.inference(
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

            self.re_pred, _, _ = flx_model.inference(
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
    High-level gaze correction interface.
    
    Takes FaceData from a FacePredictor and applies gaze correction.
    """

    def __init__(
        self,
        model_config: Optional[GazeModelConfig] = None,
        camera_config: Optional[CameraConfig] = None,
    ):
        self.model_cfg = model_config or GazeModelConfig()
        self.camera_cfg = camera_config or CameraConfig()
        self.logger = Logger("GazeCorrector")

        self.model = GazeModel(self.model_cfg)

        # Pixel border to cut when replacing eyes (reduces edge artifacts)
        self.pixel_cut = (3, 4)

    def estimate_gaze_angle(
        self, le_center: tuple[float, float], re_center: tuple[float, float]
    ) -> tuple[list[int], list[float]]:
        """
        Estimate gaze redirection angles based on eye positions.

        Args:
            le_center: Left eye center (x, y) in pixels
            re_center: Right eye center (x, y) in pixels

        Returns:
            (alpha [vertical, horizontal], eye_position [x, y, z])
        """
        cfg = self.camera_cfg

        # Estimate eye depth from inter-pupillary distance
        ipd_pixels = np.sqrt(
            (le_center[0] - re_center[0]) ** 2 + (le_center[1] - re_center[1]) ** 2
        )
        eye_z = -(cfg.focal_length * cfg.ipd) / ipd_pixels

        # Estimate eye position in 3D (camera coordinates, cm)
        eye_x = (
            -abs(eye_z)
            * (le_center[0] + re_center[0] - cfg.video_size[0])
            / (2 * cfg.focal_length)
            + cfg.camera_offset[0]
        )
        eye_y = (
            abs(eye_z)
            * (le_center[1] + re_center[1] - cfg.video_size[1])
            / (2 * cfg.focal_length)
            + cfg.camera_offset[1]
        )

        eye_position = [eye_x, eye_y, eye_z]

        # Target gaze point (looking at camera)
        target = (0, 0, 0)

        # Calculate angles
        a_v = math.degrees(math.atan((target[1] - eye_y) / (target[2] - eye_z)))
        a_h = math.degrees(math.atan((target[0] - eye_x) / (target[2] - eye_z)))

        # Add camera offset angles
        a_v += math.degrees(
            math.atan((eye_y - cfg.camera_offset[1]) / (cfg.camera_offset[2] - eye_z))
        )
        a_h += math.degrees(
            math.atan((eye_x - cfg.camera_offset[0]) / (cfg.camera_offset[2] - eye_z))
        )

        return [int(a_v), int(a_h)], eye_position

    def correct_eye(
        self, eye_data: EyeData, eye_side: str, angle: list[int]
    ) -> np.ndarray:
        """
        Apply gaze correction to a single eye.

        Args:
            eye_data: Eye extraction data
            eye_side: "L" or "R"
            angle: [vertical, horizontal] correction angles

        Returns:
            Corrected eye image resized to original size
        """
        result = self.model.infer_eye(
            eye_side, eye_data.image, eye_data.anchor_map, angle
        )
        # Resize back to original size
        return cv2.resize(result, (eye_data.original_size[1], eye_data.original_size[0]))

    def apply_correction(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        """
        Apply gaze correction to a frame using extracted face data.

        Args:
            frame: BGR video frame to modify
            face_data: Extracted face/eye data from FacePredictor

        Returns:
            Frame with corrected gaze
        """
        if face_data.left_eye is None or face_data.right_eye is None:
            return frame

        le = face_data.left_eye
        re = face_data.right_eye

        # Estimate gaze angle
        alpha, _ = self.estimate_gaze_angle(le.center, re.center)

        # Correct both eyes
        le_corrected = self.correct_eye(le, "L", alpha)
        re_corrected = self.correct_eye(re, "R", alpha)

        # Replace eye regions in frame (with border cropping)
        pc = self.pixel_cut
        frame[
            le.top_left[0] + pc[0] : le.top_left[0] + le.original_size[0] - pc[0],
            le.top_left[1] + pc[1] : le.top_left[1] + le.original_size[1] - pc[1],
        ] = (le_corrected[pc[0] : -pc[0], pc[1] : -pc[1]] * 255)

        frame[
            re.top_left[0] + pc[0] : re.top_left[0] + re.original_size[0] - pc[0],
            re.top_left[1] + pc[1] : re.top_left[1] + re.original_size[1] - pc[1],
        ] = (re_corrected[pc[0] : -pc[0], pc[1] : -pc[1]] * 255)

        return frame

    def close(self):
        """Release model resources."""
        self.model.close()
