from time import time
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

# 設定方法
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# 人臉偵測設定
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./models/face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)


# 執行人臉偵測
with FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)               # 讀取攝影鏡頭
    start_time = time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        # Calculate the timestamp (in milliseconds) from the start
        # VIDEO mode requires a strictly increasing timestamp
        frame_timestamp_ms = int((time() - start_time) * 1000)

        w = frame.shape[1]
        h = frame.shape[0]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        face_landmarks_list = face_landmarker_result.face_landmarks
        annotated_image = np.copy(frame)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles
                .get_default_face_mesh_tesselation_style())
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles
                .get_default_face_mesh_contours_style())
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles
                .get_default_face_mesh_iris_connections_style())
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles
                .get_default_face_mesh_iris_connections_style())

        cv2.imshow('oxxostudio', annotated_image)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
