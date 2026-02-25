"""
Pose estimation wrapper using MediaPipe BlazePose.

DISPOSABLE — thin wrapper around MediaPipe's PoseLandmarker. Extracts
33-point pose landmarks from video frames. Does NO analysis — just
extraction and packaging into LandmarkTimeSeries.

Requires:
    pip install -e ".[pose]"
"""

import numpy as np

from musical_perception.types import LandmarkTimeSeries

# MediaPipe landmark indices used by precision/dynamics.py.
# Defined here for reference; dynamics.py imports what it needs.
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def load_model():
    """
    Load the MediaPipe PoseLandmarker model.

    Downloads the model on first use (~30MB). Returns a configured
    PoseLandmarker in VIDEO mode.

    Returns:
        mediapipe.tasks.vision.PoseLandmarker ready for extract_landmarks().
    """
    try:
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.vision import RunningMode
        from mediapipe.tasks.python import BaseOptions
    except ImportError as e:
        raise ImportError(
            "mediapipe is not installed. Install with:\n"
            "  pip install -e '.[pose]'"
        ) from e
    import urllib.request
    import os
    import tempfile

    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    cache_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "pose_landmarker_lite.task")

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
    )
    return PoseLandmarker.create_from_options(options)


def extract_landmarks(
    landmarker,
    video_path: str,
) -> LandmarkTimeSeries:
    """
    Extract pose landmarks from every frame of a video.

    Args:
        landmarker: PoseLandmarker from load_model().
        video_path: Path to video file (.mov, .mp4, etc.)

    Returns:
        LandmarkTimeSeries with shape (N, 33, 3) landmarks array.
        Frames where detection fails are filled with NaN.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    import mediapipe as mp

    fps = cap.get(cv2.CAP_PROP_FPS)

    all_landmarks = []
    all_timestamps = []
    detected_count = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000.0 / fps)
        timestamp_s = frame_idx / fps

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            # Take first detected person
            person = result.pose_landmarks[0]
            frame_data = np.array([[lm.x, lm.y, lm.z] for lm in person])
            detected_count += 1
        else:
            frame_data = np.full((33, 3), np.nan)

        all_landmarks.append(frame_data)
        all_timestamps.append(timestamp_s)
        frame_idx += 1

    cap.release()

    n_frames = len(all_timestamps)
    detection_rate = detected_count / n_frames if n_frames > 0 else 0.0

    return LandmarkTimeSeries(
        timestamps=np.array(all_timestamps),
        landmarks=np.array(all_landmarks),
        fps=fps,
        detection_rate=detection_rate,
    )
