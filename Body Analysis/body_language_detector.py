"""
Body Language Emotion Detection using MediaPipe Tasks API, OpenCV, and TensorFlow Lite

This module detects body language emotions from a video file (.mp4).
It uses a TensorFlow Lite model (body_language.tflite) for classification
and the new MediaPipe Tasks API (PoseLandmarker + FaceLandmarker) to
extract pose + face landmarks as features.

The model was trained on 9 emotion classes:
    Angry, Confused, Excited, Happy, Pain, Sad, Surprised, Tension

Usage:
    python body_language_detector.py --video VIDEO_PATH [--model MODEL_PATH]

Dependencies:
    - mediapipe  (>= 0.10.x, Tasks API)
    - opencv-python
    - numpy
    - tensorflow  (or tflite-runtime)
"""

import argparse
import os
import urllib.request
import warnings
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite
    _Interpreter = tflite.Interpreter


CLASS_NAMES = [
    "Angry",
    "Confused",
    "Excited",
    "Happy",
    "Pain",
    "Sad",
    "Surprised",
    "Tension",
]


#MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


#model download URLs (Google's official .task bundles)
_MODEL_DIR = Path(__file__).resolve().parent / "mediapipe_models"

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

_POSE_MODEL_PATH = _MODEL_DIR / "pose_landmarker_lite.task"
_FACE_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"

# The old Holistic model produced:
#   33 pose landmarks x 4 (x, y, z, visibility) = 132
#   468 face landmarks x 4 (x, y, z, visibility) = 1872
#   Total = 2004 features
# The new FaceLandmarker gives 478 face landmarks (no visibility field),
# so we truncate to 468 and set visibility=0.0 to match the trained model.
_OLD_FACE_LANDMARK_COUNT = 468
_POSE_LANDMARK_COUNT = 33


def _ensure_models():
    """Download MediaPipe .task models if they don't already exist."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for url, dest in [(_POSE_MODEL_URL, _POSE_MODEL_PATH),
                      (_FACE_MODEL_URL, _FACE_MODEL_PATH)]:
        if not dest.exists():
            print(f"  Downloading {dest.name}…", flush=True)
            urllib.request.urlretrieve(url, str(dest))
            print(f"  Saved → {dest}", flush=True)


def _create_pose_landmarker() -> PoseLandmarker:
    """Create a PoseLandmarker in VIDEO mode."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_POSE_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _create_face_landmarker() -> FaceLandmarker:
    """Create a FaceLandmarker in VIDEO mode."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_FACE_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def extract_landmarks(pose_result, face_result) -> np.ndarray | None:
    """
    Extract pose and face landmark coordinates from the new Tasks API results
    and return them as a single flat numpy array (float32) matching the shape
    expected by the TFLite emotion model (trained on the old Holistic output).

    Returns None when either pose or face landmarks are missing.
    """
    if not pose_result.pose_landmarks or not face_result.face_landmarks:
        return None

    pose_landmarks = pose_result.pose_landmarks[0]  # first person
    face_landmarks = face_result.face_landmarks[0]  # first face

    if len(pose_landmarks) < _POSE_LANDMARK_COUNT:
        return None
    if len(face_landmarks) < _OLD_FACE_LANDMARK_COUNT:
        return None

    #pose: 33 landmarks x [x, y, z, visibility]
    pose_row = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks[:_POSE_LANDMARK_COUNT]],
        dtype=np.float32,
    ).flatten()

    #face: 468 landmarks x [x, y, z, visibility]
    # new API has no visibility, so we default it to 0.0 (as the old API did for face)
    face_row = np.array(
        [[lm.x, lm.y, lm.z, 0.0] for lm in face_landmarks[:_OLD_FACE_LANDMARK_COUNT]],
        dtype=np.float32,
    ).flatten()

    return np.concatenate([pose_row, face_row])


#TFLite inference helper
class EmotionClassifier:
    """Thin wrapper around a TFLite interpreter for emotion prediction."""

    def __init__(self, model_path: str):
        self.interpreter = _Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

    def predict(self, features: np.ndarray):
        """
        Run inference on a 1D feature vector.

        Returns
        -------
        class_name : str
            The predicted emotion label.
        probabilities : np.ndarray
            Softmax probability array over all classes.
        """
        input_data = features.astype(np.float32).reshape(
            self._input_details[0]["shape"]
        )

        self.interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self.interpreter.invoke()
        probabilities = self.interpreter.get_tensor(
            self._output_details[0]["index"]
        )[0]

        class_idx = int(np.argmax(probabilities))
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
        return class_name, probabilities


def run_analysis(video_path: str, model_path: str = None) -> dict:
    """
    Headless analysis: process every frame and return structured results.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    model_path : str, optional
        Path to the TFLite model. Defaults to body_language.tflite next to this script.

    Returns
    -------
    dict
        {
            "frames": [{"timestamp_s": float, "emotion": str, "confidence": float}, ...],
            "summary": {
                "total_frames_analyzed": int,
                "dominant_emotion": str,
                "dominant_emotion_pct": float,
                "average_confidence": float,
                "emotion_distribution": {emotion: percentage, ...},
                "duration_s": float,
            }
        }
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "body_language.tflite")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _ensure_models()

    classifier = EmotionClassifier(model_path)
    warnings.filterwarnings("ignore")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    frame_idx = 0

    pose_landmarker = _create_pose_landmarker()
    face_landmarker = _create_face_landmarker()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            row = extract_landmarks(pose_result, face_result)
            if row is not None:
                emotion, probs = classifier.predict(row)
                confidence = float(probs[int(np.argmax(probs))])
                frames.append({
                    "timestamp_s": round(frame_idx / fps, 3),
                    "emotion": emotion,
                    "confidence": round(confidence, 4),
                })

            frame_idx += 1
    finally:
        pose_landmarker.close()
        face_landmarker.close()
        cap.release()

    #build summary
    total = len(frames)
    duration = frame_idx / fps if fps > 0 else 0.0

    if total > 0:
        from collections import Counter
        emotion_counts = Counter(f["emotion"] for f in frames)
        dominant = emotion_counts.most_common(1)[0]
        avg_conf = sum(f["confidence"] for f in frames) / total
        distribution = {e: round(c / total * 100, 1) for e, c in emotion_counts.items()}
    else:
        dominant = ("Unknown", 0)
        avg_conf = 0.0
        distribution = {}

    return {
        "frames": frames,
        "summary": {
            "total_frames_analyzed": total,
            "dominant_emotion": dominant[0],
            "dominant_emotion_pct": round(dominant[1] / max(total, 1) * 100, 1),
            "average_confidence": round(avg_conf, 4),
            "emotion_distribution": distribution,
            "duration_s": round(duration, 2),
        },
    }


def draw_landmarks(image, pose_result, face_result):
    """Draw pose landmarks on the frame using OpenCV (simplified viz)."""
    h, w, _ = image.shape

    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (245, 117, 66), -1)

    if face_result.face_landmarks:
        for lm in face_result.face_landmarks[0][:_OLD_FACE_LANDMARK_COUNT]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 1, (80, 256, 121), -1)


def draw_prediction_overlay(image, body_language_class, body_language_prob):
    """Draw the predicted class + probability overlay on the frame."""
    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

    cv2.putText(
        image,
        "CLASS",
        (95, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        body_language_class.split(" ")[0],
        (90, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "PROB",
        (15, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(round(float(body_language_prob[np.argmax(body_language_prob)]), 2)),
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_detection(model_path: str, video_path: str):
    """Interactive detection with OpenCV GUI display."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _ensure_models()

    classifier = EmotionClassifier(model_path)
    warnings.filterwarnings("ignore")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose_landmarker = _create_pose_landmarker()
    face_landmarker = _create_face_landmarker()

    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            #draw landmarks
            draw_landmarks(frame, pose_result, face_result)

            try:
                row = extract_landmarks(pose_result, face_result)
                if row is not None:
                    body_language_class, body_language_prob = classifier.predict(row)
                    draw_prediction_overlay(frame, body_language_class, body_language_prob)
            except Exception:
                pass

            cv2.imshow("Body Language Detection", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            frame_idx += 1
    finally:
        pose_landmarker.close()
        face_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Body language detection from a video file using MediaPipe Tasks + TFLite."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input .mp4 video file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "body_language.tflite"),
        help="Path to the TFLite model file (default: body_language.tflite).",
    )
    args = parser.parse_args()

    run_detection(model_path=args.model, video_path=args.video)


if __name__ == "__main__":
    main()
