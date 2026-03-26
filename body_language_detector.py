"""
Body Language Emotion Detection using MediaPipe, OpenCV, and tensorflow

This module detects body language emotions from a video file (.mp4), it uses a TensorFlow model (body_language.tflite) for classification
and MediaPipe Holistic to extract pose + face landmarks as features

The model was trained on 9 emotion classes:
    Angry, Confused, Excited, Happy, Pain, Sad, Surprised, Tension


Usage:
    python body_language_detector.py --video VIDEO_PATH [--model MODEL_PATH]

Dependencies:
    - mediapipe
    - opencv-python
    - numpy
    - tensorflow  (or tflite-runtime)
"""

import argparse
import os
import warnings

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


#MediaPipe helpers
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def draw_landmarks(image, results):
    """Draw face, hand, and pose landmarks on the frame."""

    #face mesh contours
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )

    #right hand
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )

    #left hand
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )

    #pose
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def extract_landmarks(results):
    """
    Extract pose and face landmark coordinates from a MediaPipe Holistic
    result and return them as a single flat numpy array (float32).

    Returns ``None`` when either pose or face landmarks are missing.
    """
    if results.pose_landmarks is None or results.face_landmarks is None:
        return None

    pose = results.pose_landmarks.landmark
    pose_row = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose],
        dtype=np.float32,
    ).flatten()

    face = results.face_landmarks.landmark
    face_row = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in face],
        dtype=np.float32,
    ).flatten()

    return np.concatenate([pose_row, face_row])


#tensorflow inference helper
class EmotionClassifier:
    """Thin wrapper around a TF interpreter for emotion prediction"""

    def __init__(self, model_path: str):
        self.interpreter = _Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

    def predict(self, features: np.ndarray):
        """
        Run inference on a 1D feature vector

        Returns
        -------
        class_name : str
            The predicted emotion label.
        probabilities : np.ndarray
            Softmax probability array over all classes.
        """
        #ensure correct shape and dtype
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


#overlay helpers #TODO:delete later
def draw_prediction_overlay(image, body_language_class, body_language_prob, results):
  
    h, w, _ = image.shape

    # Label near the left ear
    coords = tuple(
        np.multiply(
            np.array(
                (
                    results.pose_landmarks.landmark[
                        mp_holistic.PoseLandmark.LEFT_EAR
                    ].x,
                    results.pose_landmarks.landmark[
                        mp_holistic.PoseLandmark.LEFT_EAR
                    ].y,
                )
            ),
            [w, h],
        ).astype(int)
    )

    cv2.rectangle(
        image,
        (coords[0], coords[1] + 5),
        (coords[0] + len(body_language_class) * 20, coords[1] - 30),
        (245, 117, 16),
        -1,
    )
    cv2.putText(
        image,
        body_language_class,
        coords,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

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
   if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    classifier = EmotionClassifier(model_path)

    warnings.filterwarnings("ignore")

    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            #(BGR -> RGB)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            #recolor back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_landmarks(image, results)

            try:
                row = extract_landmarks(results)
                if row is not None:
                    body_language_class, body_language_prob = classifier.predict(row)

                    draw_prediction_overlay(
                        image, body_language_class, body_language_prob, results
                    )
            except Exception:
                pass

            cv2.imshow("Body Language Detection", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()



def main():
    parser = argparse.ArgumentParser(
        description="Body language detection from a video file using MediaPipe + TFLite."
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
