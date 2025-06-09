from pathlib import Path
import numpy as np
import joblib
import cv2
from ultralytics import YOLO


class HAR_Predictor:
    def __init__(self, yolo_model_path: str, classifier_path: str, label_map: dict):
        self.pose_model = YOLO(yolo_model_path)
        self.classifier = joblib.load(classifier_path)
        self.label_map = label_map

    def _extract_keypoints(self, result) -> np.ndarray:
        try:
            if not hasattr(result, 'keypoints') or result.keypoints is None:
                return None
            if result.keypoints.xy is None or len(result.keypoints.xy) == 0:
                return None

            xy = result.keypoints.xy[0].cpu().numpy()      # (17, 2)
            conf = result.keypoints.conf[0].cpu().numpy()  # (17,)

            img = cv2.imread(result.path)
            if img is None:
                print(f"❌ Failed to read image: {result.path}")
                return None
            h, w = img.shape[:2]
            xy = xy / np.array([[w, h]])

            if xy.shape != (17, 2) or conf.shape != (17,):
                return None

            vec = np.concatenate([xy.flatten(), conf])     # (51,)
            return vec.astype(np.float32)

        except Exception as e:
            print(f"❌ Error extracting keypoints: {e}")
            return None

    def predict(self, image_path: str) -> str:
        results = self.pose_model(image_path)
        keypoint_vector = self._extract_keypoints(results[0])
        if keypoint_vector is None:
            return "❌ No person detected or invalid keypoints"

        pred = self.classifier.predict(keypoint_vector.reshape(1, -1))[0]
        label = self.label_map.get(pred, f"Unknown ({pred})")
        return label


# ======================
# Only run below block if run as script (not when imported)
# ======================
if __name__ == "__main__":
    label_map = {
        0: "walking",
        1: "LOOKING_STRAIGHT",
        2: "STANDING",
        3: "jumping_climbing",
        4: "suspicious_look",
        5: "EXERCISE_BODY_SWING",
        6: "SITTING_STANDING",
        7: "fighting",
        8: "gesturing",
        9: "LOOKING_UP"
    }

    pipeline = HAR_Predictor(
        yolo_model_path="artifacts/weights/best.pt",
        classifier_path="artifacts/classifiers/mlp.pkl",
        label_map=label_map
    )

    test_dir = Path("artifacts/TEST")
    image_exts = [".jpg", ".jpeg", ".png"]

    for img_path in test_dir.glob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue
        result = pipeline.predict(str(img_path))
        print(f"{img_path.name}: {result}")
