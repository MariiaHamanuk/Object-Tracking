"""Open-vocabulary detector (YOLO-World, ultralytics) as an extra source of re-localization candidates.

Only called on the frames where V-JEPA re-localization triggers, prompted with the video's LaSOT class
name, so it proposes whole objects with real boxes instead of V-JEPA token centres with the last box
size. Using the class name is extra information LaSOT provides (like its language annotation); OSTrack
alone does not use it.
"""
from __future__ import annotations

import cv2
import numpy as np

# LaSOT class -> a phrase YOLO-World's text encoder understands (the rest are used as they are)
PROMPTS = {
    "rubicCube": "rubik's cube", "licenseplate": "license plate", "electricfan": "electric fan",
    "gametarget": "target", "sepia": "cuttlefish", "pool": "billiard ball", "yoyo": "yo-yo",
    "surfboard": "surfboard", "skateboard": "skateboard", "volleyball": "volleyball",
    "basketball": "basketball", "racing": "racing car", "hand": "hand", "person": "person",
}


def prompt_for(lasot_class: str) -> str:
    return PROMPTS.get(lasot_class, lasot_class)


class YOLODetector:
    def __init__(self, model: str = "yolov8s-worldv2.pt", conf: float = 0.05, max_det: int = 20):
        from ultralytics import YOLO

        self.model, self.conf, self.max_det = YOLO(model), conf, max_det
        self.prompt = None

    def set_class(self, lasot_class: str):
        p = prompt_for(lasot_class)
        if p != self.prompt:
            self.model.set_classes([p])
            self.prompt = p

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """(N, 4) boxes x, y, w, h in pixels, most confident first."""
        res = self.model.predict(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), conf=self.conf,
                                 max_det=self.max_det, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        return np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]
