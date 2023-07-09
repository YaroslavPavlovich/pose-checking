import numpy as np
from ultralytics import YOLO


class Detector(YOLO):
    def __init__(self, model_path='yolov8s-pose.pt'):
        super().__init__(model=model_path)

    def pose_predict(self, image: np.ndarray):
        results = self.model(image)
        points = results[0][0].keypoints.xy.cpu()
        print(points)


