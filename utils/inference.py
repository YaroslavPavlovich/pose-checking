import cv2
import numpy as np

from models.pose_checking import predict, PoseChecking
from models.yolo import Detector
from utils.create_labels import normalize_points


def inference_frame(image: np.ndarray, yolo_model: Detector, model_points: PoseChecking):
    height, width, _ = image.shape
    results = yolo_model(image)
    for batch in results:
        for result in batch:
            points = result.keypoints.xy.cpu()
            new_points = normalize_points(points[0], width, height)
            if predict(model_points, new_points):
                color = (0, 255, 0)
            else:
                print('FOUND PERSON WITH NOT NORMAL POSE')
                color = (0, 0, 255)
            for point in points[0]:
                x = int(point[0])
                y = int(point[1])
                cv2.circle(image, (x, y), 5, color, 5)
    return image


def process_image(image_path: str, yolo_model: Detector, model_points: PoseChecking,
                  visualizing: bool, saving: bool, saving_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError('Wrong path to the source')
    image = inference_frame(image, yolo_model, model_points)
    if visualizing:
        cv2.imshow('result', image)
        cv2.waitKey()
    if saving:
        if not saving_path:
            raise Exception("Empty path for result dir")
        cv2.imwrite(saving_path + '/' + image_path.split('/')[-1], image)


def process_dir(sources: [], yolo_model: Detector, model_points: PoseChecking,
                visualizing: bool, saving: bool, saving_path=None):
    for source in sources:
        process_image(source, yolo_model, model_points, visualizing, saving, saving_path)


def process_video(video_path: str, yolo_model: Detector, model_points: PoseChecking,
                  visualizing: bool, saving: bool, saving_path=None):
    cap = cv2.VideoCapture(video_path)
    if saving:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if not saving_path:
            raise Exception("Empty path for result dir")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(saving_path + '/' + video_path.split('/')[-1], fourcc, 25.0, (width, height))
    while cap.isOpened():
        # Read frame from video
        ret, image = cap.read()
        if ret:
            image = inference_frame(image, yolo_model, model_points)
            if visualizing:
                cv2.imshow('result', image)
                cv2.waitKey(20)
            if saving:
                out.write(image)
        else:
            break
    if saving:
        out.release()
