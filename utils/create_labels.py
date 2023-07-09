import os

import cv2
import torch

from models.yolo import Detector


def make_points(path: str, model: Detector):
    points_array = []
    files = os.listdir(path)
    for file in files:
        img = cv2.imread(path + file)
        height, width, _ = img.shape
        results = model(img)
        for batch in results:
            for result in batch:
                points = result.keypoints.xy.cpu()
                array = []
                for point in points[0]:
                    array.append([float(int(point[0]) / width), float(int(point[1]) / height)])
                points_array.append(array)
    return points_array


def mix_data(normal_points: [], not_normal_points: []) -> ([], []):
    train_data = []
    train_labels = []
    while len(normal_points) != 0 or len(not_normal_points) != 0:
        if len(normal_points) > 0:
            train_data.append(normal_points.pop())
            train_labels.append([1])
        if len(not_normal_points) > 0:
            train_data.append(not_normal_points.pop())
            train_labels.append([0])
    return train_data, train_labels


def normalize_points(points: [], width: int, height: int):
    new_coord = []
    for point in points:
        new_coord.append(float(int(point[0]) / width))
        new_coord.append(float(int(point[1]) / height))
    return torch.tensor(new_coord, dtype=torch.float32)
