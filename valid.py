import argparse
import json
import os
import zipfile

import gdown
import torch

from entity.data import Data
from models.pose_checking import predict, PoseChecking
from models.yolo import Detector
from utils.create_labels import make_points


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=str, default='yolov8s-pose.pt', help='initial weights path for yolov8')
    parser.add_argument('--pose-checking-model', type=str,
                        default='pose_checking.pt', help='result training weights path')
    parser.add_argument('--dataset', type=str, default='pose/val', help='path for dataset')
    parser.add_argument('--labels', type=str, default='points.json', help='folder of labels with points of pose')
    parser.add_argument('--normal-frames', type=str, default='normal', help='folder of photos with normal pose')
    parser.add_argument('--not-normal-frames', type=str, default='not_normal',
                        help='folder of photos with not normal pose')
    parser.add_argument('--dataset-url', type=str,
                        default='https://drive.google.com/file/d/1f60Jb8GIF4keTod3Z3yoBVc6xHrbbJ1G/view?usp=sharing',
                        help='folder of labels with not normal pose')
    parser.add_argument('--model-url', type=str,
                        default='https://drive.google.com/file/d/1unmGjSGOaRRoUHrlew7DNSqwUTfxXnnO/view?usp=sharing',
                        help='URL of pretrained model')
    return parser.parse_args()


def count_answers(model_points: PoseChecking, array_points: []):
    not_normal = 0
    normal = 0
    for points in array_points:
        new_coord = []
        for point in points:
            new_coord.append(point[0])
            new_coord.append(point[1])
        new_coord = torch.tensor(new_coord, dtype=torch.float32)
        if predict(model_points, new_coord):
            normal += 1
        else:
            not_normal += 1
    return normal, not_normal


def main(opt):
    name_dir = opt.dataset
    # Checking dataset
    if not os.path.isdir(opt.dataset):
        name_file = gdown.download(url=opt.dataset_url, fuzzy=True)
        with zipfile.ZipFile(name_file, 'r') as zip_ref:
            zip_ref.extractall()
    if not os.path.isfile(name_dir + '/' + opt.labels):
        # Load a model
        yolo_model = Detector(opt.yolo_model)
        normal_points = make_points(path=name_dir + '/' + opt.normal_frames + '/', model=yolo_model)
        not_normal_points = make_points(path=name_dir + '/' + opt.not_normal_frames + '/', model=yolo_model)
        data = Data(normal_points, not_normal_points)
        with open(name_dir + '/' + opt.labels, 'w') as f:
            json.dump(data.__dict__, f)
    if not os.path.isfile(opt.pose_checking_model):
        if 'pose_checking.pt' != opt.pose_checking_model:
            raise FileNotFoundError('No model file: ' + opt.pose_checking_model)
        gdown.download(url=opt.model_url, fuzzy=True)
    with open(name_dir + '/' + opt.labels, 'r') as f:
        data = json.load(f)
    model_points = torch.jit.load(opt.pose_checking_model)
    normal_points = data['normal_points']
    not_normal_points = data['not_normal_points']
    negative = 0
    positive = 0
    normal, not_normal = count_answers(model_points, normal_points)
    positive += normal
    negative += not_normal
    normal, not_normal = count_answers(model_points, not_normal_points)
    positive += not_normal
    negative += normal
    accuracy = positive / (positive + negative)
    print('Accuracy of pose checking: ' + str(accuracy))


if __name__ == '__main__':
    main(parse_arguments())
