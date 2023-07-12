import argparse
import json
import os
import zipfile

import gdown
import torch

from entity.data import Data
from models.pose_checking import PoseChecking, train_model
from utils.create_labels import make_points, mix_data
from models.yolo import Detector


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=str, default='yolov8s-pose.pt', help='initial weights path for yolov8')
    parser.add_argument('--result-model', type=str, default='pose_checking.pt', help='result training weights path')
    parser.add_argument('--dataset', type=str, default='pose/train', help='path for dataset')
    parser.add_argument('--labels', type=str, default='points.json', help='folder of labels with points of pose')
    parser.add_argument('--normal-frames', type=str, default='normal', help='folder of photos with normal pose')
    parser.add_argument('--not-normal-frames', type=str, default='not_normal',
                        help='folder of photos with not normal pose')
    parser.add_argument('--dataset-url', type=str,
                        default='https://drive.google.com/file/d/1f60Jb8GIF4keTod3Z3yoBVc6xHrbbJ1G/view?usp=sharing',
                        help='folder of labels with not normal pose')
    return parser.parse_args()


def main(opt):
    name_dir = opt.dataset
    # Checking dataset
    if not os.path.isdir(opt.dataset):
        name_file = gdown.download(url=opt.dataset_url, fuzzy=True)
        with zipfile.ZipFile(name_file, 'r') as zip_ref:
            zip_ref.extractall()
        name_dir = name_file.replace('.zip', '')
    if not os.path.isfile(name_dir + '/' + opt.labels):
        # Load a model
        yolo_model = Detector(opt.yolo_model)
        normal_points = make_points(path=name_dir + '/' + opt.normal_frames + '/', model=yolo_model)
        not_normal_points = make_points(path=name_dir + '/' + opt.not_normal_frames + '/', model=yolo_model)
        data = Data(normal_points, not_normal_points)
        with open(name_dir + '/' + opt.labels, 'w') as f:
            json.dump(data.__dict__, f)
    with open(name_dir + '/' + opt.labels, 'r') as f:
        data = json.load(f)
    train_data, train_labels = mix_data(data['normal_points'], data['not_normal_points'])
    model_points = PoseChecking()

    # Make tensors
    train_data = torch.tensor(train_data, dtype=torch.float32).view(len(train_data), -1)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    # Train model
    train_model(model_points, train_data, train_labels, epochs=300)
    model_scripted = torch.jit.script(model_points)  # Export to TorchScript
    model_scripted.save(opt.result_model)  # Save


if __name__ == '__main__':
    main(parse_arguments())
