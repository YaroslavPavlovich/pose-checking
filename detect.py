import argparse
import os
from pathlib import Path

import torch

from models.yolo import Detector
from utils.inference import process_image, process_video, process_dir
from venv.bin import gdown

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=str, default='yolov8s-pose.pt', help='weights path for yolov8')
    parser.add_argument('--pose-checking-model', type=str, default='pose_checking.pt',
                        help='weights path for pose checking')
    parser.add_argument('--source', type=str, default='data', help='initial weights path for yolov8')
    parser.add_argument('--result-dir', type=str, default='results', help='directory for results')
    parser.add_argument('--save-results', action='store_true', help='save results')
    parser.add_argument('--show-results', action='store_true', help='show results')
    return parser.parse_args()


def check_dir(path: str):
    try:
        dir_list = os.listdir(path)
    except FileNotFoundError:
        raise Exception("Can't find files or type of source is not available")
    sources = []
    for file in dir_list:
        new_source = path + '/' + file
        if Path(new_source).suffix[1:] in IMG_FORMATS:
            sources.append(new_source)
    return sources


def main(opt):
    yolo_model = Detector(opt.yolo_model)
    model_points = torch.jit.load(opt.pose_checking_model)
    source = str(opt.source)
    if not os.path.isfile(opt.pose_checking_model):
        if 'pose_checking.pt' != opt.pose_checking_model:
            raise FileNotFoundError('No model file: ' + opt.pose_checking_model)
        gdown.download(url=opt.dataset_url, fuzzy=True)

    is_vid = Path(source).suffix[1:] in VID_FORMATS
    is_img = Path(source).suffix[1:] in IMG_FORMATS

    if opt.result_dir:
        Path(opt.result_dir).mkdir(parents=True, exist_ok=True)  # make dir

    if is_img:
        process_image(source, yolo_model, model_points, opt.show_results, opt.save_results, opt.result_dir)
    elif is_vid:
        process_video(source, yolo_model, model_points, opt.show_results, opt.save_results, opt.result_dir)
    else:
        sources = check_dir(source)
        if len(sources) <= 0:
            raise FileNotFoundError("Can't find files")
        process_dir(sources, yolo_model, model_points, opt.show_results, opt.save_results, opt.result_dir)


if __name__ == '__main__':
    main(parse_arguments())
