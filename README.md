# Pose checking
A project for assessing a person's pose. The objective of the project is to predict whether a person's pose is normal or abnormal.

A project using [yolov8](https://github.com/ultralytics/ultralytics) to evaluate a pose. A project using a custom model to test a person's pose based on points from yolov8-pose.

<div>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/drive/1AzPYK9b-nV_xHuyCy2Ndc4SZzwycn8V3#scrollTo=Ts48larfN8S8" alt="Open In Colab"></a>
</div>

## Install
```
git clone https://github.com/YaroslavPavlovich/pose-checking
cd pose-checking
python3 -m venv venv
pip install -r requirements.txt
```

## Training
Example for start train script:
```
python3 train.py
```

## Validation
Example for start validation script:
```
python3 valid.py
```

## Inference
Example for start inference script:
```
python3 detect.py --save-results --source data/example1.jpg
```

## Dataset
You can use custom dataset. Example of using dataset is in [train.py](train.py) and [valid.py](valid.py). [The dataset of this project.](https://drive.google.com/file/d/1f60Jb8GIF4keTod3Z3yoBVc6xHrbbJ1G/view?usp=sharing)

## Model
### Pose-estimation
Project uses yolov8 for pose estimation model. In each script you can choose model, that you need in '--yolo-model' param.
Models: yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt
Example:
```
python3 detect.py --yolo-model yolov8n-pose.pt
```

### Pose-checking
Project uses custom model for pose checking. Model is in [pose_checking.py](models%2Fpose_checking.py)
You can train custom model with [train.py](train.py) or download [pretrained model](https://drive.google.com/file/d/1unmGjSGOaRRoUHrlew7DNSqwUTfxXnnO/view?usp=sharing).
