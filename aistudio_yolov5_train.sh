#!/bin/bash
python aistudio_yolov5_data.py
echo "kfold 0";
python train.py --img 1024 --batch 4 --epochs 100 --data ./data/wheat0.yaml --cfg ./models/yolov5x.yaml --name yolov5x_fold0 --weights ./weights/yolov5x.pt;
echo "kfold 1";
python train.py --img 1024 --batch 4 --epochs 100 --data ./data/wheat1.yaml --cfg ./models/yolov5x.yaml --name yolov5x_fold1 --weights ./weights/yolov5x.pt;
echo "kfold 2";
python train.py --img 1024 --batch 4 --epochs 100 --data ./data/wheat2.yaml --cfg ./models/yolov5x.yaml --name yolov5x_fold2 --weights ./weights/yolov5x.pt;
echo "kfold 3";
python train.py --img 1024 --batch 4 --epochs 100 --data ./data/wheat3.yaml --cfg ./models/yolov5x.yaml --name yolov5x_fold3 --weights ./weights/yolov5x.pt;
echo "kfold 4";
python train.py --img 1024 --batch 4 --epochs 100 --data ./data/wheat4.yaml --cfg ./models/yolov5x.yaml --name yolov5x_fold4 --weights ./weights/yolov5x.pt;