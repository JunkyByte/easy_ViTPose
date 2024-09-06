#! /usr/bin/env bash
DIR=$(dirname "$0")
wget -O $DIR/vitpose-l-ap10k.onnx https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/ap10k/vitpose-l-ap10k.onnx
wget -O $DIR/yolov8l.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8l.pt
