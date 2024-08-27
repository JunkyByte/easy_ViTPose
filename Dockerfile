FROM nvcr.io/nvidia/pytorch:24.07-py3
COPY . /easy_ViTPose
WORKDIR /easy_ViTPose/
ENV DEBAIN_FRONTEND=noninteractive
RUN pip uninstall -y $(pip list --format=freeze | grep opencv) && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
RUN pip install -e . && pip install -r requirements.txt && pip install -r requirements_gpu.txt
# OpenCV dependencies
RUN apt-get update && apt-get install -y libgl1

# docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ./models:/models -v ~/cats:/cats easy_vitpose python inference.py --det-class cat --input /cats/image.jpg --output-path /cats --save-img --model /models/vitpose-l-coco.onnx --yolo /models/yolov8l.pt
