FROM nvcr.io/nvidia/pytorch:24.07-py3
COPY . /easy_ViTPose
WORKDIR /easy_ViTPose/
ENV DEBIAN_FRONTEND=noninteractive

RUN pip uninstall -y $(pip list --format=freeze | grep opencv) && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
RUN pip install -e . && pip install -r requirements.txt && pip install -r requirements_gpu.txt

# OpenCV dependency
RUN apt-get update && apt-get install -y libgl1
