import cv2
import os
import time

from huggingface_hub import hf_hub_download

from easy_ViTPose import VitInference
from easy_ViTPose.vit_utils.inference import VideoReader

REPO_ID = 'JunkyByte/easy_ViTPose'

MODEL_TYPE = 'torch'  # ["onnx", "torch"]
MODEL_SIZE = 's'  # ['s', 'b', 'l', 'h']
YOLO_TYPE = 'torch'  # ['onnx', 'torch']
YOLO_SIZE = 's'  # ['s', 'n']


def process_video(model, video_path):

    output_path = 'results.mp4'

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()

    cap.release()

    assert ret
    assert fps > 0

    output_size = frame.shape[:2][::-1]

    out_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        output_size
    )

    time_start = time.time()

    reader = VideoReader(video_path)

    frame_count = 0

    for frame in reader:
        print(frame_count + 1, 'out of', total_frames)

        model.inference(frame)

        img = model.draw(show_yolo=True)

        out_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # cv2.imshow('image', img[..., ::-1])
        # cv2.waitKey(0)
        # break

        frame_count += 1

    out_writer.release()

    print('seconds elapsed', time.time() - time_start)


def main():
    video_path = '/Users/jimmybuffi/Downloads/Synergy Video Samples 2/synergy_right_1.mp4'

    ext = {'tensorrt': '.engine', 'onnx': '.onnx', 'torch': '.pth'}[MODEL_TYPE]
    ext_yolo = {'onnx': '.onnx', 'torch': '.pt'}[YOLO_TYPE]

    filename = os.path.join(MODEL_TYPE, 'vitpose-25-' + MODEL_SIZE) + ext

    filename_yolo = 'yolov5/yolov5' + YOLO_SIZE + ext_yolo

    print(f'Downloading model {REPO_ID}/{filename}')

    model_path = hf_hub_download(repo_id=REPO_ID, filename=filename)

    yolo_path = hf_hub_download(repo_id=REPO_ID, filename=filename_yolo)

    # If you want to use MPS (on new macbooks) use the torch checkpoints for both ViTPose and Yolo
    # If device is None will try to use cuda -> mps -> cpu (otherwise specify 'cpu', 'mps' or 'cuda')
    model = VitInference(
        model_path, yolo_path, model_name='s', yolo_size=320, is_video=True, yolo_step=1, device='cpu'
    )

    process_video(model, video_path)


if __name__ == '__main__':
    main()
