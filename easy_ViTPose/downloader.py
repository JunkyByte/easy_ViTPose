import os
import argparse
from huggingface_hub import hf_hub_download


parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, required=True, choices=['torch', 'onnx', 'tensorrt'],
                    help='Model backend [torch, onnx, tensorrt]')
parser.add_argument('--model-name', type=str, required=True, choices=['s', 'b', 'l', 'h'],
                    help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H, YOLO-S: YOLO-S, YOLO-N: YOLO-N]')
parser.add_argument('--output', type=str, default='ckpts/',
                    help='Dir path for checkpoint output')
args = parser.parse_args()

REPO_ID = 'JunkyByte/easy_ViTPose'
is_yolo = 'YOLO' in args.model_name
if is_yolo:
    ext = {'onnx': '.onnx', 'torch': '.pt'}[args.backend]
    FILENAME = 'yolov5/yolov5' + args.model_name + ext
else:
    ext = {'tensorrt': '.engine', 'onnx': '.onnx', 'torch': '.pth'}[args.backend]
    FILENAME = os.path.join(args.backend, 'vitpose-25-' + args.model_name) + ext

print(f'>>> Downloading model {REPO_ID}/{FILENAME}')
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                             local_dir=args.output,
                             local_dir_use_symlinks=False)
