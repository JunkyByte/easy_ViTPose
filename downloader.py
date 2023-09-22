import os
import argparse
from huggingface_hub import hf_hub_download
from easy_ViTPose.vit_utils.util import infer_dataset_by_path


parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, required=True, choices=['torch', 'onnx', 'tensorrt'],
                    help='Model backend [torch, onnx, tensorrt]')
parser.add_argument('--model-name', type=str, required=True, choices=['s', 'b', 'l', 'h'],
                    help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H, YOLO-S: YOLO-S, YOLO-N: YOLO-N]')
parser.add_argument('--output', type=str, default='ckpts/',
                    help='Dir path for checkpoint output')
parser.add_argument('--dataset', type=str, required=False, default=None,
                    help='Name of the dataset. If None it"s extracted from the file name. \
                          ["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')
args = parser.parse_args()


dataset = args.dataset
if dataset is None:
    dataset = infer_dataset_by_path(args.model_ckpt)
assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], \
    'The specified dataset is not valid'

REPO_ID = 'JunkyByte/easy_ViTPose'
is_yolo = 'YOLO' in args.model_name
if is_yolo:
    ext = {'onnx': '.onnx', 'torch': '.pt'}[args.backend]
    FILENAME = 'yolov5/yolov5' + args.model_name + ext
else:
    ext = {'tensorrt': '.engine', 'onnx': '.onnx', 'torch': '.pth'}[args.backend]
    FILENAME = os.path.join(f'{args.backend}/{dataset}/', 'vitpose-25-' + args.model_name) + ext

print(f'>>> Downloading model {REPO_ID}/{FILENAME}')
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                             local_dir=args.output,
                             local_dir_use_symlinks=False)
