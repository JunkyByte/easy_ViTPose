import os
import torch
import argparse

from easy_ViTPose.vit_models.model import ViTPose
from easy_ViTPose.vit_utils.util import infer_dataset_by_path, dyn_model_import


parser = argparse.ArgumentParser()
parser.add_argument('--model-ckpt', type=str, required=True,
                    help='The torch model that shall be used for conversion')
parser.add_argument('--model-name', type=str, required=True, choices=['s', 'b', 'l', 'h'],
                    help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
parser.add_argument('--output', type=str, default='ckpts/',
                    help='File (without extension) or dir path for checkpoint output')
parser.add_argument('--dataset', type=str, required=False, default=None,
                    help='Name of the dataset. If None it"s extracted from the file name. \
                          ["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')
args = parser.parse_args()


# Get dataset and model_cfg
dataset = args.dataset
if dataset is None:
    dataset = infer_dataset_by_path(args.model_ckpt)
assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], \
    'The specified dataset is not valid'
model_cfg = dyn_model_import(dataset, args.model_name)

# Convert to onnx and save
print('>>> Converting to ONNX')
CKPT_PATH = args.model_ckpt
C, H, W = (3, 256, 192)

model = ViTPose(model_cfg)

ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
if 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']

model.load_state_dict(ckpt)
model.eval()

input_names = ["input_0"]
output_names = ["output_0"]

device = next(model.parameters()).device
inputs = torch.randn(1, C, H, W).to(device)

dynamic_axes = {'input_0': {0: 'batch_size'},
                'output_0': {0: 'batch_size'}}

out_name = os.path.basename(args.model_ckpt).replace('.pth', '.onnx')
if not os.path.isdir(args.output):
    out_name = os.path.basename(args.output)
output_onnx = os.path.join(os.path.dirname(args.output), out_name)

torch_out = torch.onnx.export(model, inputs, output_onnx, export_params=True, verbose=False,
                              input_names=input_names, output_names=output_names,
                              dynamic_axes=dynamic_axes)
print(f">>> Saved at: {os.path.abspath(output_onnx)}")
print('=' * 80)
print()

try:
    import torch_tensorrt
except ModuleNotFoundError:
    print('>>> TRT module not found, skipping')
    import sys
    sys.exit()

# From yolo convert script, onnx -> trt
print('>>> Converting to TRT')
trt_ts_module = torch_tensorrt.compile(model,
    # If the inputs to the module are plain Tensors, specify them via the `inputs` argument:
    inputs = [
        torch_tensorrt.Input( # Specify input object with shape and dtype
            shape=[1, C, H, W],
            dtype=torch.float32
        )
    ],

    # TODO: ADD Datatype for inference. Allowed options torch.(float|half|int8|int32|bool)
    enabled_precisions = {torch.float32}, # half Run with FP16
    workspace_size = 1 << 28
)

# Export
output_trt = output_onnx.replace('.onnx', '.engine')

input_names = ["input_0"]
output_names = ["output_0"]

device = next(model.parameters()).device
torch.jit.save(trt_ts_module, output_trt) # save the TRT embedded Torchscript

print(f">>> Saved at: {os.path.abspath(output_trt)}")
