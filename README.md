# easy_ViTPose
<p align="center">
<img src="https://user-images.githubusercontent.com/24314647/236082274-b25a70c8-9267-4375-97b0-eddf60a7dfc6.png" width=375> easy_ViTPose
</p>

## Accurate 2d human pose estimation, finetuned on 25 keypoints COCO skeleton + feet  
### Easy to use SOTA `ViTPose` [Y. Xu et al., 2022] models for fast inference.  

### Results

![resimg](https://user-images.githubusercontent.com/24314647/236281199-98e45ab5-2a18-45b7-ba5c-36bdec4450f4.png)
(Model - ViTPose-b)

https://user-images.githubusercontent.com/24314647/236281644-344ccc0e-a5ea-49a3-9671-d221153d56a6.mov

(Credits: https://www.youtube.com/watch?v=p-rSdt0aFuw&pp=ygUhZGFuY2UgZXZvbHV0aW9uIGZyb20gMTk1MCB0byAyMDE5)  
(s - small, b - base, l - large, h - huge)

#### Benchmark:
Run on `GTX1080ti`, consider that tensorrt > onnx > torch.  
These benchmarks are relative to `ViTPose` models, they do not consider Yolo detection that is done before `ViTPose` inference.  
Tensorrt:  
- ViTPose-s: ~250 fps
- ViTPose-b: ~125 fps
- ViTPose-l: ~45 fps
- ViTPose-h: ~24.5 fps
(these are relative to single person pose estimation)

### Features
- Image / Video / Webcam support
- Torch / ONNX / Tensorrt models
- 4 ViTPose architectures with different sizes
- cpu / gpu support

## Usage
- Download the models from [Huggingface](https://huggingface.co/JunkyByte/easy_ViTPose)  
Right now the yolo models are loaded from same folder of `inference.py` so place them there :)

```bash
$ python inference.py --help
usage: inference.py [-h] [--input INPUT] [--output-path OUTPUT_PATH] --model MODEL [--model-name MODEL_NAME]
                    [--yolo-size YOLO_SIZE] [--yolo-nano] [--show] [--show-yolo] [--save-img] [--save-json]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         image or video path
  --output-path OUTPUT_PATH
                        output path
  --model MODEL         ckpt path
  --model-name MODEL_NAME
                        [s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]
  --yolo-size YOLO_SIZE
                        YOLOv5 image size during inference
  --yolo-nano           Whether to use (the very fast) yolo nano (instead of small)
  --show                preview result
  --show-yolo           preview yolo result
  --save-img            save image result
  --save-json           save json result```
```

## Finetuning
Finetuning is done with `train.py` on COCO + feet.  
Check `datasets/COCO.py`, `config.yaml` and `train.py` for details.

---
## Reference
This code is substantially a fork of [jaehyunnn/ViTPose_pytorch](https://github.com/jaehyunnn/ViTPose_pytorch), without Jaehyunnn work this repo would not be possible. Thanks to the VitPose authors and their official implementation [ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose).
