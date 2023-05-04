# easy_ViTPose
<p align="center">
<img src="https://user-images.githubusercontent.com/24314647/236082274-b25a70c8-9267-4375-97b0-eddf60a7dfc6.png" width=375> easy_ViTPose
</p>

## Accurate 2d human pose estimation, finetuned on 25 keypoints COCO skeleton + feet  
### Easy to use SOTA `ViTPose` [Y. Xu et al., 2022] models for fast inference.  

### Features

- Image / Video / Webcam support
- Torch / ONNX / Tensorrt models
- 4 ViTPose architectures with different sizes
- cpu / gpu support

## Usage
### | **Inference**
```
python inference.py --image-path './examples/img1.jpg'
```

### | **Training**
```
python train.py --config-path config.yaml --model-name 'b'
```
- `model_name` must be in (`b`, `l`, `h`)


---
## Reference
This code is substantially a fork of [jaehyunnn/ViTPose_pytorch](https://github.com/jaehyunnn/ViTPose_pytorch), without Jaehyunnn work this repo would not be possible. Thanks to the VitPose authors and their official implementation [ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose).
