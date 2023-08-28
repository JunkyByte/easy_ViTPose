# easy_ViTPose
<p align="center">
<img src="https://user-images.githubusercontent.com/24314647/236082274-b25a70c8-9267-4375-97b0-eddf60a7dfc6.png" width=375> easy_ViTPose
</p>

## Accurate 2d human pose estimation, finetuned on 25 keypoints COCO skeleton + feet  

<a target="_blank" href="https://colab.research.google.com/github/JunkyByte/easy_ViTPose/blob/main/colab_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Easy to use SOTA `ViTPose` [Y. Xu et al., 2022] models for fast inference.  

## Results

![resimg](https://user-images.githubusercontent.com/24314647/236281199-98e45ab5-2a18-45b7-ba5c-36bdec4450f4.png)
(Model - ViTPose-b)

https://github.com/JunkyByte/easy_ViTPose/assets/24314647/a43ca37b-3e64-4c19-bf07-813fdf45f112

(Credits: https://www.youtube.com/watch?v=p-rSdt0aFuw&pp=ygUhZGFuY2UgZXZvbHV0aW9uIGZyb20gMTk1MCB0byAyMDE5)  
(s - small, b - base, l - large, h - huge)

## Features
### NEW: You can now use MPS on Apple silicon computers!
- Image / Video / Webcam support
- Video support using SORT algorithm to track bboxes between frames and mantain multi pose identification
- Torch / ONNX / Tensorrt models
- 4 ViTPose architectures with different sizes
- cpu / gpu / mps (apple silicon gpu)
- save output images / videos and json

### Benchmark:
Run on `GTX1080ti`, consider that tensorrt > onnx > torch.  
These benchmarks are relative to `ViTPose` models, they do not consider Yolo detection that is done before `ViTPose` inference.  
Tensorrt:  
- ViTPose-s: ~250 fps
- ViTPose-b: ~125 fps
- ViTPose-l: ~45 fps
- ViTPose-h: ~24.5 fps
(these are relative to single person pose estimation)

### Skeleton reference
The skeleton keypoint ordering can be found in [visualization.py](https://github.com/JunkyByte/easy_ViTPose/blob/main/src/vit_utils/visualization.py#L14) or below.  
<details>
  <summary>Skeleton reference image</summary>
  
  ![skeleton](https://github.com/JunkyByte/easy_ViTPose/assets/24314647/cf0eefa0-3768-4acf-9638-8a1673e32830)
</details>

## Installation and Usage
#### NEW: easy_ViTPose is now a package for easier custom inference  
> ![IMPORTANT]   
> I did not enforce the `requirements.txt` as they are not thoroughly tested, be sure to install the necessary packages by yourself.
> To use MPS be sure to install a compatible torch version.

```bash
git clone git@github.com:JunkyByte/easy_ViTPose.git
cd easy_ViTPose/
pip install -e .
pip install -r requirements.txt
```

### Download models
- Download the models from [Huggingface](https://huggingface.co/JunkyByte/easy_ViTPose) or with `downloader.py`
```bash
$ python downloader.py --help
usage: downloader.py [-h] --backend {torch,onnx,tensorrt} --model-name {s,b,l,h} [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --backend {torch,onnx,tensorrt}
                        Model backend [torch, onnx, tensorrt]
  --model-name {s,b,l,h}
                        [s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H, YOLO-S: YOLO-S, YOLO-N: YOLO-N]
  --output OUTPUT       Dir path for checkpoint output
```

> [!NOTE]   
> If you encounter any problem with ONNX and TRT checkpoints, generate them again by yourself using `export.py`
```bash
$ python export.py --help
usage: export.py [-h] --model-ckpt MODEL_CKPT --model-name {s,b,l,h} [--output OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --model-ckpt MODEL_CKPT
                        The torch model that shall be used for conversion
  --model-name {s,b,l,h}
                        [s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]
  --output OUTPUT       File (without extension) or dir path for checkpoint output
```

### Run inference
To run inference from command line you can use the `inference.py` script as follows:  
(be sure to `cd easy_ViTPose/easy_ViTPose/`)  
```bash
$ python inference.py --help
usage: inference.py [-h] [--input INPUT] [--output-path OUTPUT_PATH] --model MODEL [--yolo YOLO]
                    [--model-name {s,b,l,h}] [--yolo-size YOLO_SIZE] [--conf-threshold CONF_THRESHOLD]
                    [--rotate {0,90,180,270}] [--yolo-step YOLO_STEP] [--single-pose] [--show] [--show-yolo]
                    [--show-raw-yolo] [--save-img] [--save-json]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         path to image / video or webcam ID (=cv2)
  --output-path OUTPUT_PATH
                        output path, if the path provided is a directory output files are "input_name
                        +_result{extension}".
  --model MODEL         checkpoint path of the model
  --yolo YOLO           checkpoint path of the yolo model
  --model-name {s,b,l,h}
                        [s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]
  --yolo-size YOLO_SIZE
                        YOLOv5 image size during inference
  --conf-threshold CONF_THRESHOLD
                        Minimum confidence for keypoints to be drawn. [0, 1] range
  --rotate {0,90,180,270}
                        Rotate the image of [90, 180, 270] degress counterclockwise
  --yolo-step YOLO_STEP
                        The tracker can be used to predict the bboxes instead of yolo for performance, this flag
                        specifies how often yolo is applied (e.g. 1 applies yolo every frame). This does not have any
                        effect when is_video is False
  --single-pose         Do not use SORT tracker because single pose is expected in the video
  --show                preview result during inference
  --show-yolo           draw yolo results
  --show-raw-yolo       draw yolo result before that SORT is applied for tracking (only valid during video inference)
  --save-img            save image results
  --save-json           save json results
```

You can run inference from code as follows:
```python
import cv2
from easy_ViTPose import VitInference

# Image to run inference RGB format
img = cv2.imread('./examples/img1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# set is_video=True to enable tracking in video inference
# be sure to use VitInference.reset() function to reset the tracker after each video
# There are a few flags that allows to customize VitInference, be sure to check the class definition
model_path = './ckpts/vitpose-25-s.onnx'
yolo_path = './yolov5s.onnx'

# If you want to use MPS (on new macbooks) use the torch checkpoints for both ViTPose and Yolo
# If device is None will try to use cuda -> mps -> cpu (otherwise specify 'cpu', 'mps' or 'cuda')
model = VitInference(model_path, yolo_path, model_name='s', yolo_size=320, is_video=False, device=None)

# Infer keypoints, output is a dict where keys are person ids and values are keypoints (np.ndarray (25, 3): (y, x, score))
# If is_video=True the IDs will be consistent among the ordered video frames.
keypoints = model.inference(img)

# call model.reset() after each video

img = model.draw(show_yolo=True)  # Returns RGB image with drawings
cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); cv2.waitKey(0)
```
> [!NOTE]   
> If the input file is a video [SORT](https://github.com/abewley/sort) is used to track people IDs and output consistent identifications.

### OUTPUT json format
The output format of the json files:

```
{
    "keypoints":
    [  # The list of frames, len(json['keypoints']) == len(video)
        {  # For each frame a dict
            "0": [  #  keys are id to track people and value the keypoints
                [121.19, 458.15, 0.99], # Each keypoint is (y, x, score)
                [110.02, 469.43, 0.98],
                [110.86, 445.04, 0.99],
            ],
            "1": [
                ...
            ],
        },
        {
            "0": [
                [122.19, 458.15, 0.91],
                [105.02, 469.43, 0.95],
                [122.86, 445.04, 0.99],
            ],
            "1": [
                ...
            ]
        }
    ],
    "skeleton":
    {  # Skeleton reference, key the idx, value the name
        "0": "nose",
        "1": "left_eye",
        "2": "right_eye",
        "3": "left_ear",
        "4": "right_ear",
        "5": "neck",
        ...
    }
}
```


## Finetuning
Finetuning is done with `train.py` on COCO + feet.  
Check `datasets/COCO.py`, `config.yaml` and `train.py` for details.

---

## TODO:
- Tensorrt version of yolo
- ~~Add possibility to not use tracker if single pose is expected in a video (benchmark the tracker)~~
- ~~package setup~~
- ~~Download models with script~~
- benchmark and check bottlenecks of inference pipeline
- parallel batched inference
- ~~tuning the parameters of the SORT~~ (to be tested)
- ~~allow for skip frames of yolo detection (to have faster inference) leveraging the SORT for tracking during those frames.~~
- ~~confidence masking on skeleton drawing (add arg)~~
  
Feel free to open issues, pull requests and contribute on these TODOs.

## Reference
This code started as a fork of [jaehyunnn/ViTPose_pytorch](https://github.com/jaehyunnn/ViTPose_pytorch).  
Thanks to the VitPose authors and their official implementation [ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose).  
The SORT code is taken from [abewley/sort](https://github.com/abewley/sort)
