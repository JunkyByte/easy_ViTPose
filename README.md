# ViTPose (simple version)
An unofficial implementation of ViTPose [Y. Xu et al., 2022] 

## Usage
```
python inference.py --image-path './examples/img1.jpg'
```

## Note
1.  Download the trained model (.pth)
    - [ViTPose-B-Multi-COCO.pth](https://1drv.ms/u/s!AimBgYV7JjTlgSrlMB093JzJtqq-?e=Jr5S3R)
    - [ViTPose-L-Multi-COCO.pth](https://1drv.ms/u/s!AimBgYV7JjTlgTBm3dCVmBUbHYT6?e=fHUrTq)
    - [ViTPose-H-Multi-COCO.pth](https://1drv.ms/u/s!AimBgYV7JjTlgS5rLeRAJiWobCdh?e=41GsDd)
2. Set the config. according to the trained model
    - [ViTPose-B-COCO-256x192](/Users/jaehyun/workspace/ViTPose_pytorch/configs/ViTPose_base_coco_256x192.py) 
    - [ViTPose-L-COCO-256x192](/Users/jaehyun/workspace/ViTPose_pytorch/configs/ViTPose_large_coco_256x192.py) 
    - [ViTPose-H-COCO-256x192](/Users/jaehyun/workspace/ViTPose_pytorch/configs/ViTPose_huge_coco_256x192.py) 

---
## Reference
All codes were written with reference to [the official ViTPose repo.](https://github.com/ViTAE-Transformer/ViTPose)