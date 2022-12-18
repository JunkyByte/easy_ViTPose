import torch
import cv2
import numpy as np
from time import time

from PIL import Image
from torchvision.transforms import transforms

from models.model import ViTPose
from utils.top_down_eval import keypoints_from_heatmaps

if __name__ == "__main__":
    from configs.ViTPose_base_coco_256x192 import model as model_cfg
    from configs.ViTPose_base_coco_256x192 import data_cfg
    vit_pose = ViTPose(model_cfg)
    
    ckpt = torch.load('/Users/jaehyun/workspace/ViTPose_pytorch/vitpose-b-multi-coco.pth')['state_dict']
    vit_pose.load_state_dict(ckpt)
    
    img = Image.open("/Users/jaehyun/workspace/ViTPose_pytorch/examples/img1.jpg")
    img_w, img_h = img.size
    print(f">>> Original image size: {img_h} X {img_w}")
    
    resize_size = data_cfg['image_size']
    img_tensor = transforms.Compose (
        [transforms.Resize((resize_size[1], resize_size[0])),
         transforms.ToTensor()]
    )(img).unsqueeze(0)
    
    
    with torch.no_grad():
        tic = time()
        out = vit_pose(img_tensor)
        print(f">>> Output size: {out.shape} ---> {time()-tic:.4f} sec. elapsed")
        
    out_h, out_w = out.shape[-2:]
    kpts, probs = keypoints_from_heatmaps(heatmaps=out.numpy(),
                                          center=np.array([[img_w//2, img_h//2]]), # x, y
                                          scale=np.array([[img_h/resize_size[1], img_w/resize_size[0]]]), # h, w
                                          unbiased=False,
                                          post_process='default',
                                          kernel=11,
                                          valid_radius_factor=0.0546875,
                                          use_udp=False,
                                          target_type='GaussianHeatmap')
    
    img = np.array(img)
    for coord_y, coord_x in kpts[0]:
        img = cv2.drawMarker(img, (int(coord_y), int(coord_x)), color=([0, 0, 255]), thickness=3)
        
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()