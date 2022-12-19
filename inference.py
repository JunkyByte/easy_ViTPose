import argparse
import os.path as osp

import torch

from pathlib import Path
import cv2
import numpy as np


from time import time
from PIL import Image
from torchvision.transforms import transforms

from models.model import ViTPose
from utils.top_down_eval import keypoints_from_heatmaps
from utils.visualization import draw_points_and_skeleton, joints_dict


__all__ = ['inference']


@torch.no_grad()
def inference(img_path: Path, img_size: tuple[int, int],
              model_cfg: dict, ckpt_path: Path, device: torch.device, save_result: bool=True) -> np.ndarray:
    
    # Prepare model
    vit_pose = ViTPose(model_cfg)
    vit_pose.load_state_dict(torch.load(ckpt_path)['state_dict'])
    vit_pose.to(device)
    
    # Prepare input data
    img = Image.open(img_path)
    org_w, org_h = img.size
    print(f">>> Original image size: {org_h} X {org_w}")
    
    img_tensor = transforms.Compose (
        [transforms.Resize((img_size[1], img_size[0])),
         transforms.ToTensor()]
    )(img).unsqueeze(0).to(device)
    
    # Feed to model
    tic = time()
    heatmaps = vit_pose(img_tensor).detach().cpu().numpy()
    elapsed_time = time()-tic
    print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")
    
    kpts, probs = keypoints_from_heatmaps(heatmaps=heatmaps,
                                          center=np.array([[org_w//2, org_h//2]]), # x, y
                                          scale=np.array([[org_w/img_size[1], org_h/img_size[0]]]), # h, w
                                          unbiased=False,
                                          post_process='default',
                                          kernel=11,
                                          valid_radius_factor=0.0546875,
                                          use_udp=False,
                                          target_type='GaussianHeatmap')
    points = np.concatenate([kpts[:, ::-1], probs], axis=2) # batch, num_people, (2+1)

    # Visualization 
    if save_result:
        for pid, point in enumerate(points):
            img = np.array(img)[:,:,::-1] # RGB to BGR for cv2 modules
            img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                           points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                           points_palette_samples=10, confidence_threshold=0.4)
            save_name = img_path.replace(".jpg", "_result.jpg")
            cv2.imwrite(save_name, img)
    
    return points
    

if __name__ == "__main__":
    from configs.ViTPose_base_coco_256x192 import model as model_cfg
    from configs.ViTPose_base_coco_256x192 import data_cfg
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', nargs='+', type=str, default='examples/img1.jpg', help='image path(s)')
    args = parser.parse_args()
    
    CUR_DIR = osp.dirname(__file__)
    CKPT_PATH = f"{CUR_DIR}/vitpose-b-multi-coco.pth"
    
    img_size = data_cfg['image_size']
    for img_path in args.image_path:
        keypoints = inference(img_path=img_path, img_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH, 
                            device=torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu'),
                            save_result=True)