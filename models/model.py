import torch
import torch.nn as nn

from .backbone.vit import ViT
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead


__all__ = ['ViTPose']


class ViTPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(ViTPose, self).__init__()
        
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}
        
        print(f">>> Backbone config: {backbone_cfg}")
        print(f">>> Head config: {head_cfg}")
        
        self.backbone = ViT(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)
        
    def forward(self, x):
        return self.keypoint_head(self.backbone(x))
    
if __name__ == "__main__":
    from configs.ViTPose_base_coco_256x192 import model as model_cfg
    # patch_size = model['backbone']['patch_size']
    # embed_dim = model['backbone']
    vit_pose = ViTPose(model_cfg)
    
    ckpt = torch.load('/Users/jaehyun/workspace/simple_ViTPose/vitpose-b-multi-coco.pth')['state_dict']
    vit_pose.load_state_dict(ckpt)
    sample = torch.zeros([1,3,192,256])
    
    with torch.no_grad():
        out = vit_pose(sample)
    
    