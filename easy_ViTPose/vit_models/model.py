import torch.nn as nn

from .backbone.vit import ViT
from .backbone.vit_moe import ViTMoE
from .head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead


__all__ = ['ViTPose']


class ViTPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(ViTPose, self).__init__()
        
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}
        
        if cfg['backbone']['type'] == 'ViT':
            backbone = ViT
        elif cfg['backbone']['type'] == 'ViTMoE':
            backbone = ViTMoE
        else:
            raise TypeError(f'The backbone {cfg["backbone"]["type"]} specified is not supported')

        self.backbone = backbone(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)
    
    def forward_features(self, x):
        return self.backbone(x)
    
    def forward(self, x):
        return self.keypoint_head(self.backbone(x))
