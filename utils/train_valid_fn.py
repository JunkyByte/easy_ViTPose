import torch
import torch.nn as nn

from torch.utils.data import Dataset

from utils.dist_util import get_dist_info, init_dist
from utils.logging import get_root_logger

def train_model(model: nn.Module, datasets: Dataset, cfg: dict, distributed: bool, validate: bool,  timestamp: str, meta: dict) -> None:
    logger = get_root_logger()
    
    # Prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    
    dataloader_cfg = {
        **dict(seed=cfg.seed,
               drop_last=False,
               dist=distributed,
               num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        # **dict((k, cfg['data'][k]))
    }