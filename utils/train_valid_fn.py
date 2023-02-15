import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from models.losses import JointsMSELoss
from models.optimizer import LayerDecayOptimizer

from utils.dist_util import get_dist_info, init_dist
from utils.logging import get_root_logger



def train_model(model: nn.Module, datasets: Dataset, cfg: dict, distributed: bool, validate: bool,  timestamp: str, meta: dict) -> None:
    logger = get_root_logger()
    
    # Prepare data loaders
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    if distributed:
        samplers = [DistributedSampler(ds, num_replicas=len(cfg.gpu_ids), rank=torch.cuda.current_device(), shuffle=True, drop_last=False) for ds in datasets]
    else:
        samplers = [None for ds in datasets]
    dataloaders = [DataLoader(ds, batch_size=cfg.data['samples_per_gpu'], sampler=sampler, num_workers=cfg.data['workers_per_gpu'], pin_memory=False) for ds, sampler in zip(datasets, samplers)]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        model = DistributedDataParallel(model, 
                                        device_ids=[torch.cuda.current_device()], 
                                        broadcast_buffers=False, 
                                        find_unused_parameters=find_unused_parameters)
    else:
        model = DataParallel(model, device_ids=cfg.gpu_ids)
    
    # Loss function
    criterion = JointsMSELoss(use_target_weight=cfg.model['keypoint_head']['loss_keypoint']['use_target_weight'])
    
    # Optimizer
    lr = cfg.optimizer['lr']
    optimizer = AdamW(model.parameters(), lr=lr, betas=cfg.optimizer['betas'], weight_decay=cfg.optimizer['weight_decay'])
    
    # Layer-wise learning rate decay
    lr_mult = [cfg.optimizer['paramwise_cfg']['layer_decay_rate']] * cfg.optimizer['paramwise_cfg']['num_layers']
    layerwise_optimizer = LayerDecayOptimizer(optimizer, lr_mult)
    
    
    """ TODO: Implementation of Warmup LR"""
    lr_scheduler = MultiStepLR(optimizer, milestones=cfg.lr_config['Step'])
    
    model.train()
    for dataloader in dataloaders:
        for epoch in cfg.total_epochs:
            for batch_idx, batch in dataloader:
                layerwise_optimizer.zero_grad()
                
                images, targets, target_weights, __ = batch
                outputs = model(images)
                
                loss = criterion(outputs, targets) # if use_target_weight=True, then criterion(outputs, targets, target_weights)
                loss.backward()
                clip_grad_norm_(model.parameters(), **cfg.optimizer_config['grad_clip'])
                
                layerwise_optimizer.step()
            lr_scheduler.step()
            

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        raise NotImplementedError()
    
    # validation
    if validate:
        raise NotImplementedError()
