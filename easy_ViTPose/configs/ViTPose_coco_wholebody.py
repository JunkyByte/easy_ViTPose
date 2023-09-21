from .ViTPose_common import *

# Channel configuration
channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)))

# Set models channels
model_small['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
model_base['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
model_large['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
model_huge['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
