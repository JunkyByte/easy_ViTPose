from .ViTPose_common import *

# Channel configuration
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# Set models channels
model_small['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
model_base['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
model_large['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
model_huge['keypoint_head']['out_channels'] = channel_cfg['num_output_channels']
