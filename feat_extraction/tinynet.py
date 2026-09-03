# 2021.01.09-Changed for main script for testing TinyNet on ImageNet
# Huawei Technologies Co., Ltd. <foss@huawei.com>
""" 
An implementation of TinyNet
Requirements: timm==0.1.20
"""
import torch
import torch.nn as nn
import re
import types
import collections.abc as container_abcs

# --- STEP 1: MONKEY PATCH FOR PYTORCH 2.0+ / PYTHON 3.12 ---
if not hasattr(torch, "_six"):
    torch._six = types.ModuleType("_six")
    torch._six.container_abcs = container_abcs

# --- STEP 2: FLEXIBLE TIMM IMPORTS ---
try:
    from timm.models.efficientnet_builder import decode_arch_def, round_channels, resolve_bn_args
except ImportError:
    from timm.models._efficientnet_builder import decode_arch_def, round_channels, resolve_bn_args

from timm.models.efficientnet import EfficientNet, _cfg
from timm.models.registry import register_model

# --- STEP 3: MODEL DEFINITION ---
class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
def _gen_tinynet(variant_cfg, channel_multiplier=1.0, depth_multiplier=1.0, depth_trunc='round', **kwargs):
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'], ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'], ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'], ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    
    # 1. Manual Regex scaling for the blocks
    scaled_arch_def = []
    for stage in arch_def:
        scaled_stage = []
        for block_str in stage:
            channels = re.search(r'c(\d+)', block_str)
            if channels:
                orig_ch = int(channels.group(1))
                new_ch = round_channels(orig_ch, channel_multiplier, 8, None)
                block_str = block_str.replace(channels.group(0), f'c{new_ch}')
            scaled_stage.append(block_str)
        scaled_arch_def.append(scaled_stage)

    # 2. Build the model kwargs
    # We use max(1280, ...) to ensure the head doesn't shrink below 1280 
    # unless specifically intended, matching TinyNet-E's architecture.
    model_kwargs = dict(
        block_args=decode_arch_def(scaled_arch_def, depth_multiplier, depth_trunc=depth_trunc), 
        num_features=max(1280, round_channels(1280, channel_multiplier, 8, None)),
        stem_size=32, # TinyNet-E usually keeps the stem at 32
        fix_stem=True,
        act_layer=Swish,
        **kwargs,
    )
    
    model_kwargs.pop('channel_multiplier', None)
    model_kwargs.pop('round_chs_fn', None)
    
    model = EfficientNet(**model_kwargs)
    model.default_cfg = variant_cfg
    return model

@register_model
def tinynet(r=1.0, w=1.0, d=1.0, **kwargs):
    """ TinyNet """
    # 1. Force in_chans to 3 if not specified, 
    # as TinyNet-E pretrained weights are RGB
    in_chans = kwargs.get('in_chans', 3)
    
    hw = int(224 * r)
    
    # 2. Update the input_size in the config to match in_chans
    model = _gen_tinynet(
        _cfg(input_size=(in_chans, hw, hw)), 
        channel_multiplier=w, 
        depth_multiplier=d, 
        **kwargs
    )
    return model