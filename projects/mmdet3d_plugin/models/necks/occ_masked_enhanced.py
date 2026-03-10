import torch
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import NECKS
from ...ops import bev_pool_v2
from ..model_utils import DepthNet
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from ..vmamba.vmamba import VSSBlock

@NECKS.register_module()
class OCC_Masked_Enhanced(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 ):
        super(OCC_Masked_Enhanced, self).__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        

        # 用于上采样high-level的feature map
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.in_channel1 = in_channels * 8 // (8+2)
        self.in_channel2 = in_channels * 2 // (8+2)
        

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )
            # self.up2 = nn.Sequential(
            #     nn.Upsample(size=(200, 200), mode='bilinear', align_corners=False),
            #     nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
            #     build_norm_layer(norm_cfg, out_channels)[1],
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            # )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True)
            )
        #mamba in fpn
        for i in range(1,3):  
            setattr(self, f'mamba_blockccw_{i}', VSSBlock(forward_type='v4_noz', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
            setattr(self, f'mamba_blockcw_{i}', VSSBlock(forward_type='v5_noz', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
        for i in range(3,5):  
            setattr(self, f'mamba_blockccw_{i}', VSSBlock(forward_type='v4_noz', channel_first=False,
                                              hidden_dim=self.in_channel1+self.in_channel2, ssm_ratio=1.0))
            setattr(self, f'mamba_blockcw_{i}', VSSBlock(forward_type='v5_noz', channel_first=False,
                                              hidden_dim=self.in_channel1+self.in_channel2, ssm_ratio=1.0))
        
        #rerange scan ways 
        
            
    def forward(self, feats):
        """
        Args:
            feats: List[Tensor,] multi-level features
                List[(B, C1, H, W), (B, C2, H/2, W/2), (B, C3, H/4, W/4)]
        Returns:
            x: (B, C_out, 2*H, 2*W)
        """
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.mamba_blockccw_1(x1.permute(0, 2, 3, 1).contiguous())
        x1 = self.mamba_blockcw_1(x1.contiguous())
        x1 = self.mamba_blockccw_2(x1.contiguous())
        x1 = self.mamba_blockcw_2(x1.contiguous()).permute(0, 3, 1, 2).contiguous()
        bev_mask_feat = x1
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        x1 = self.mamba_blockccw_3(x1.permute(0, 2, 3, 1).contiguous())
        x1 = self.mamba_blockcw_3(x1.contiguous())
        x1 = self.mamba_blockccw_4(x1.contiguous())
        x1 = self.mamba_blockcw_4(x1.contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        return x, bev_mask_feat