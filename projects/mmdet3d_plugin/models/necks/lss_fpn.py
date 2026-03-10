import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from torch.utils.checkpoint import checkpoint
from mmcv.cnn.bricks import ConvModule
from mmdet.models import NECKS
from ..vmamba.vmamba import VSSBlock

@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super(FPN_LSS, self).__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        # 用于上采样high-level的feature map
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

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

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True)
            )

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
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        return x


@NECKS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False):
        super().__init__()
        self.up1 = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU', inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        """
        Args:
            feats: List[
                (B, C, Dz, Dy, Dx),
                (B, 2C, Dz/2, Dy/2, Dx/2),
                (B, 4C, Dz/4, Dy/4, Dx/4)
            ]
        Returns:
            x: (B, C, Dz, Dy, Dx)
        """
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)       # (B, 2C, Dz, Dy, Dx)
        x_32 = self.up2(x_32)       # (B, 4C, Dz, Dy, Dx)
        x = torch.cat([x_8, x_16, x_32], dim=1)     # (B, 7C, Dz, Dy, Dx)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)    # (B, C, Dz, Dy, Dx)
        return x


@NECKS.register_module()
class FPN_LSS_Mamba(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super(FPN_LSS_Mamba, self).__init__()
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
            setattr(self, f'mamba_block{i}', VSSBlock(forward_type='v4_noz', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
        for i in range(3,5):  
            setattr(self, f'mamba_block{i}', VSSBlock(forward_type='v4_noz', channel_first=False,
                                              hidden_dim=self.in_channel1+self.in_channel2, ssm_ratio=1.0))
        self.mamba_block5 = VSSBlock(forward_type='v4_noz', channel_first=False,
                                              hidden_dim=self.out_channels, ssm_ratio=1.0, drop_path=0.2)
        
        
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
        x1 = self.mamba_block1(x1.permute(0, 2, 3, 1).contiguous())
        x1 = self.mamba_block2(x1.contiguous()).permute(0, 3, 1, 2).contiguous()
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        x1 = self.mamba_block3(x1.permute(0, 2, 3, 1).contiguous())
        x1 = self.mamba_block4(x1.contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        x = self.mamba_block5(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        return x



@NECKS.register_module()
class FPN_LSS_Mamba_FOBA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 use_input_conv=False):
        super(FPN_LSS_Mamba_FOBA, self).__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        # 用于上采样high-level的feature map
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.in_channel1 = in_channels * 8 // (8+2)
        self.in_channel2 = in_channels * 2 // (8+2)
        self.in_channel1_half = self.in_channel1 // 2
        self.in_channel2_half = self.in_channel2 // 2
        self.out_channels_half = out_channels // 2

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(self.in_channel1_half, self.in_channel1, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, self.in_channel1)[1],
            nn.LeakyReLU(inplace=True),
        )
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(self.in_channel2_half, self.in_channel2, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, self.in_channel2)[1],
            nn.LeakyReLU(inplace=True),
        )

        channels_factor = 2 if self.extra_upsample else 1
        self.conv_back = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )
        self.conv_fore = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )
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
            self.up2_back = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )
            self.up2_fore = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )
  
        #mamba in fpn (background)
        for i in range(1,2):  
            setattr(self, f'mamba_block_backccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
            setattr(self, f'mamba_block_backcw_{i}', VSSBlock(forward_type='v5', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
        for i in range(2,3):  
            setattr(self, f'mamba_block_backccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                              hidden_dim = self.in_channel1+self.in_channel2, ssm_ratio=1.0))
            setattr(self, f'mamba_block_backcw_{i}', VSSBlock(forward_type='v5', channel_first=False,
                                              hidden_dim = self.in_channel1+self.in_channel2, ssm_ratio=1.0))
        
        #mamba in fpn (foreground)
        for i in range(1,2):  
            setattr(self, f'mamba_block_foreccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
            setattr(self, f'mamba_block_forecw_{i}', VSSBlock(forward_type='v5', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
        for i in range(2,3):  
            setattr(self, f'mamba_block_foreccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                              hidden_dim = self.in_channel1+self.in_channel2, ssm_ratio=1.0))
            setattr(self, f'mamba_block_forecw_{i}', VSSBlock(forward_type='v5', channel_first=False,
                                              hidden_dim = self.in_channel1+self.in_channel2, ssm_ratio=1.0))

        #mamba in fpn (merge)
        for i in range(1,2):  
            setattr(self, f'mamba_block_mergeccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
            setattr(self, f'mamba_block_mergecw_{i}', VSSBlock(forward_type='v5', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
        # for i in range(3,5):  
        #     setattr(self, f'mamba_block_merge_{i}', VSSBlock(forward_type='v4', channel_first=False,
        #                                       hidden_dim=self.in_channel1+self.in_channel2, ssm_ratio=1.0, drop_path=0.1))
        # self.mamba_block_5 = VSSBlock(forward_type='v4', channel_first=False,
        #                                       hidden_dim=self.out_channels, ssm_ratio=1.0)
            
    def forward(self, feats):
        """
        Args:
            feats: List[Tensor,] multi-level features
                List[(B, C1, H, W), (B, C2, H/2, W/2), (B, C3, H/4, W/4)]
        Returns:
            x: (B, C_out, 2*H, 2*W)
        """
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        x2_background = x2[:, :self.in_channel2_half, :, :]
        x2_foreground = x2[:, self.in_channel2_half:, :, :]
        x1_background = x1[:, :self.in_channel1_half, :, :]
        x1_foreground = x1[:, self.in_channel1_half:, :, :]

        x2_background = self.conv_x2(x2_background)
        x2_foreground = self.conv_x2(x2_foreground)
        
        x1_background = self.conv_x1(x1_background)
        x1_foreground = self.conv_x1(x1_foreground)

        #background_process
        x1_background = self.mamba_block_backccw_1(x1_background.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x1_background = self.mamba_block_backcw_1(x1_background.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x1_background = self.up(x1_background)    
        x1_background = torch.cat([x2_background, x1_background], dim=1)
        x1_background = self.mamba_block_backccw_2(x1_background.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x1_background = self.mamba_block_backcw_2(x1_background.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x_background = self.conv_back(x1_background)
        if self.extra_upsample:
            x_background = self.up2_back(x_background)
        # x_background = self.mamba_block_back_5(x_background.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        
        #foreground_process
        x1_foreground = self.mamba_block_foreccw_1(x1_foreground.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x1_foreground = self.mamba_block_forecw_1(x1_foreground.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x1_foreground = self.up(x1_foreground)    
        x1_foreground = torch.cat([x2_foreground, x1_foreground], dim=1)     
        x1_foreground = self.mamba_block_foreccw_2(x1_foreground.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x1_foreground = self.mamba_block_forecw_2(x1_foreground.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x_foreground = self.conv_fore(x1_foreground)   
        if self.extra_upsample:
            x_foreground = self.up2_fore(x_foreground)
        # x_foreground = self.mamba_block_fore_5(x_foreground.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        # merge_process 
        x1 = self.mamba_block_mergeccw_1(x1.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x1 = self.mamba_block_mergecw_1(x1.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        # x1 = self.mamba_block_merge_3(x1.permute(0, 2, 3, 1).contiguous())
        # x1 = self.mamba_block_merge_4(x1.contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        # x = self.mamba_block_5(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        return x, x_background, x_foreground



@NECKS.register_module()
class FPN_LSS_Mamba_FOBA_V2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super(FPN_LSS_Mamba_FOBA_V2, self).__init__()
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
        self.mamba_blockccw_5 = VSSBlock(forward_type='v4_noz', channel_first=False,
                                              hidden_dim=self.out_channels, ssm_ratio=1.0)
        self.mamba_blockcw_5 = VSSBlock(forward_type='v5_noz', channel_first=False,
                                              hidden_dim=self.out_channels, ssm_ratio=1.0)
        
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
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        x1 = self.mamba_blockccw_3(x1.permute(0, 2, 3, 1).contiguous())
        x1 = self.mamba_blockcw_3(x1.contiguous())
        x1 = self.mamba_blockccw_4(x1.contiguous())
        x1 = self.mamba_blockcw_4(x1.contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        x = self.mamba_blockccw_5(x.permute(0, 2, 3, 1).contiguous())
        x = self.mamba_blockcw_5(x.contiguous()).permute(0, 3, 1, 2).contiguous()
        return x
    

@NECKS.register_module()
class FPN_LSS_Mamba_Mask(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 ):
        super(FPN_LSS_Mamba_Mask, self).__init__()
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
    


@NECKS.register_module()
class FPN_LSS_Mamba_FO(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 ):
        super(FPN_LSS_Mamba_FO, self).__init__()
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

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True)
            )
        #mamba in fpn
        for i in range(1,3):  
            setattr(self, f'mamba_blockccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
            setattr(self, f'mamba_blockcw_{i}', VSSBlock(forward_type='v5', channel_first=False,
                                              hidden_dim = self.in_channel1, ssm_ratio=1.0))
        for i in range(3,4):  
            setattr(self, f'mamba_blockccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                              hidden_dim=self.in_channel1+self.in_channel2, ssm_ratio=1.0))
            setattr(self, f'mamba_blockcw_{i}', VSSBlock(forward_type='v5', channel_first=False,
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
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        x1 = self.mamba_blockccw_3(x1.permute(0, 2, 3, 1).contiguous())
        x1 = self.mamba_blockcw_3(x1.contiguous()).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        return x