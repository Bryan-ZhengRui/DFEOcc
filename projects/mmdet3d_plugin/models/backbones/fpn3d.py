import torch.utils.checkpoint as checkpoint
import torch
from torch import nn
import torch.nn.functional as F  
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
from ..vmamba.vmamba import VSSBlock

@BACKBONES.register_module()
class FPN3D(nn.Module):  
    def __init__(self, in_channels, hidden_channels, out_channels):  
        """  
        Implement a simple 3D FPN with two scales: 1/2 and 1/8.  
        Args:  
            in_channels: Number of input feature channels C  
            out_channels: Output feature channels at each scale  
        """  
        super(FPN3D, self).__init__()  

        # Downsampling stages  
        # Scale 1/2  
        self.encoder1 = nn.Sequential(  
            # Second layer: depthwise (grouped) convolution, stride=2  
            nn.Conv3d(in_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, groups=4),  # grouped conv  
            nn.BatchNorm3d(hidden_channels * 2),  
            # Third layer: pointwise convolution (1x1x1) for channel interaction  
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=1, stride=1, padding=0),  
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        # Scale 1/8  
        self.encoder2 = nn.Sequential(  
            nn.Conv3d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1, groups=4),  # grouped conv  
            nn.BatchNorm3d(hidden_channels * 4),   
            nn.Conv3d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, stride=2, padding=1, groups=4),  
            nn.BatchNorm3d(hidden_channels * 4),  
            nn.Conv3d(hidden_channels * 4, hidden_channels * 4, kernel_size=1, stride=1, padding=0),  
            nn.BatchNorm3d(hidden_channels * 4),  
            nn.ReLU(inplace=True),  
        )  
        self.convmerge_feat2 = nn.Sequential(  
            nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        self.convmerge_fused_feat = nn.Sequential(  
            nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        # Fusion stage (upsampling)  
        # Final output convolution, upsampled to input resolution  
        self.output_conv = nn.Sequential(  
            # First layer: depthwise convolution (4 groups)  
            nn.Conv3d(hidden_channels * 2, out_channels * 3 // 2, kernel_size=3, stride=1, padding=1, groups=4, bias=False),  
            nn.BatchNorm3d(out_channels * 3 // 2),  
            nn.ReLU(inplace=True),  
            # Second layer: depthwise convolution (4 groups)  
            nn.Conv3d(out_channels * 3 // 2, out_channels, kernel_size=5, stride=1, padding=2, groups=4, bias=False),  
            nn.BatchNorm3d(out_channels),  
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )  

    def forward(self, x):  
        """  
        Args:  
            x: input 3D feature map, shape (B, C, Dz, Dy, Dx)  
        Returns:  
            fused feature map, same shape as input (B, C, Dz, Dy, Dx)  
        """  
        # Stage 1: original resolution -> 1/2 downsampling  
        feat1 = self.encoder1(x)  # (B, C * 2, Dz/2, Dy/2, Dx/2)  
        # Stage 2: 1/2 -> 1/8 downsampling  
        feat2 = self.encoder2(feat1)  # (B, C * 4, Dz/8, Dy/8, Dx/8) 
        feat2 = self.convmerge_feat2(feat2)
        # Stage 3: 1/8 -> 1/2 upsampling  
        up_feat2 = F.interpolate(feat2, scale_factor=4, mode='trilinear', align_corners=True) # (B, C * 2, Dz/2, Dy/2, Dx/2)  
        # Fuse 1/2 downsampled features and upsampled features  
        fused_feat = torch.cat([feat1, up_feat2], dim=1)    # (B, C*4, Dz/2, Dy/2, Dx/2)  
        fused_feat = self.convmerge_fused_feat(fused_feat)
        # Stage 4: 1/2 -> 1, upsample to original resolution  
        out = F.interpolate(fused_feat, size=x.shape[2:], mode='trilinear', align_corners=True)  # Upsample to original size  
        out = self.output_conv(out)  # Final convolution  
        
        return out  
    

@BACKBONES.register_module()
class FPN3D_Mamba(nn.Module):  
    def __init__(self, in_channels, hidden_channels, out_channels):  
        """  
        Implement a simple 3D FPN with two scales: 1/2 and 1/8.  
        Args:  
            in_channels: Number of input feature channels C  
            out_channels: Output feature channels at each scale  
        """  
        super(FPN3D_Mamba, self).__init__()  

        # Downsampling stages  
        # Scale 1/2  
        self.encoder1 = nn.Sequential(  
            # Second layer: depthwise (grouped) convolution, stride=2  
            nn.Conv3d(in_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, groups=4),  
            nn.BatchNorm3d(hidden_channels * 2),  
            # Third layer: pointwise convolution (1x1x1) for channel interaction  
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=1, stride=1, padding=0),  
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        # Scale 1/8  
        self.encoder2 = nn.Sequential(  
            nn.Conv3d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1, groups=4),  
            nn.BatchNorm3d(hidden_channels * 4),   
            nn.Conv3d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, stride=2, padding=1, groups=4),  
            nn.BatchNorm3d(hidden_channels * 4),  
            nn.Conv3d(hidden_channels * 4, hidden_channels * 4, kernel_size=1, stride=1, padding=0),  
            nn.BatchNorm3d(hidden_channels * 4),  
            nn.ReLU(inplace=True),  
        )  
        self.convmerge_feat2 = nn.Sequential(  
            nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        self.convmerge_fused_feat = nn.Sequential(  
            nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        # Fusion stage (upsampling)  
        self.output_conv = nn.Sequential(  
            nn.Conv3d(hidden_channels * 2, out_channels * 3 // 2, kernel_size=3, stride=1, padding=1, groups=4, bias=False),  
            nn.BatchNorm3d(out_channels * 3 // 2),  
            nn.ReLU(inplace=True),  
            nn.Conv3d(out_channels * 3 // 2, out_channels, kernel_size=5, stride=1, padding=2, groups=4, bias=False),  
            nn.BatchNorm3d(out_channels),  
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )  

        for i in range(1,2):  
            setattr(self, f'hbev_mamba_blockccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                            hidden_dim = hidden_channels*8, ssm_ratio=1.2))
            setattr(self, f'hbev_mamba_blockcw_{i}', VSSBlock(forward_type='v5', channel_first=False,
                                            hidden_dim = hidden_channels*8, ssm_ratio=1.2))


    def forward(self, x):  
        """  
        Args:  
            x: Input 3D feature map, shape (B, C, Dz, Dy, Dx)  
        Returns:  
            Fused feature map, same shape as input (B, C, Dz, Dy, Dx)  
        """  
        # Stage 1: Original resolution -> 1/2 downsampling  
        feat1 = self.encoder1(x)  # (B, C * 2, Dz/2, Dy/2, Dx/2)  
        # Stage 2: 1/2 -> 1/8 downsampling  
        feat2 = self.encoder2(feat1)  # (B, C * 4, Dz/8, Dy/8, Dx/8) 
        B, C, Dz, Dy, Dx = feat2.shape
        feat2 = feat2.permute(0, 3, 4, 1, 2)  # Change dimension order to (B, Dy/8, Dx/8, C * 4, Dz/8) 
        feat2 = feat2.reshape(B, Dy, Dx, C * Dz)  # Merge C and Dz/8, shape becomes (B, Dy/8, Dx/8, C * Dz/8 * 4)  

        feat2 = self.hbev_mamba_blockccw_1(feat2.contiguous())
        feat2 = self.hbev_mamba_blockcw_1(feat2.contiguous()).permute(0, 3, 1, 2).contiguous()
        feat2 = feat2.view(B, C, Dz, Dy, Dx)  
        feat2 = self.convmerge_feat2(feat2)
        # Stage 3: 1/8 -> 1/2 upsampling  
        up_feat2 = F.interpolate(feat2, scale_factor=4, mode='trilinear', align_corners=True) # (B, C * 2, Dz/2, Dy/2, Dx/2)  
        # Fuse 1/2 downsampled features and upsampled features  
        fused_feat = torch.cat([feat1, up_feat2], dim=1)    # (B, C*4, Dz/2, Dy/2, Dx/2)  
        fused_feat = self.convmerge_fused_feat(fused_feat)
        # Stage 4: 1/2 -> 1, upsample to original resolution  
        out = F.interpolate(fused_feat, size=x.shape[2:], mode='trilinear', align_corners=True)  # Upsample to original size  
        out = self.output_conv(out)  
        
        return out



@BACKBONES.register_module()
class FPN3D_Mamba_4D(nn.Module):  
    def __init__(self, in_channels, hidden_channels, out_channels):  
        """  
        Implement a simple 3D FPN with two scales: 1/2 and 1/8.  
        Args:  
            in_channels: Number of input feature channels C  
            out_channels: Output feature channels at each scale  
        """  
        super(FPN3D_Mamba_4D, self).__init__()  

        # Downsampling stages  
        # Scale 1/2  
        self.encoder1 = nn.Sequential(  
            # Second layer: depthwise (grouped) convolution, stride=2  
            nn.Conv3d(in_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, groups=4),  
            nn.BatchNorm3d(hidden_channels * 2),  
            # Third layer: pointwise convolution (1x1x1) for channel interaction  
            nn.Conv3d(hidden_channels * 2, hidden_channels * 2, kernel_size=1, stride=1, padding=0),  
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        # Scale 1/8  
        self.encoder2 = nn.Sequential(  
            nn.Conv3d(hidden_channels * 2, hidden_channels*4, kernel_size=3, stride=2, padding=1, groups=4),  
            nn.BatchNorm3d(hidden_channels*4),   
            nn.Conv3d(hidden_channels*4, hidden_channels*4, kernel_size=3, stride=2, padding=1, groups=4),  
            nn.BatchNorm3d(hidden_channels*4),  
            nn.Conv3d(hidden_channels*4, hidden_channels*4, kernel_size=1, stride=1, padding=0),  
            nn.BatchNorm3d(hidden_channels*4),  
            nn.ReLU(inplace=True),  
        )  
        self.convmerge_feat2 = nn.Sequential(  
            nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        self.convmerge_fused_feat = nn.Sequential(  
            nn.Conv3d(hidden_channels * 4, hidden_channels * 2, kernel_size=3, stride=1, padding=1),   
            nn.BatchNorm3d(hidden_channels * 2),  
            nn.ReLU(inplace=True),  
        )  
        # Fusion stage (upsampling)  
        # Final output convolution, upsampled to input resolution  
        self.output_conv = nn.Sequential(  
            nn.Conv3d(hidden_channels * 2, out_channels * 3 // 2, kernel_size=3, stride=1, padding=1, groups=4,bias=False),  
            nn.BatchNorm3d(out_channels * 3 // 2),  
            nn.ReLU(inplace=True),  
            nn.Conv3d(out_channels * 3 // 2, out_channels, kernel_size=5, stride=1, padding=2, groups=4,bias=False),  
            nn.BatchNorm3d(out_channels),  
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Softplus(), 
        )  

        for i in range(1,2):  
            setattr(self, f'hbev_mamba_blockccw_{i}', VSSBlock(forward_type='v4', channel_first=False,
                                            hidden_dim = hidden_channels*8, ssm_ratio=1.0))
            setattr(self, f'hbev_mamba_blockcw_{i}', VSSBlock(forward_type='v5', channel_first=False,
                                            hidden_dim = hidden_channels*8, ssm_ratio=1.0))


    def forward(self, x):  
        """  
        Args:  
            x: Input 3D feature map, shape (B, C, Dz, Dy, Dx)  
        Returns:  
            Fused feature map with same shape as input (B, C, Dz, Dy, Dx)  
        """  
        # Stage 1: Original resolution -> 1/2 downsampling  
        feat1 = self.encoder1(x)  # (B, C * 2, Dz/2, Dy/2, Dx/2)  
        # Stage 2: 1/2 -> 1/8 downsampling  
        feat2 = self.encoder2(feat1)  # (B, C * 4, Dz/8, Dy/8, Dx/8) 
        B, C, Dz, Dy, Dx = feat2.shape
        feat2 = feat2.permute(0, 3, 4, 1, 2)  # Change dimension order to (B, Dy/8, Dx/8, C * 4, Dz/8) 
        feat2 = feat2.reshape(B, Dy, Dx, C * Dz)  # Merge C and Dz/8. Now shape is (B, Dy/8, Dx/8, C * Dz/8 * 4)  

        feat2 = self.hbev_mamba_blockccw_1(feat2.contiguous())
        feat2 = self.hbev_mamba_blockcw_1(feat2.contiguous()).permute(0, 3, 1, 2).contiguous()
        feat2 = feat2.view(B, C, Dz, Dy, Dx)  
        feat2 = self.convmerge_feat2(feat2)
        # Stage 3: 1/8 -> 1/2 upsampling  
        up_feat2 = F.interpolate(feat2, scale_factor=4, mode='trilinear', align_corners=True) # (B, C * 2, Dz/2, Dy/2, Dx/2)  
        # Fuse 1/2 downsampled features and upsampled features  
        fused_feat = torch.cat([feat1, up_feat2], dim=1)    # (B, C*4, Dz/2, Dy/2, Dx/2)  
        fused_feat = self.convmerge_fused_feat(fused_feat)
        # Stage 4: 1/2 -> 1, upsample to original resolution  
        out = F.interpolate(fused_feat, size=x.shape[2:], mode='trilinear', align_corners=True)  # Upsample to original size  
        out = self.output_conv(out)  # Final convolution processing  
        
        return out  