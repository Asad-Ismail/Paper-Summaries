import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import timm

class ScaleAwareAttention(nn.Module):
    """Scale-aware attention to fuse features across levels"""
    def __init__(self, in_channels, out_channels,num_levels):
        super().__init__()
        self.out_channels = out_channels
        
        # Channel mapping layers for each feature level
        self.channel_maps = nn.ModuleList([nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels ])
        
        # Scale attention function
        self.scale_function = nn.Conv2d(num_levels, 1, 1)
        
    def forward(self, features):
        """
        Args:
            features: List[Tensor] of shape [B, Ci, Hi, Wi] from different levels
            where Ci varies per level
        """
        # Find target spatial size (using median level's resolution)
        mid_level = len(features) // 2
        target_size = features[mid_level].shape[-2:]
        
        # Align both spatial and channel dimensions
        aligned_features = []
        for feat, channel_map in zip(features, self.channel_maps):
            # First map channels to common dimension
            feat = channel_map(feat)  # Now [B, out_channels, Hi, Wi]
            
            # Then align spatial dimensions
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            aligned_features.append(feat)
            
        # Stack aligned features: [B, L, C, H, W]
        x = torch.stack(aligned_features, dim=1)
        B, L, C, H, W = x.shape
        
        # Global pooling across spatial dims and channels
        pooled = x.mean(dim=(2,3, 4))  # [B, L]
        
        # Learn importance weights per level
        weights = self.scale_function(pooled.unsqueeze(-1).unsqueeze(-1))  # [B, L, 1, 1]
        weights = (weights + 1) / 2  # Hard sigmoid
        weights = torch.clamp(weights, 0, 1)
        
        # Weight features across levels
        weighted = x * weights.view(B, 1, 1, 1, 1)
        return weighted



backbone = timm.create_model("resnet50", pretrained=False, features_only=True)
print(backbone.feature_info.channels())
sa = ScaleAwareAttention(in_channels=backbone.feature_info.channels()[0:3],
                         out_channels=backbone.feature_info.channels()[1],
                         num_levels=3) 
x = torch.randn(2, 3, 224, 224)
features = backbone(x)
output = sa(features[0:3])  # Process last 3 levels
print(output.shape)  # [2, 256, H, W]