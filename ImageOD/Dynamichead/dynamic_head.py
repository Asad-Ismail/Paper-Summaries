import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import timm

class ScaleAwareAttention(nn.Module):
    """
    Scale-aware attention combining paper theory with official implementation insights
    """
    def __init__(self, in_channels, out_channels, num_levels):
        super().__init__()
        self.out_channels = out_channels
        
        # Channel mapping layers (like official impl)
        self.channel_maps = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        # Attention computation (following official impl style)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights like official impl
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def hard_sigmoid(self, x):
        """Matches official h_sigmoid implementation"""
        return F.relu6(x + 3) / 6
    
    def forward(self, features):
        # Align dimensions like before
        mid_level = len(features) // 2
        target_size = features[mid_level].shape[-2:]
        
        # Process features level by level (like official impl)
        processed_features = []
        attention_weights = []
        
        for feat, channel_map in zip(features, self.channel_maps):
            # Channel alignment
            feat = channel_map(feat)
            
            # Spatial alignment
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            # Compute attention (following official style)
            attn = self.attention(feat)
            
            processed_features.append(feat)
            attention_weights.append(attn)
        
        # Stack features and attention weights
        stacked_features = torch.stack(processed_features, dim=1)  # [B, L, C, H, W]
        stacked_attention = torch.stack(attention_weights, dim=1)  # [B, L, 1, 1, 1]
        
        # Apply attention with hard sigmoid
        attention_weights = self.hard_sigmoid(stacked_attention)
        weighted_features = stacked_features * attention_weights
        
        return weighted_features  # [B, L, C, H, W]



backbone = timm.create_model("resnet50", pretrained=False, features_only=True)
print(backbone.feature_info.channels())
sa = ScaleAwareAttention(in_channels=backbone.feature_info.channels()[0:3],
                         out_channels=backbone.feature_info.channels()[1],
                         num_levels=3) 
x = torch.randn(2, 3, 224, 224)
features = backbone(x)
output = sa(features[0:3])  # Process last 3 levels
print(output.shape)  # [2, 256, H, W]