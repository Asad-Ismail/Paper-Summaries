import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import timm
from typing import Dict, List


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        super(Conv, self).__init__()
    
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False 
        )
        
        # GroupNorm with 16 groups
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        # kwargs are ignored as regular conv doesn't need offset/mask
        x = self.conv(input.contiguous())
        x = self.bn(x)
        return x
    
    
class MultiLevelFusion(nn.Module):
    """
    Handles multi-level feature fusion with channel alignment, upsampling and downsampling
    """
    def __init__(self, conv_func, out_channels: int,in_channels: Dict[str, int]=None):
        """
        Args:
            conv_func: Convolution function to use for convolution e.g normal or deformable
            out_channels: Number of output channels for all levels
            in_channelst: Optional  provides dictionary mapping level names to their input channels
                            e.g., {'p3': 256, 'p4': 512, 'p5': 1024}
        """
        super().__init__()
        
        # Channel mapping convolutions for each level
        if in_channels:
            self.channel_maps = nn.ModuleDict({
                name: nn.Conv2d(in_ch, out_channels, 1) 
                for name, in_ch in in_channels_dict.items()
            })
        else:
            self.channel_maps = None
        
        self.next_conv = conv_func(out_channels, out_channels, kernel_size=3, 
                                 stride=1, padding=1)
        self.curr_conv = conv_func(out_channels, out_channels, kernel_size=3, 
                                 stride=1, padding=1)
        self.prev_conv = conv_func(out_channels, out_channels, kernel_size=3, 
                                 stride=2, padding=1)
        
        
    def _process_level(self, 
                      features: Dict[str, torch.Tensor],
                      aligned_features: Dict[str, torch.Tensor],
                      level_name: str,
                      conv_args: dict) -> List[torch.Tensor]:
        """Process single level and collect features from adjacent levels"""
        curr_feat = aligned_features[level_name] 
        feature_names = list(features.keys())
        level_idx = feature_names.index(level_name)
        target_size = curr_feat.shape[-2:]
        
        # Collect features for this level
        level_features = []
        
        # 1. Current level feature
        level_features.append(self.curr_conv(curr_feat, **conv_args))
        
        # 2. Previous level feature (if exists)
        if level_idx > 0:
            prev_name = feature_names[level_idx - 1]
            prev_feat = aligned_features[prev_name]  
            level_features.append(self.prev_conv(prev_feat, **conv_args))
            
        # 3. Next level feature (if exists)
        if level_idx < len(features) - 1:
            next_name = feature_names[level_idx + 1]
            next_feat = aligned_features[next_name] 
            next_conv_out = self.next_conv(next_feat, **conv_args)
            next_feat_up = F.interpolate(next_conv_out, 
                                       size=target_size,
                                       mode='bilinear', 
                                       align_corners=False)
            level_features.append(next_feat_up)
            
        return level_features
        
    def forward(self, features: Dict[str, torch.Tensor], conv_args: dict = None) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            features: Dictionary mapping level names to features
                     e.g., {'p3': tensor_p3, 'p4': tensor_p4, 'p5': tensor_p5}
            conv_args: Optional arguments for convolution

        Returns:
           freatures: Dictonary with each vlaue contaiing list of featues of tensor 
           current leve, one level below and one level above. 
        """
        if conv_args is None:
            conv_args = {}
            
        # First align all channels
        aligned_features = {
            name: self.channel_maps[name](feat)
            for name, feat in features.items()
        }
            
        # Process each level
        output = {}
        for level_name in features.keys():
            output[level_name] = self._process_level(features, aligned_features, level_name, conv_args)
            
        return output
    



# Example usage
if __name__ == "__main__":
    # Create dummy features with different channels
    batch_size = 2
    features = {
        'p3': torch.randn(batch_size, 256, 64, 64),
        'p4': torch.randn(batch_size, 512, 32, 32),
        'p5': torch.randn(batch_size, 1024, 16, 16)
    }
    
    # Channel configuration
    in_channels_dict = {
        'p3': 256,
        'p4': 512,
        'p5': 1024
    }
    out_channels = 256
    
    # Create fusion module
    fusion = MultiLevelFusion(Conv,out_channels=out_channels,in_channels=in_channels_dict)
    
    # Process features
    output = fusion(features)
    
    # Print shapes
    print("Input shapes:")
    for k, v in features.items():
        print(f"{k}: {v.shape}")
        
    print("\nOutput shapes:")
    for k, v in output.items():
        print(f"{k}: {[f.shape for f in v]}")



#backbone = timm.create_model("resnet50", pretrained=False, features_only=True)
#print(backbone.feature_info.channels())
#sa = ScaleAwareAttention(in_channels=backbone.feature_info.channels()[0:3],
#                         out_channels=backbone.feature_info.channels()[1],
#                         num_levels=3) 
#x = torch.randn(2, 3, 224, 224)
#features = backbone(x)
#output = sa(features[0:3])  # Process last 3 levels
#print(output.shape)  # [2, 256, H, W]