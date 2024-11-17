import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Optional, Tuple
from torchvision.ops import box_convert, box_iou

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CrossModalityAttention(nn.Module):
    """Cross Modality Attention module for vision-language fusion"""
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections for cross attention
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, Nx, C = x.shape
        _, Ny, _ = y.shape

        # Project queries, keys, values
        q = self.q_linear(x).reshape(B, Nx, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(y).reshape(B, Ny, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(y).reshape(B, Ny, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, Nx, C)
        out = self.out_proj(out)
        
        return out


class DeepFusionLayer(nn.Module):
    """Single deep fusion layer combining visual and text features"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cross_attn = CrossModalityAttention(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, vis_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # Cross attention
        vis_features = vis_features + self.dropout(
            self.cross_attn(self.norm1(vis_features), text_features)
        )
        
        # Self attention
        vis_features = vis_features + self.dropout(
            self.self_attn(self.norm2(vis_features))[0]
        )
        
        # FFN
        vis_features = vis_features + self.dropout(
            self.ffn(self.norm3(vis_features))
        )
        
        return vis_features

class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature processing"""
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        last_inner = self.inner_blocks[-1](features[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))

        for feature, inner_block, layer_block in zip(
            features[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_top_down = F.interpolate(last_inner, size=feature.shape[-2:], mode="nearest")
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))

        return torch.cat([result.flatten(2).transpose(1, 2) for result in results], dim=1)


class DyHeadBlock(nn.Module):
    """DyHead Block for dynamic feature adaptation"""
    def __init__(self, channels: int = 256, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        
        # Spatial Attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Channel Attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # Feature Transform
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # Scale Attention
        self.scale_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply spatial attention
        spatial_weight = self.spatial_attn(x)
        x = x * spatial_weight
        
        # Apply channel attention
        channel_weight = self.channel_attn(x)
        x = x * channel_weight
        
        # Transform features
        x = self.transform(x)
        
        # Apply scale attention
        scale_weight = self.scale_attn(x)
        x = x * scale_weight
        
        return x

class DyHead(nn.Module):
    """Dynamic Head for object detection"""
    def __init__(
        self, 
        in_channels: int = 256,
        num_blocks: int = 6,
        num_heads: int = 8,
        num_levels: int = 5
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # DyHead blocks for each level
        self.blocks = nn.ModuleList([
            DyHeadBlock(in_channels, num_heads) 
            for _ in range(num_blocks)
        ])
        
        # Level embeddings for scale-aware features
        self.level_embeddings = nn.Parameter(
            torch.zeros(num_levels, in_channels)
        )
        
        # Fusion layers
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.fusion_norm = nn.BatchNorm2d(in_channels)
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Add level embeddings
        for i in range(len(features)):
            features[i] = features[i] + self.level_embeddings[i].view(1, -1, 1, 1)
        
        # Apply DyHead blocks
        for block in self.blocks:
            new_features = []
            for level, feature in enumerate(features):
                feature = block(feature)
                new_features.append(feature)
            features = new_features
            
        # Feature fusion
        for i in range(len(features)):
            features[i] = self.fusion_norm(
                self.fusion_conv(features[i])
            )
            
        return features

class DeepFusionDetector(nn.Module):
    def __init__(
        self,
        num_classes: int = 365,
        hidden_dim: int = 256,
        num_fusion_layers: int = 3,
        num_dyhead_blocks: int = 6,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Swin-Tiny backbone
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3)
        )
        
        # Get backbone channels
        backbone_channels = [
            self.backbone.feature_info[i]['num_chs'] 
            for i in range(len(self.backbone.feature_info))
        ]
        
        # FPN
        self.fpn = FeaturePyramidNetwork(backbone_channels, hidden_dim)
        
        # DyHead
        self.dyhead = DyHead(
            in_channels=hidden_dim,
            num_blocks=num_dyhead_blocks,
            num_levels=len(backbone_channels)
        )
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Linear(768, hidden_dim)
        
        # Deep fusion layers
        self.fusion_layers = nn.ModuleList([
            DeepFusionLayer(hidden_dim) 
            for _ in range(num_fusion_layers)
        ])
        
        # Detection heads
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        )
        
        self.box_head = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # Initialize weights
        self._init_weights()
        
    def forward(
        self, 
        images: torch.Tensor,
        text_tokens: Dict[str, torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        # Extract backbone features
        features = self.backbone(images)
        
        # FPN processing
        fpn_features = self.fpn(features)
        
        # DyHead processing
        dyhead_features = self.dyhead(fpn_features)
        
        # Flatten and combine features
        batch_size = images.shape[0]
        visual_features = torch.cat([
            f.flatten(2).transpose(1, 2) 
            for f in dyhead_features
        ], dim=1)
        
        # Text processing
        text_output = self.text_encoder(**text_tokens)
        text_features = self.text_proj(text_output.last_hidden_state)
        
        # Deep fusion
        for fusion_layer in self.fusion_layers:
            visual_features = fusion_layer(
                visual_features, text_features
            )
        
        # Predictions
        class_logits = self.class_head(visual_features)
        box_coords = self.box_head(visual_features).sigmoid()
        
        out = {
            'pred_logits': class_logits,
            'pred_boxes': box_coords,
            'text_features': text_features,
            'visual_features': visual_features
        }
        
        if targets is not None:
            loss_dict = self.compute_loss(out, targets)
            return out, loss_dict
            
        return out

def build_model(pretrained: bool = True,checkpoint_path: Optional[str] = None):
    model = DeepFusionDetector(
        pretrained=pretrained,
        num_dyhead_blocks=6
    )
    if checkpoint_path is not None:
        model.load_objects365_weights(checkpoint_path)
    
    return model


def get_sample_input():
    """Helper function to create sample inputs for testing"""
    images = torch.randn(2, 3, 224, 224)
    text = ["person. car. dog"]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_tokens = tokenizer(text, return_tensors='pt', padding=True)
    return images, text_tokens

if __name__ == "__main__":
    model = build_model(pretrained=True)
    images, text_tokens = get_sample_input()
    outputs = model(images, text_tokens)
    print("Model output keys:", outputs.keys())