import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from einops import rearrange, repeat
import math
from typing import List, Tuple, Optional



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, causal=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.causal = causal
        
    def forward(self, x):
        # Causal self-attention
        attn_mask = None
        if self.causal:
            seq_len = x.shape[1]
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len), diagonal=1
            ).bool()
            attn_mask = attn_mask.to(x.device)
            
        # Apply attention
        x = x + self.attn(
            self.norm1(x), 
            self.norm1(x), 
            self.norm1(x),
            attn_mask=attn_mask
        )[0]
        
        # Apply MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class UnifiedVideoPyramidalFlow(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_spatial_stages: int = 4,
        num_history_frames: int = 3,
        base_resolution: int = 32,
        depth: int = 6,
        heads: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.num_spatial_stages = num_spatial_stages
        self.num_history_frames = num_history_frames
        self.base_resolution = base_resolution
        
        # Main transformer backbone
        self.transformer = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=heads, causal=True)
            for _ in range(depth)
        ])
        
        # Input/output projections
        self.input_proj = nn.Linear(3, dim)
        self.output_proj = nn.Linear(dim, 3)
        
        # Position encodings
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, num_history_frames + 1, dim)
        )
        self.spatial_pos_embeds = nn.ModuleList([
            nn.Parameter(torch.randn(1, (base_resolution * 2**i)**2, dim))
            for i in range(num_spatial_stages)
        ])
        
        # Flow prediction network
        self.flow_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_spatial_stages)
        ])

    def get_endpoints(
        self,
        x1: torch.Tensor,
        stage_k: int,
        sk: float,
        ek: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute start and end points for a stage using coupled noise sampling
        Implements equations 9 and 10
        """
        # Sample shared noise for coupling
        noise = torch.randn_like(x1)
        
        # End point (equation 9)
        x_end = F.interpolate(
            x1, scale_factor=1/2**stage_k, mode='bilinear', align_corners=False
        )
        x_end = ek * x_end + (1 - ek) * noise
        
        # Start point (equation 10)
        x_start = F.interpolate(
            x1, scale_factor=1/2**(stage_k + 1), mode='bilinear', align_corners=False
        )
        x_start = F.interpolate(
            x_start, scale_factor=2, mode='bilinear', align_corners=False
        )
        x_start = sk * x_start + (1 - sk) * noise
        
        return x_start, x_end

    def process_temporal_condition(
        self,
        history_frames: List[torch.Tensor],
        stage_k: int,
        training: bool = True
    ) -> List[torch.Tensor]:
        """
        Process history frames with pyramidal temporal condition
        Implements equations 16 and 17
        """
        processed = []
        
        for i, frame in enumerate(history_frames):
            # Earlier frames use lower resolution
            scale = 2 ** (stage_k + (1 if i < len(history_frames)-1 else 0))
            
            # Downsample
            down_res = self.base_resolution // scale
            x_down = F.interpolate(
                frame, size=(down_res, down_res),
                mode='bilinear', align_corners=False
            )
            
            # Add noise during training (equation 16)
            if training:
                noise_scale = 0.1 * (1 - i/len(history_frames))
                x_down = x_down + torch.randn_like(x_down) * noise_scale
                
            processed.append(x_down)
            
        return processed

    def renoise_jump_point(
        self,
        x: torch.Tensor,
        sk: float
    ) -> torch.Tensor:
        """
        Apply renoising at jump points between stages
        Implements equation 15
        """
        # Upsample
        x_up = F.interpolate(x, scale_factor=2, mode='nearest')
        
        # Coefficients from equation 15
        scale_coef = (1 + sk) / 2
        noise_coef = math.sqrt(3 * (1 - sk)) / 2
        
        # Generate blockwise correlated noise (equation 14)
        noise = torch.randn_like(x_up)
        noise = self.apply_block_correlation(noise, gamma=-1/3)
        
        return scale_coef * x_up + noise_coef * noise

    def apply_block_correlation(
        self,
        noise: torch.Tensor,
        gamma: float = -1/3
    ) -> torch.Tensor:
        """
        Apply blockwise correlation to noise
        Implements equation 14
        """
        B, C, H, W = noise.shape
        
        # Reshape to blocks
        noise = noise.view(B, C, H//2, 2, W//2, 2)
        noise = noise.permute(0, 1, 2, 4, 3, 5).contiguous()
        noise = noise.view(B, C, -1, 4)
        
        # Correlation matrix
        corr = torch.ones(4, 4, device=noise.device) * gamma
        corr.diagonal().fill_(1)
        
        # Apply correlation
        noise = torch.matmul(noise, corr)
        noise = noise.view(B, C, H//2, W//2, 2, 2)
        noise = noise.permute(0, 1, 2, 4, 3, 5).contiguous()
        noise = noise.view(B, C, H, W)
        
        return noise

    def forward_training(
        self,
        history_frames: List[torch.Tensor],
        current_frame: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training
        Combines spatial and temporal pyramids
        """
        # Determine stage
        stage_k = int(t * self.num_spatial_stages)
        sk = stage_k / self.num_spatial_stages
        ek = (stage_k + 1) / self.num_spatial_stages
        
        # Get endpoints for spatial pyramid
        x_start, x_end = self.get_endpoints(current_frame, stage_k, sk, ek)
        
        # Process temporal condition
        history = self.process_temporal_condition(
            history_frames, stage_k, training=True
        )
        
        # Interpolate current frame position
        local_t = (t - sk) / (ek - sk)
        x_t = (1 - local_t) * x_start + local_t * x_end
        
        # Embed all frames
        embedded_sequence = []
        
        # Embed and position-encode history
        for i, frame in enumerate(history):
            h_emb = self.input_proj(rearrange(frame, 'b c h w -> b (h w) c'))
            h_emb = h_emb + self.temporal_pos_embed[:, i]
            h_emb = h_emb + self.spatial_pos_embeds[stage_k][:, :h_emb.shape[1]]
            embedded_sequence.append(h_emb)
        
        # Embed and position-encode current frame
        x_emb = self.input_proj(rearrange(x_t, 'b c h w -> b (h w) c'))
        x_emb = x_emb + self.temporal_pos_embed[:, -1]
        x_emb = x_emb + self.spatial_pos_embeds[stage_k]
        embedded_sequence.append(x_emb)
        
        # Concatenate sequence
        x = torch.cat(embedded_sequence, dim=1)
        
        # Apply transformer layers
        for block in self.transformer:
            x = block(x)
        
        # Extract prediction
        pred = x[:, -x_emb.shape[1]:]
        pred = self.output_proj(pred)
        pred = rearrange(pred, 'b (h w) c -> b c h w', 
                        h=int(math.sqrt(pred.shape[1])))
        
        return pred

    @torch.no_grad()
    def sample(
        self,
        first_frame: torch.Tensor,
        num_frames: int,
        guidance_scale: float = 7.5,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Autoregressive video sampling
        """
        device = first_frame.device
        video_frames = [first_frame]
        
        for i in range(num_frames - 1):
            # Get history frames
            history = video_frames[-self.num_history_frames:]
            
            # Initialize from noise
            current = torch.randn_like(first_frame)
            
            # Generate through pyramid stages
            for k in range(self.num_spatial_stages-1, -1, -1):
                sk = k / self.num_spatial_stages
                ek = (k + 1) / self.num_spatial_stages
                
                # Steps within stage
                for t in torch.linspace(sk, ek, num_steps//self.num_spatial_stages):
                    t = t.to(device)
                    
                    # Get prediction
                    pred = self.forward_training(history, current, t)
                    
                    # Apply classifier-free guidance
                    if guidance_scale > 1:
                        null_cond = [torch.zeros_like(h) for h in history]
                        null_pred = self.forward_training(null_cond, current, t)
                        pred = null_pred + guidance_scale * (pred - null_pred)
                    
                    current = pred
                
                # Apply renoising between stages
                if k > 0:
                    current = self.renoise_jump_point(current, sk)
            
            video_frames.append(current)
        
        return torch.stack(video_frames, dim=1)

def train_video_flow(
    model: UnifiedVideoPyramidalFlow,
    train_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """
    Training loop for unified video pyramidal flow
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            # Assume batch contains video clips
            videos = batch['video'].to(device)  # [B, T, C, H, W]
            B, T, C, H, W = videos.shape
            
            # Randomly sample sequence length
            seq_len = torch.randint(2, T+1, (1,)).item()
            
            optimizer.zero_grad()
            
            # Random timestamp for pyramid stage
            t = torch.rand(1, device=device)
            
            # Get history and current frames
            current_idx = torch.randint(model.num_history_frames, seq_len, (1,)).item()
            history_frames = [
                videos[:, current_idx-i-1] for i in range(model.num_history_frames)
            ]
            current_frame = videos[:, current_idx]
            
            # Forward pass
            pred = model.forward_training(history_frames, current_frame, t)
            
            # Compute loss
            loss = F.mse_loss(pred, current_frame)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log epoch results
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

# Example usage:
if __name__ == "__main__":
    # Create model
    model = UnifiedVideoPyramidalFlow(
        dim=512,
        num_spatial_stages=4,
        num_history_frames=3,
        base_resolution=32
    )
    
    # Assume we have a dataloader
    # train_dataloader = get_video_dataloader()
    
    # Train model
    # train_video_flow(model, train_dataloader)
    
    # Generate video
    first_frame = torch.randn(1, 3, 256, 256)
    video = model.sample(
        first_frame=first_frame,
        num_frames=16,
        guidance_scale=7.5
    )