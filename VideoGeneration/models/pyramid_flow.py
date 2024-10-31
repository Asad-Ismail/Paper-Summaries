import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidalFlowMatching(nn.Module):
    def __init__(self, num_stages=4, base_channels=64):
        """
        Initialize Pyramidal Flow Matching.
        
        Args:
            num_stages (int): Number of resolution stages (K in the paper)
            base_channels (int): Base number of channels for the model
        """
        super().__init__()
        self.num_stages = num_stages
        
        # Create downsampling and upsampling layers for each stage
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
            for _ in range(num_stages)
        ])
        
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
            for _ in range(num_stages)
        ])

    def downsample(self, x, scale):
        """
        Downsample input tensor by a given scale factor.
        
        Args:
            x (torch.Tensor): Input tensor
            scale (int): Scale factor (power of 2)
        """
        for i in range(int(torch.log2(torch.tensor(scale)))):
            x = self.downsample_layers[i](x)
        return x

    def upsample(self, x, scale):
        """
        Upsample input tensor by a given scale factor.
        
        Args:
            x (torch.Tensor): Input tensor
            scale (int): Scale factor (power of 2)
        """
        for i in range(int(torch.log2(torch.tensor(scale)))):
            x = self.upsample_layers[i](x)
        return x

    def interpolate_stage(self, x_start, x_end, t, stage):
        """
        Interpolate between two resolutions within a stage.
        
        Args:
            x_start (torch.Tensor): Starting point tensor
            x_end (torch.Tensor): Ending point tensor
            t (float): Timestep within the stage (0 to 1)
            stage (int): Current stage number
        """
        # Equation 6 from the paper
        x_start_down = self.downsample(x_start, 2**(stage+1))
        x_start_up = self.upsample(x_start_down, 2)
        x_end_down = self.downsample(x_end, 2**stage)
        
        return t * x_end_down + (1 - t) * x_start_up

    def forward(self, x0, x1, t):
        """
        Forward pass implementing the pyramidal flow.
        
        Args:
            x0 (torch.Tensor): Starting point
            x1 (torch.Tensor): Target point
            t (torch.Tensor): Global timestep (0 to 1)
        """
        # Calculate stage boundaries
        stage_width = 1.0 / self.num_stages
        stage = int(t / stage_width)
        local_t = (t - stage * stage_width) / stage_width  # t' in equation 6
        
        # Handle boundary case
        if stage >= self.num_stages:
            stage = self.num_stages - 1
            local_t = 1.0
            
        # Calculate interpolation for current stage
        result = self.interpolate_stage(x0, x1, local_t, stage)
        
        # Add noise at jump points if needed (Figure 2b)
        if abs(local_t - 1.0) < 1e-6 and stage < self.num_stages - 1:
            noise_scale = 0.1  # This can be adjusted
            noise = torch.randn_like(result) * noise_scale
            result = result + noise
            
        return result