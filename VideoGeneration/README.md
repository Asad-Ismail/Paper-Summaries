## Video Generation Models

### Basics of different Image Generation videos

### Diffusion Models:
- `x_0`: Original clean data (e.g., a clean image/video frame)
- `x_t`: Noisy data at timestep t (partially noised version of x_0)
- `t`: Timestep (ranges from 0 to 1 or 1 to 0 depending on implementation)
- `T`: Total number of diffusion steps

```python
# In diffusion, we start with x_0 (clean data) and gradually add noise:
def forward_diffusion(x_0, t):
    # x_0: Original clean data
    # t: Current timestep
    alpha_t = noise_schedule(t)  # How much noise to add at this step
    epsilon = torch.randn_like(x_0)  # Random noise
    
    # x_t is a mixture of original data and noise
    x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
    return x_t
```



### 1. Pyramidal Flow Matching for Efficient Video Generative Modeling (2024)

[Paper](https://arxiv.org/pdf/2410.05954)


Key Features:

Autoregressive video generation
Multiple resolution stages
Classifier-free guidance for better quality
Temporal condition handling
