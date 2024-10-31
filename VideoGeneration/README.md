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

**Training**: 1000+ steps recommended
**Inference**: Can use DDIM for fewer steps (50-250)
Quality directly correlates with step count

### Flow Matching
- `x_0`: Starting point (usually random noise)
- `x_1`: Target data point (clean data)
- `x_t`: Intermediate point along the path from x_0 to x_1
- `v_t`: Velocity field (direction of flow)

```python
# In flow matching, we learn a direct path:
def flow_matching(x_0, x_1, t):
    # x_0: Starting point (noise)
    # x_1: Target point (clean data)
    # t: Position along the path (0 to 1)
    
    # Linear interpolation between start and end
    x_t = (1 - t) * x_0 + t * x_1
    
    # Predict velocity field
    v_t = flow_network(x_t, t)
    return v_t
```

**Training**: Random sampling within stages
**Inference**: 5-20 steps per stage usually sufficient
More steps in early stages often beneficial




### 1. Pyramidal Flow Matching for Efficient Video Generative Modeling (2024)

[Paper](https://arxiv.org/pdf/2410.05954)


Key Features:

Autoregressive video generation
Multiple resolution stages
Classifier-free guidance for better quality
Temporal condition handling
