## Video Generation Models

### Basics of different Image Generation videos

### GANs

### GANs (Generative Adversarial Networks)


```python
class GAN:
    def train_step(self, real_data):
        # Generator: Direct mapping from noise to data
        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)
        
        # Two-player game
        disc_real = discriminator(real_data)
        disc_fake = discriminator(fake_data)
        
        # Adversarial losses
        generator_loss = -torch.mean(torch.log(disc_fake))
        discriminator_loss = -torch.mean(torch.log(disc_real) + torch.log(1 - disc_fake))
```

Features:
Based on game theory (minimax game)
Direct transformation from noise to data
No explicit probability modeling
❌ Mode collapse issues
❌ Training instability
✅ Fast generation
✅ Sharp results

### Diffusion Models:
- `x_0`: Original clean data (e.g., a clean image/video frame)
- `x_t`: Noisy data at timestep t (partially noised version of x_0)
- `t`: Timestep (ranges from 0 to 1 or 1 to 0 depending on implementation)
- `T`: Total number of diffusion steps

```python

class Diffusion:
    def forward_diffusion(x_0, t):
        # x_0: Original clean data
        # t: Current timestep
        alpha_t = noise_schedule(t) 
        epsilon = torch.randn_like(x_0)  # Random noise
        
        # x_t is a mixture of original data and noise
        x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        return x_t
    def train_step(self, real_data):
        # Gradual noise addition with known schedule
        t = torch.randint(0, num_steps, (batch_size,))
        noised_data = forward_diffusion(real_data, t)
        
        # Learn to predict noise
        predicted_noise = model(noised_data, t)
        loss = F.mse_loss(predicted_noise, noise)
    # In diffusion, we start with x_0 (clean data) and gradually add noise:
```

**Training**: 1000+ steps recommended
**Inference**: Can use DDIM for fewer steps (50-250)
Quality directly correlates with step count

Based on thermodynamics (heat equation)
Gradually converts data to/from noise
Explicit probability modeling
✅ Stable training
✅ High quality results
❌ Slow generation
✅ Good mode coverage



### Flow Matching
- `x_0`: Starting point (usually random noise)
- `x_1`: Target data point (clean data)
- `x_t`: Intermediate point along the path from x_0 to x_1
- `v_t`: Velocity field (direction of flow)

```python
# In flow matching, we learn a direct path:

class Flow:
    def flow_matching(x_0, x_1, t):
        # x_0: Starting point (noise)
        # x_1: Target point (clean data)
        # t: Position along the path (0 to 1)
        
        # Linear interpolation between start and end
        x_t = (1 - t) * x_0 + t * x_1
        
        # Predict velocity field
        v_t = flow_network(x_t, t)
        return v_t
    def train_step(self, real_data):
        # Learn direct path through probability space
        t = torch.rand(batch_size, 1)
        noise = torch.randn_like(real_data)
        
        # Interpolate between noise and data
        x_t = (1-t) * noise + t * real_data
        velocity = flow_matching(noise,real_data, t)
        
        # Match velocity field
        target_velocity = real_data - noise
        loss = F.mse_loss(velocity, target_velocity)
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
