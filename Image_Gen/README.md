

## Stable Diffusion

Image diffusion models learn to progressively denoise
images and generate samples from the training domain. The
denoising process can occur in pixel space or in a latent
space encoded from training data. Stable Diffusion uses
latent images as the training domain as working in this space
has been shown to stabilize the training process [72]. Specifically, Stable Diffusion uses a pre-processing method similar
to VQ-GAN [19] to convert 512 × 512 pixel-space images
into smaller 64 × 64 latent images. 

### Controlled Generation

#### 1. Classifier-Free Guidance (CFG) in Stable Diffusion

**Classifier-Free Guidance (CFG)** is a technique that helps diffusion models like Stable Diffusion improve the quality of generated images by controlling the balance between **unconditional** and **conditional** outputs during the denoising process. This allows the model to generate images that follow the guidance (such as a text prompt) while maintaining some flexibility to avoid overfitting the guidance.

The formula for CFG is as follows:



Where:
- **ε_prd**: The final predicted output used in the denoising process.
- **ε_uc**: The unconditional noise prediction (without any conditioning).
- **ε_c**: The conditional noise prediction (with the text prompt or guidance).
- **β_cfg**: The guidance scale, a user-specified parameter that controls how strongly the model should follow the conditional output.

#### How CFG Works

The basic idea is to combine the **unconditional** output and the **conditional** output using the **guidance scale** (β_cfg). By adjusting this guidance scale, you can control how much influence the text prompt (or other conditioning) has on the generated image.

- **When β_cfg = 0**: The model completely ignores the guidance and uses only the unconditional prediction.
- **When β_cfg = 1**: The model gives equal importance to the unconditional and conditional predictions.
- **When β_cfg > 1**: The model puts more emphasis on the guidance, making the output more strongly aligned with the prompt. However, setting β_cfg too high may lead to artifacts or unrealistic images.

$$\epsilon_{\text{prd}} = \epsilon_{\text{uc}} + \beta_{\text{cfg}} (\epsilon_{\text{c}} - \epsilon_{\text{uc}})$$


In essense

```python
epsilon_uc = np.random.randn(256, 256)  # Unconditional prediction
epsilon_c = np.random.randn(256, 256)   # Conditional prediction (guided by text prompt)

# User-specified guidance scale (CFG weight)
beta_cfg = 7.5  # Typical values range from 5 to 15

# Classifier-Free Guidance calculation
epsilon_prd = epsilon_uc + beta_cfg * (epsilon_c - epsilon_uc)

# Output: epsilon_prd is the final noise prediction used for denoising
```
#### Guidance scale 
Tuning the Guidance Scale
Lower values (e.g., 1-5) produce more generic images.
Higher values (e.g., 7-15) result in stronger adherence to the prompt, but extreme values might distort the output.
A good starting point is to use a β_cfg value between 7 and 10 for typical text-to-image generation tasks.


2. #### ControlNet
[Paper](https://arxiv.org/abs/2302.05543)

ControlNet enables diffusion models to generate images that are not only conditioned on text prompts but also other forms of control inputs such as edge maps, sketches, or semantic masks. This allows for more specific control over the generated images.


1. **ControlNet Architecture**: 
   - Built upon pre-trained text-to-image diffusion models.
   - Extends the model by adding new control signals (e.g., edge maps, semantic masks).
   - Maintains the original model’s text-to-image generation ability while incorporating new input modalities.
   <p align="center">
    <img src="imgs/controlnet.png" alt="ControlNet Architecture" width="300" height="400">
</p>


2. **Zero Convolution Layer**:
   - A unique feature of ControlNet is the use of a "zero convolution" layer, with weights initialized to 0.
   - Despite the initial zero weights, the layer learns from non-zero inputs, allowing ControlNet to incorporate new control signals without disrupting the original model. Zero Convolution layer in the ControlNet GitHub repository: [Zero Conv Explanation](https://github.com/lllyasviel/ControlNet?tab=readme-ov-file).

3. **Training Strategy**:
   - ControlNet is trained in a way that new conditional controls do not override the pre-existing text-to-image generation capabilities.
   - This ensures a balanced incorporation of new control inputs while retaining the base model’s strengths.

4. **Sudden Convergence**:
   - The paper observes a phenomenon of **sudden convergence**, where the model demonstrates rapid performance improvement during later stages of training.
   - This highlights the effectiveness of the zero convolution layer in maintaining equilibrium between new control signals and the original model.

   <p align="center">
    <img src="imgs/sudden_convergence.png" alt="Sudden Convergence" width="300" height="300">
</p>

