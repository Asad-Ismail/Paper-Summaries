# Image Generation

Image generation involves creating new images from scratch or modifying existing ones using machine learning models. This directory contains summaries and reviews of papers related to image generation techniques and models.

## Table of Contents
- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [Paper Summaries](#paper-summaries)
- [References](#references)

## Introduction
Image generation models learn to create new images by understanding patterns and structures in the training data. These models can generate realistic images, modify existing images, or create images based on textual descriptions.

## Key Concepts
- **Diffusion Models**: Models that generate images by progressively denoising a noisy image.
- **Latent Space**: A lower-dimensional space where images are encoded for efficient processing.
- **Classifier-Free Guidance (CFG)**: A technique to balance unconditional and conditional outputs during image generation.
- **ControlNet**: A model that allows additional control inputs like edge maps or sketches for more specific image generation.

## Paper Summaries

### Notable Papers

- **"High-Resolution Image Synthesis with Latent Diffusion Models"** by Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
  - **Summary**: This paper introduces the Stable Diffusion model, which uses latent space for high-resolution image synthesis.
  - **Key Concepts**:
    - **Main Architecture**: Latent Diffusion Model (LDM) with a VQ-GAN encoder-decoder.
    - **Main Idea**: Use latent space to stabilize training and generate high-resolution images.
    - **Inputs**: 512 × 512 pixel-space images.
    - **Outputs**: 64 × 64 latent images.
    - **Modules**:
      - **VQ-GAN Encoder**: Converts high-resolution images into a lower-dimensional latent space.
      - **Diffusion Model**: Operates in the latent space to progressively denoise the latent representation.
      - **VQ-GAN Decoder**: Converts the denoised latent representation back into a high-resolution image.
    - **Loss Function**: Combination of perceptual loss and adversarial loss.
  - **Link**: [Read the paper](https://arxiv.org/abs/2112.10752)

- **"Classifier-Free Diffusion Guidance"** by Jonathan Ho, Tim Salimans
  - **Summary**: This paper presents the Classifier-Free Guidance technique to improve the quality of generated images by balancing unconditional and conditional outputs.
  - **Key Concepts**:
    - **Main Architecture**: Diffusion model with guidance mechanism.
    - **Main Idea**: Combine unconditional and conditional predictions using a guidance scale.
    - **Inputs**: Noisy image, text prompt (optional).
    - **Outputs**: Denoised image.
    - **Modules**:
      - **Diffusion Model**: Generates images by denoising a noisy input.
      - **Guidance Mechanism**: Balances the influence of unconditional and conditional predictions.
    - **Loss Function**: Mean squared error (MSE) between predicted and true noise.
  - **Link**: [Read the paper](https://arxiv.org/abs/2207.12598)

  ```python
  import numpy as np

  epsilon_uc = np.random.randn(256, 256)  # Unconditional prediction
  epsilon_c = np.random.randn(256, 256)   # Conditional prediction (guided by text prompt)

  # User-specified guidance scale (CFG weight)
  beta_cfg = 7.5  # Typical values range from 5 to 15

  # Classifier-Free Guidance calculation
  epsilon_prd = epsilon_uc + beta_cfg * (epsilon_c - epsilon_uc)

  # Output: epsilon_prd is the final noise prediction used for denoising
  ```

- **"ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models"** by Lvmin Zhang, Maneesh Agrawala
  - **Summary**: This paper introduces ControlNet, which extends text-to-image models with additional control inputs like edge maps and semantic masks.
  - **Key Concepts**:
    - **Main Architecture**: Extension of pre-trained text-to-image diffusion models with additional control signals.
    - **Main Idea**: Incorporate new control inputs without disrupting the original model.
    - **Inputs**: Text prompt, control signals (e.g., edge maps, semantic masks).
    - **Outputs**: Generated image.
    - **Modules**:
      - **Pre-trained Diffusion Model**: Generates images based on text prompts.
      - **Zero Convolution Layer**: A convolution layer with weights initialized to zero, allowing the model to learn new control signals without affecting the original model's performance.
      - **Control Signal Integration**: Mechanism to incorporate additional control inputs into the generation process.
    - **Loss Function**: Combination of reconstruction loss and control signal loss.
  - **Link**: [Read the paper](https://arxiv.org/abs/2302.05543)

  <p align="center">
    <img src="imgs/controlnet.png" alt="ControlNet Architecture" width="300" height="400">
  </p>

  ```python
  import torch
  import torch.nn as nn

  class ZeroConv2d(nn.Conv2d):
      def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
          super(ZeroConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
          nn.init.zeros_(self.weight)
          if self.bias is not None:
              nn.init.zeros_(self.bias)

  # Example usage of ZeroConv2d
  zero_conv = ZeroConv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
  ```

  - **Training Strategy**:
    - ControlNet is trained in a way that new conditional controls do not override the pre-existing text-to-image generation capabilities.
    - This ensures a balanced incorporation of new control inputs while retaining the base model’s strengths.

  - **Sudden Convergence**:
    - The paper observes a phenomenon of **sudden convergence**, where the model demonstrates rapid performance improvement during later stages of training.
    - This highlights the effectiveness of the zero convolution layer in maintaining equilibrium between new control signals and the original model.

  <p align="center">
    <img src="imgs/sudden_convergence.png" alt="Sudden Convergence" width="300" height="300">
  </p>

## References
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
- [VQ-GAN: Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

