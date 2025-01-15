# 2D Virtual Try On

## Table of Contents
- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [Paper Summaries](#paper-summaries)
- [References](#references)

## Introduction
Image-based virtual try-on can be regarded as a conditional image generation task that uses in-shop clothing image `Ic` and person image `Ip` as raw data. The raw data is pre-processed as conditioned information to guide the model for generating try-on images `Itry−on = G(Ip, Ic)`. Three key modules are usually involved in image-based virtual try-on:

## Key Concepts
### Try-On Indication
This module’s primary function is to create a "prior" about how the clothing should be warped and fused onto the person’s body in the virtual try-on process. 

- **Input Data**:
  - **Semantic information**: (like arms, torso, legs) in a segmented form.
  - **DensePose**: Provides a more precise, dense mapping of a 3D body to a 2D image.
  - **OpenPose**: Tracks skeletal and pose key points (like joints) of the person.

- **Output**: Predicts the spatial structure of the person’s body under the try-on state.

### Cloth Warping
Transforms the clothing image to the spatial distribution under the try-on state. Inputs include clothing images and person body features such as cloth-agnostic person representation or dressed person representation obtained in the Try-On Indication module.

### Try-On Module
Generates the final try-on image by fusing the person body and clothing features. Interpolation or generative networks are designed for this module.

### Cloth-agnostic Person Representation
Cloth-agnostic person representations are used to preserve body features (like pose, shape, and semantics) while removing old clothing. These representations can be categorized into the following types:

- RGB Image (P1, P2, P3, P4)
  - (P1): Original person image with existing clothing.
  - (P2): Image with only the head visible, parsed using human parsing techniques.
  - (P3): Masked image where try-on areas are covered in gray.
  - (P4): Deleted clothing areas, keeping only the background visible.
  
- Pose Keypoints (P5)
  - Pose keypoints provide the positions of 18 body parts (like shoulders, hips, etc.). These keypoints are crucial for guiding the deformation of the new clothing to align with the person’s body.

- Silhouette (P6, P7)
  - Silhouettes offer the contour of the body, giving rough shape and pose information. This representation provides an outline of the person without showing details of the original clothing.

- DensePose (P8, P9)
  - (P8): Semantic parsing from DensePose, giving detailed 3D body shape.
  - (P9): UV mapping coordinates for aligning a 3D model with the body, offering a more accurate body shape under clothing.

- Semantic Segmentation (P10, P11, P12)
  - (P10): Provides segmentation with contours of the original clothing, but this can affect the try-on process.
  - (P11): Combines skin and clothing areas, eliminating the influence of the original clothing.
  - (P12): Completely removes the original clothing, keeping only unrelated semantic regions. However, this also removes some important body shape information.

- Landmarks (P13)
  - Landmark representations are used for shape alignment. They guide clothing deformation by ensuring that key points (e.g., shoulders, neck) on the clothing align with the corresponding body points.

<p align="center">
  <img src="imgs/person_agnostic.png" alt="Person Agnostic" width="410" height="250">
</p>

### Cloth Warping Methods
- **Thin Plate Spline (TPS)**
  - TPS simulates 2D deformations, using control points to map the original and target positions of the clothing. This method is widely used for warping clothing in virtual try-on systems.
  - TPS uses a set of control points (such as grid nodes) and interpolates the transformation between these points to generate smooth deformations. However, a key challenge with TPS is the lack of ground truth for the target positions in virtual try-on tasks, making it difficult to estimate precise transformations.

```python
from scipy.interpolate import Rbf

# Control points: chosen manually or derived from a method like shape context matching
source_points = np.array([[100, 100], [200, 100], [150, 200], [120, 300]])  # Source points (on the clothing)
target_points = np.array([[110, 120], [210, 130], [160, 210], [130, 320]])  # Target points (on the person)

# Thin Plate Spline (TPS) warping using Radial Basis Functions (RBF)
def tps_warp(source_points, target_points, img):
    grid_x, grid_y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    # Apply Radial Basis Function interpolation for x and y
    rbf_x = Rbf(source_points[:, 0], source_points[:, 1], target_points[:, 0], function='thin_plate')
    rbf_y = Rbf(source_points[:, 0], source_points[:, 1], target_points[:, 1], function='thin_plate')

    # Warp grid based on RBF interpolations
    warped_x = rbf_x(grid_x, grid_y)
    warped_y = rbf_y(grid_x, grid_y)

    # Map the original image to the new warped grid
    warped_image = cv2.remap(img, warped_x.astype(np.float32), warped_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    
    return warped_image
```

- **Spatial Transformation Network (STN)**
  - STNs are often used in conjunction with TPS to better control the warping process. STNs help constrain the transformation and prevent excessive or unrealistic deformations. STNs can work in an unsupervised or weakly supervised manner, even without direct ground truth (GT) for every transformation.

```python
# Spatial Transformer Network (STN) 
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        // ...existing code...
        
    def forward(self, x):
        // ...existing code...
        return x

# Create the STN model
stn = STN()

# Apply the STN to the clothing image
warped_cloth = stn(cloth_img)
```

- **Flow Estimation**
  - Flow indicates the offset of pixel or feature before and after transformation. Flow estimation methods for cloth warping can be classified in terms of prediction target such as pixel and feature or prediction steps such as single layer or multiple layers.

<p align="center">
  <img src="imgs/cloth_warp.png" alt="Person Agnostic" width="520" height="370">
</p>

- **Implicit Transformation**
  - Implicit Transformation methods achieve clothing alignment to the target body posture without explicit spatial transformations. Instead, they operate in feature space.

### Try-On Module Methods
- **Mask Combination**
  - This generates the final try-on image by blending three elements: the coarse try-on image (`I_coarse`), the warped clothing image (`C`), and a mask (`M`) representing the clothing region on the dressed person.

- **Generation-Based**
  - Generation-based methods in virtual try-on leverage generators to create high-quality try-on images, often utilizing U-Net and diffusion-based networks.

### Loss Functions
- **Local Constraints**
- **Semantic Constraints**
- **Diffusion-Specific Losses**

### Datasets and Evaluation for Virtual Try-On
- **Popular Datasets**: VITON-HD, Deep Fashion, MPV, Dress Code, SHHQ, UPT / ESF
- **In-the-Wild Datasets**: StreetTryOn, LH-400K, WPose, WVTON

### Evaluation Metrics
- **Inception Score (IS)**
- **Frechet Inception Distance (FID)**
- **LPIPS (Learned Perceptual Image Patch Similarity)**
- **Semantic Score (CLIP-based)**
- **Human Evaluation**

### Improvement Focus Areas
1. Performance in the Wild
2. Pipeline Efficiency
3. Handling Complex Poses & Occlusions
4. Multimodal Capabilities

## Paper Summaries

### Notable Papers

- **"StableViton"** by Authors
  - **Summary**: This work aims to expand the applicability of the pre-trained diffusion model to provide a standalone model for the virtual try-on task.
  - **Link**: [Read the paper](https://arxiv.org/pdf/2312.01725)

## References
- [Paper](https://arxiv.org/pdf/2311.04811)
- [StableViton](https://arxiv.org/pdf/2312.01725)

