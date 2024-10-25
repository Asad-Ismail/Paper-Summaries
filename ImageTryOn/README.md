# 2D Virtual Try On

Image-based virtual try-on can be regarded as a
conditional image generation task that uses inshop clothing image Ic and person image Ip as
raw data, and pre-processes the raw data as conditioned information to guide the model for generating try-on images Itry−on = G(Ip, Ic). Three
key modules are usually involved in image-based
virtual try-on:

## StableViton

[Paper](https://arxiv.org/pdf/2312.01725)

In this work, we aim to expand the applicability of the
pre-trained diffusion model to provide a standalone model
for the virtual try-on task. In the effort to adapt the pretrained diffusion model for virtual try-on, a significant challenge is to preserve the clothing details while harnessing
the knowledge of the pre-trained diffusion model. This
can be achieved by learning the semantic correspondence
between clothing and the human body using the provided
dataset. Recent research that has employed pretrained diffusion models in virtual try-on has shown limitations due to the following two issues:
1. Insufficient spatial information available for learning the semantic correspondence
2. The pre-trained diffusion model not
being fully utilized, as it pastes the warped clothing in the
RGB space, relying on external warping networks as previous approaches for aligning the input condition.

