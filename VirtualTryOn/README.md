# 2D Virtual Try On

Given a clothing image and a person image, an image based virtual try-on aims to generate a customized image
that appears natural and accurately reflects the characteristics of the clothing image. Generally virtual tryons consist of two modules

1.  A warping network to learn the semantic correspondence between the clothing and the human body.
2.  A generator
that fuses the warped clothing and the person image.

The nature
of matching clothing and individuals in the virtual try-on
dataset  makes it challenging to collect data in diverse environments, which in turn leads to limitations
in the generator’s generative capability

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

