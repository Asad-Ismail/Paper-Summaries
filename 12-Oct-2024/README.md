
## VITRON: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing

[Code](https://vitron-llm.github.io/) | [Paper](https://haofei.vip/downloads/papers/Skywork_Vitron_2024.pdf)

### Summary:

Vitron tries to address two common gaps with MLLMs (multimodal LLMs):
1. MLLMs are generalist e.g., they have coarse-grain instance-level understanding.
2. Lack of unified support for both images and videos and their coverage for visual understanding and generation.

### Architecture:

VITRON employs a common 'encoder-LLM-decoder' architecture, similar to other popular MLLMs. The framework consists of three main components:

1. **Frontend Vision & Language Encoders**: These encoders process the input images and text.
2. **Central LLM**: This component is responsible for semantic understanding and text generation.
3. **Backend Decoder Modules**: These modules handle user responses and vision manipulation.

<p align="center">
    <img src="imgs/vitron-arch.png" alt="VITRON Architecture" width="600" height="350">
</p>





















