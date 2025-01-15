# Classification

Classification is a fundamental task in machine learning where the goal is to assign a label to an input based on its features. This directory contains summaries and reviews of papers related to classification algorithms and methods for handling samples that are very different from the training data.

## Table of Contents
- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [Paper Summaries](#paper-summaries)
- [References](#references)

## Introduction
Classification involves predicting the category or class of a given data point. It is widely used in various applications such as image recognition, spam detection, and medical diagnosis. One of the challenges in classification is dealing with out-of-distribution (OOD) samples, which are data points that differ significantly from the training data.

## Key Concepts
- **In-Distribution (ID) Samples**: Data points that are similar to the training data.
- **Out-of-Distribution (OOD) Samples**: Data points that differ significantly from the training data and may lead to incorrect predictions.
- **Robustness**: The ability of a classification model to handle OOD samples effectively.
- **Anomaly Detection**: Techniques used to identify OOD samples before making predictions.
- **Confidence Scoring**: Methods to assess the reliability of predictions and identify uncertain cases.
- **Adversarial Training**: Training the model with adversarial examples to improve its robustness.

## Paper Summaries

### Notable Papers


  - **"ImageNet Classification with Deep Convolutional Neural Networks"** by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
    - **Summary**: This seminal paper introduced the AlexNet architecture, which significantly improved image classification performance on the ImageNet dataset.
    - **Link**: [Read the paper](https://dl.acm.org/doi/10.1145/3065386)
  
  - **"Deep Residual Learning for Image Recognition"** by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    - **Summary**: This paper introduced the ResNet architecture, which uses residual connections to train very deep neural networks effectively.
    - **Link**: [Read the paper](https://arxiv.org/abs/1512.03385)
  
  - **"A Survey of Deep Learning for Image Classification"** by Yonghui Li, et al.
    - **Summary**: This survey paper provides a comprehensive overview of deep learning techniques for image classification.
    - **Link**: [Read the paper](https://arxiv.org/abs/1801.03505)


- **"Deep Anomaly Detection with Outlier Exposure"** by Dan Hendrycks, Mantas Mazeika, Thomas Dietterich
  - **Summary**: This paper introduces a method for improving the performance of anomaly detection models by exposing them to outliers during training.
  - **Key Points**:
    - Outlier exposure helps the model learn to distinguish between in-distribution and out-of-distribution samples.
    - The method improves the robustness of the model to OOD samples.
  - **Link**: [Read the paper](https://arxiv.org/abs/1812.04606)

- **"Confidence-Aware Learning for Deep Neural Networks"** by Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
  - **Summary**: This paper proposes a confidence-aware learning approach to improve the reliability of predictions made by deep neural networks.
  - **Key Points**:
    - The approach involves training the model to output calibrated confidence scores.
    - Confidence-aware learning helps in identifying uncertain predictions and handling OOD samples.
  - **Link**: [Read the paper](https://arxiv.org/abs/1706.04599)

- **"Adversarial Training for Free!"** by Shafahi et al.
  - **Summary**: This paper presents a method for adversarial training that does not require additional computational resources.
  - **Key Points**:
    - The method leverages the existing training process to incorporate adversarial examples.
    - Adversarial training improves the robustness of the model to adversarial attacks and OOD samples.
  - **Link**: [Read the paper](https://arxiv.org/abs/1904.12843)

## References
- [Anomaly Detection in Machine Learning](https://arxiv.org/abs/1904.09751)
- [Confidence Scoring for Robust Classification](https://arxiv.org/abs/1702.01806)
- [Adversarial Training for Robust Models](https://arxiv.org/abs/1903.06059)
- [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606)
- [Confidence-Aware Learning for Deep Neural Networks](https://arxiv.org/abs/1706.04599)
- [Adversarial Training for Free!](https://arxiv.org/abs/1904.12843)
- [Know what you dont know Blog!](https://www.imt.ch/en/expert-blog-detail/know-what-you-dont-know-en)
