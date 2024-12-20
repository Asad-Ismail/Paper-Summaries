## Why Small scale LLM models
#### Advantages of Small Models

1. **Accessibility**: Smaller models are easier to deploy and use.
2. **Cost-Effective**: They require less computational power, reducing deployment and maintenance costs.
3. **Faster Training**: Their smaller size allows for quicker training and fine-tuning.
4. **Domain-Specific Performance**: Fine-tuning on specific domains can yield high performance, often surpassing larger models in accuracy and relevance.
5. **Enhanced Privacy**: Deploying small models locally ensures data remains secure and private.
6. **Optimization (Price per Token)**: Optimizing the cost per token is crucial for efficient model deployment. Fine-tuning smaller models on specific domains can significantly reduce costs while maintaining high performance.
7. **Model Quality**: Fine-tuned models on specific domains often outperform larger, general-purpose language models (LLMs). Tailoring a model to a particular domain ensures higher accuracy and relevance in responses.

By leveraging small models and Parameter-Efficient Fine-Tuning (PEFT) methods, organizations can achieve high performance while maintaining cost-efficiency and data privacy.

## Fine-Tuning a Small Model

Fine-tuning a small model involves several steps to ensure it meets the desired performance and accuracy. Here are the key objectives:

1. **Inject New Domain Knowledge (Continuous Pretraining)**: Start with a good pretrained model and continue training it on new domain-specific data to inject relevant knowledge.
2. **Answer the Way I Want (Fine-Tuning)**: Fine-tune the model to ensure it responds in the desired manner.
3. **Provide Examples of Good and Bad Answers (Alignment)**: Align the model by providing examples of what constitutes a good answer and what does not.

### Example Process

If you need to train a model on millions of internal documents, which probably contain billions of tokens, consider the following:

- Training the entire model in FP16 or BF16 precision can be very compute-intensive.
- A more efficient approach is using Parameter-Efficient Fine-Tuning (PEFT) methods. PEFT works well for fine-tuning but is less effective for continuous training.
- Spectrum-based methods might be beneficial as they identify key layers and train them with full precision.

### PEFT Methods

- **LoRA (Low-Rank Adaptation)**
- **QLoRA (Quantized LoRA)**

### Alignment Techniques

- **RLHF (Reinforcement Learning from Human Feedback)**: Effective but hard to scale.
- **DPO (Direct Preference Optimization)**: More cost-effective.

### Merging Models

An alternative to training is merging models. This involves averaging the weights of useful models to create a new model. The Merge Kit library can facilitate this process. Merging can occur at any of the above three stages. For example, check out the Supernova 70B models for more insights.


