# MultiModal Learning 

This section contains multimodal learning not just multimodal llm but general mltimodal learning

## Contrastive Learning

### Clip
Pseudocode for CLIP training procedure using contrastive learning

```python

    # Encode images and texts to get their embeddings
    image_embeddings = image_encoder(images)
    text_embeddings = text_encoder(texts)
    
    # Normalize embeddings
    image_embeddings = normalize(image_embeddings)  # L2-normalize
    text_embeddings = normalize(text_embeddings)    # L2-normalize
    
    # Compute similarity matrix between all image and text embeddings
    logit_scale = learnable_parameter()  # Temperature parameter
    logits_per_image = logit_scale * np.dot(image_embeddings, text_embeddings.T)
    logits_per_text = logit_scale * np.dot(text_embeddings, image_embeddings.T)
    
    # Create ground truth labels - diagonal matrix since correct pairing is (i,i)
    labels = np.arange(N)
    
    # Compute cross-entropy loss for both directions
    loss_i2t = cross_entropy_loss(logits_per_image, labels)
    loss_t2i = cross_entropy_loss(logits_per_text, labels)
    
    # Total loss is symmetric sum of both directions
    total_loss = (loss_i2t + loss_t2i) / 2.0
    
    # Backpropagate and update parameters
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```


### SigLip

SigLip in general is much better than clip can use much smaller batch size for contrastive learning, we consider each image text pair for loss calculation and dont have to create N**2 noramization pair, it is also much more efficeent for paralllelism. We add a bias of -10 to make training more stable since most of pairs are negative so we dont want to give too high gradient signal initally during training solely based too many negative pairs.

``` python

    # Embeddings
    image_embeds = normalize(image_encoder(images))  # shape [B, D]
    text_embeds = normalize(text_encoder(texts))     # shape [B, D]
    
    # Compute pairwise similarities
    logits = image_embeds @ text_embeds.T  # shape [B, B]
    logits *= temperature.exp()  # Learnable temperature
    
    # Add learnable bias to counteract initial negative dominance
    logits += bias  # Learnable bias initialized to -10
    
    # Labels: 1 for diagonal (positive pairs), -1 otherwise
    labels = 2 * torch.eye(B) - 1  # [B, B]
    
    # Sigmoid cross-entropy loss
    loss = -torch.log(torch.sigmoid(labels * logits)).mean()
    return loss
```

### At inference time for zero-shot classification:

``` python
def zero_shot_classify(image, class_names):
    # Get image embedding
    image_embedding = normalize(image_encoder(image))
    
    # Embed class names as text
    text_descriptions = [f"A photo of a {name}" for name in class_names]
    text_embeddings = normalize(text_encoder(text_descriptions))
    
    # Compute similarity scores
    logits = logit_scale * np.dot(image_embedding, text_embeddings.T)
    
    # Convert to probabilities
    probs = softmax(logits)
    
    return probs

```