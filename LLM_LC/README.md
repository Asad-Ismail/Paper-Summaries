## Long context LLMs

## Transformer XL

### Summary
The main idea of Transformer-XL is to enhance the standard Transformer architecture for language modeling by enabling it to capture longer-term dependencies and resolve the context fragmentation problem associated with fixed-length contexts. Unlike vanilla Transformers that process each segment independently, Transformer-XL introduces a recurrence mechanism between segments.
The hidden states computed for the previous segment are cached and reused as an extended context when processing the next segment. This allows information to flow across segments, enabling the model to capture longer-term dependencies. The hidden state of the current segment is computed by incorporating the cached hidden state from the previous segment.
Transformer-XL's cached representation is a specific form of caching designed for segment-level recurrence, where the entire hidden state sequence of a previous segment is cached and used as an extended context for the next segment. It allows the model to overcome limitations related to fixed-length contexts. This contrasts with the KV cache in standard Transformers, which primarily serves to optimize inference speed within a single segment or sequence. Transformer-XL's cache can also store multiple previous segments as memory.


1. Segment Processing:

The input sequence is divided into fixed-length segments, as discussed in our previous conversation.
A segment is represented as sτ = [xτ,1, ..., xτ,L], where L is the segment length.
Consecutive segments are represented as sτ and sτ+1.


2. Segment-Level Recurrence:

The hidden state sequence from the previous segment is cached and reused. 
For the n-th layer, the hidden state of the previous segment hn-1τ is cached. 
The hidden state for the current segment sτ+1 at layer n, denoted as hnτ+1, is computed by concatenating the cached hidden state and the current segment's hidden state: h̃n−1τ+1 = [ SG(hn−1τ ) ◦ hn−1τ+1 ], where SG means stop-gradient.


def segment_level_recurrence(previous_hidden_state, current_segment_input):
    # previous_hidden_state: Hidden state from the previous segment (hn-1_tau)
    # current_segment_input: Input embeddings for the current segment (hn-1_tau+1)
    # Concatenate previous_hidden_state and current_segment_input
    extended_context = concatenate(stop_gradient(previous_hidden_state), current_segment_input)
    #Compute key and value vectors based on extended_context
    key = compute_key_vector(extended_context)
    value = compute_value_vector(extended_context) 

    return key, value, extended_context # Return key, value, and extended context



3. Relative Positional Encodings:
Instead of absolute positional encodings, Transformer-XL uses relative positional encodings.
The attention score Arel between query vector qi and key vector kj is computed as:

Arel i,j = E>xiW>q Wk,EExj + E>xiW>q Wk,RRi-j + u>Wk,EExj + v>Wk,RRi-j.

where the first term is content-based, the second is a content-dependent positional bias, the third is a global content bias and the fourth is a global positional bias.
R is a sinusoid encoding matrix, and u and v are trainable parameters.

Pseudo-code 
def relative_positional_encoding(query, key, content_key, position_key, u, v, Ri_minus_j):
    # query: Query vector (E_xi * Wq)
    # key: Content-based key vector (Wk,E * Exj)
    # content_key: Content-based key vector (Wk,E * Exj)
    # position_key: location-based key vector (Wk,R * Ri-j)
    # u: trainable parameter for global content bias
    # v: trainable parameter for global position bias
    # Ri_minus_j: relative position encoding
    term_a = torch.matmul(query, key.transpose(-2, -1)) # content-based addressing
    term_b = torch.matmul(query, position_key.transpose(-2, -1)) #content-dependent positional bias
    term_c = torch.matmul(u, content_key.transpose(-2, -1)) # global content bias
    term_d = torch.matmul(v, position_key.transpose(-2, -1)) # global positional bias

    attention_score = term_a + term_b + term_c + term_d

    return attention_score


4. Transformer Layer:

The core of Transformer-XL uses the self-attention mechanism and feed-forward networks, similar to standard Transformers.
The computation involves calculating attention scores, applying a mask to the attention scores, normalizing the output, and passing it through a feed-forward network.

#Pseudo-code 

def transformer_layer(hidden_state, memory, Wq, Wk_E, Wv, Wk_R, u, v, R, mask):
    # hidden_state: Hidden state sequence from the previous layer (h_n-1_tau)
    # memory: Cached hidden states from previous segments (m_n-1_tau)
    # Wq, Wk_E, Wv, Wk_R: Weight matrices
    # u,v: trainable parameters
    # R: relative position encoding
    # mask: mask to hide future tokens

    # Concatenate memory and hidden state to create input for the layer
    h_tilde = concatenate(stop_gradient(memory), hidden_state)
    # Compute query, key, and value vectors
    query = torch.matmul(h_tilde, Wq)
    key_E = torch.matmul(h_tilde, Wk_E)
    value = torch.matmul(h_tilde, Wv)
    key_R = torch.matmul(h_tilde, Wk_R)

    # Compute attention scores with relative positional encoding
    attention_scores = relative_positional_encoding(query, key_E, key_E, key_R, u, v, R)

    # Apply mask to attention scores
    masked_attention_scores = mask_attention(attention_scores, mask)

    # Compute attention weights
    attention_weights = masked_softmax(masked_attention_scores)

    # Apply attention to values
    attended_values = torch.matmul(attention_weights, value)

    # Perform layer normalization
    normalized_output = layer_norm(linear(attended_values) + hidden_state)

    # Position-wise feed-forward network
    output = positionwise_feed_forward(normalized_output)

    return output


5. Overall Architecture:

The input is processed through multiple Transformer layers.
The output of each layer is used as input to the next layer.
The final output is a hidden representation used for predicting the next token.


During evaluation, cached hidden states from previous segments are reused to speed up the process.

def transformer_xl_model(input_sequence, segment_length, num_layers, Wq, Wk_E, Wv, Wk_R, u, v, R):
    # input_sequence: Input sequence of tokens
    # segment_length: Length of each segment
    # num_layers: Number of Transformer layers
    # Wq, Wk_E, Wv, Wk_R: Weight matrices
    # u,v: trainable parameters
    # R: relative position encoding
    segments = create_segments(input_sequence, segment_length)
    memory = torch.zeros() # Initialize memory
    output = torch.tensor([]) # Initialize output

    for segment in segments:
      hidden_state = embedding_lookup(segment) # Get word embeddings
      for layer in range(num_layers):
          key, value, extended_context = segment_level_recurrence(memory, hidden_state)
          hidden_state = transformer_layer(hidden_state, memory, Wq, Wk_E, Wv, Wk_R, u, v, R, mask)
          memory = hidden_state #Cache hidden states
      output = torch.cat((output, hidden_state)) #Concatenate segment output

    #Output logits
    logits = linear(output)
    return logits



## Leave No context behind [Paper](https://arxiv.org/pdf/2404.07143v1)


"Leave No Context Behind" paper introduces Infini-attention, a novel attention mechanism that enables Transformer-based Large Language Models (LLMs) to process infinitely long inputs with bounded memory and computation. Infini-attention incorporates a compressive memory into the standard attention mechanism, combining both masked local attention and long-term linear attention within a single Transformer block.
Here's a breakdown of the key concepts:

Segment-Level Processing: Like Transformer-XL, Infini-Transformer operates on a sequence of input segments.

Local Attention: Within each segment, Infini-attention computes the standard causal dot-product attention. This is similar to the attention mechanism in standard transformers.

Compressive Memory: Instead of discarding old key-value (KV) states like in standard attention, Infini-attention reuses them to maintain the entire context history with a compressive memory.

Memory Update and Retrieval: The model reuses the query (Q), key (K), and value (V) states from the dot-product attention computation. The old KV states are stored in the compressive memory. When processing subsequent sequences, values are retrieved from the memory using the attention query states.

Linear Attention: The memory update and retrieval process is cast as a linear attention mechanism for simplicity and computational efficiency. This involves using an associative matrix to parameterize the memory.

Memory Update Rule: The memory is updated with new KV entries. The update rule can be either Linear or Linear+Delta. Linear update adds new key-value bindings to the memory, while Linear+Delta first retrieves existing value entries and subtracts them from the new values before applying the associative bindings.

Context Aggregation: Infini-attention aggregates the long-term memory-retrieved values and the local attention contexts to compute the final contextual output.

Gating Mechanism: A learned gating scalar determines the trade-off between long-term memory and local information flows.

Bounded Memory: The model has a constant memory complexity, unlike models where memory complexity grows along with the sequence dimension.


Pseudo-code :
function InfiniAttention(X, Ms_prev, zs_prev):
    // X: Input segment, Ms_prev: Previous memory, zs_prev: Previous normalization term
    // 1. Compute Q, K, and V (Query, Key, Value) for the current segment
    K = X * W_K
    V = X * W_V
    Q = X * W_Q

    // 2. Local Dot-Product Attention
    A_dot = softmax(Q * K^T / sqrt(d_model)) * V

    // 3. Compressive Memory Retrieval
    A_mem =  σ(Q) * Ms_prev / zs_prev    // σ is ELU+1

    // 4. Memory Update using associative binding operator
    Ms = Ms_prev + σ(K)^T * V  // Linear update
    //or
    Ms = Ms_prev + σ(K)^T * (V - A_mem ) // Linear + Delta update

    zs = zs_prev + sum(σ(K))

    // 5. Context Aggregation with a gating scalar
    β = learnable_parameter // for each attention head
    A = sigmoid(β) * A_mem + (1 - sigmoid(β)) * A_dot

    // 6. Multi-Head Attention and output projection
    O =  Concatenate(A_1, ..., A_H) * W_O
    return O, Ms, zs
end function

// Initialize memory M_0 and normalization z_0
M = M_0
z = z_0

// Process sequence of segments
for each segment X_s in input_sequence:
    O_s, M, z = InfiniAttention(X_s, M, z)
    // O_s is the output for the current segment


### Other Intersting Paper Ideas to follow on long context handelling

[Nvidia Ruler](https://github.com/NVIDIA/RULER)
[LongRope](https://arxiv.org/pdf/2402.13753)
[LCVsRAG](https://arxiv.org/pdf/2501.01880)
[CAG](https://github.com/hhhuang/CAG/tree/main)
[MichelAngelo](https://arxiv.org/html/2409.12640v1)

**ROPE VS LONGROPE:**


def rope_encoding(position, dimension, base=10000):
  """
  Calculates the RoPE encoding for a given position.
  """
  encoding = []
  for i in range(dimension // 2):
    theta = base ** (-2 * i / dimension)
    encoding.append(cos(position * theta))
    encoding.append(sin(position * theta))
  return encoding

position = 5
embedding_dimension = 128
rope_values = rope_encoding(position, embedding_dimension)



LongRoPE (Conceptual)
def longrope_encoding(position, dimension, rescale_factors, start_token_threshold, base=10000):
  """
  Calculates the LongRoPE encoding for a given position.
  """
  encoding = []
  for i in range(dimension // 2):
        theta = base ** (-2 * i / dimension)
        if position < start_token_threshold:
            rescale_factor = 1  # No rescaling for initial tokens
        else:
            rescale_factor = 1/rescale_factors[i]

        encoding.append(cos(position * theta * rescale_factor ))
        encoding.append(sin(position * theta * rescale_factor ))
  return encoding

position = 5
embedding_dimension = 128
# rescale_factors is a list of rescale factors obtained through search, one for each dimension
rescale_factors = [1.0, 1.2, 0.9, ...] # this values are the lambda
start_token_threshold = 4 # this value is n_hat
longrope_values = longrope_encoding(position, embedding_dimension, rescale_factors, start_token_threshold )