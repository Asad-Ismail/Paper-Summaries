# Retrieval

Understanding the evolution from keyword-based to semantic search systems.

A great comparison can be found here from Qdrant [Article](https://qdrant.tech/articles/modern-sparse-neural-retrieval/)

## Lexical Retrieval (BM25-like)

```python
def keyword_search(query, documents):
    matching_docs = []
    query_terms = set(query.lower().split())
    
    for doc in documents:
        doc_terms = set(doc.lower().split())
        if query_terms.intersection(doc_terms):  
            matching_docs.append(doc)
    
    return matching_docs

query = "tasty cheese"
# Won't match documents containing "Gouda" or "Brie"
```

### Inverted Index Structure

Simple representation of how inverted indices work:
```python
class InvertedIndex:
    def __init__(self):
        self.index = {}
    
    def add_document(self, doc_id, content):
        terms = content.lower().split()
        for term in terms:
            if term not in self.index:
                self.index[term] = set()
            self.index[term].add(doc_id)
    
    def search(self, query):
        query_terms = query.lower().split()
        # Return documents that contain ANY query term
        matching_docs = set()
        for term in query_terms:
            if term in self.index:
                matching_docs.update(self.index[term])
        return matching_docs
```

### BM25 Score (Classic Keyword Search)

```
BM25(D,Q) = ∑(IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl)))

Where:
- D = Document
- Q = Query terms
- f(qi,D) = Term frequency in document
- |D| = Document length
- avgdl = Average document length
- k1, b = Free parameters
```

### Challenges with Keyword-Based Search

1. Vocabulary Mismatch:
```
Query: "tasty cheese"
Document: "Gouda is a premium Dutch cheese"
Result: No match (despite semantic relevance)
```

2. Semantic Mismatch:
```
Query: "car repair"
Document: "automotive maintenance services"
Result: No match (despite meaning the same thing)
```

### Limitations of Traditional Keyword Search

- Exact match requirement
- No understanding of context
- Misses semantic relationships
- Vocabulary dependence


### Sparse Neural Retrieval

The authors of one of the first sparse retrievers, the Deep Contextualized Term Weighting framework (DeepCT), predict an integer word’s impact value separately for each unique word in a document and a query. They use a linear regression model on top of the contextual representations produced by the basic BERT model, the model’s output is rounded.

```python
class DeepCTScorer:
    def __init__(self, bert_model, linear_layer):
        self.bert = bert_model
        self.linear = linear_layer
        
    def calculate_importance(self, text: str) -> Dict[str, float]:
        # 1. Process entire text at once
        tokens = self.bert.tokenize(text)
        # Shape: (sequence_length, 768)
        contextual_embeddings = self.bert(tokens)
        
        # 2. Track original words to token mapping
        word_to_tokens = self.get_word_to_token_mapping(text, tokens)
        
        # 3. Calculate importance scores with context
        scores = {}
        words = text.split()
        
        for word in words:
            # Get token indices for this word
            token_indices = word_to_tokens[word]
            
            # Get contextual embeddings for word's tokens
            word_embeddings = contextual_embeddings[token_indices]
            
            # Pool if word was split into multiple tokens
            if len(token_indices) > 1:
                word_embedding = self.pool_wordpiece_embeddings(word_embeddings)
            else:
                word_embedding = word_embeddings[0]
            
            # Calculate importance score
            importance = self.linear(word_embedding)
            scores[word] = round(float(importance))
            
        return scores

    def get_word_to_token_mapping(self, text: str, tokens: List[str]) -> Dict[str, List[int]]:
        """Maps original words to their token indices"""
        mapping = {}
        current_word = ""
        current_indices = []
        
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                # Continue previous word
                current_indices.append(i)
            else:
                # New word starts
                if current_word:
                    mapping[current_word] = current_indices
                current_word = token
                current_indices = [i]
        #for last word
        if current_word:
            mapping[current_word] = current_indices
            
        return mapping

    def pool_wordpiece_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Combine multiple wordpiece embeddings into one"""
        # Common pooling strategies:
        # 1. Take first token (CLS-like)
        # return embeddings[0]
        
        # 2. Average pooling
        return np.mean(embeddings, axis=0)

```

Limitation of DeepCT is tha tto train the linear layer we need to have the ground truth of scores of words, which is hard to define what is the most, second most important word in a document of 5 pages.

### Deep Impact

It’s much easier to define whether a document as a whole is relevant or irrelevant to a query. That’s why the DeepImpact Sparse Neural Retriever authors directly used the relevancy between a query and a document as a training objective. DeepImpact Sparse Neural Retriever authors directly used the relevancy between a query and a document as a training objective. They take BERT’s contextualized embeddings of the document’s words, transform them through a simple 2-layer neural network in a single scalar score and sum these scores up for each word overlapping with a query.
The training objective is to make this score reflect the relevance between the query and the document.
The DeepImpact model (like the DeepCT model) takes the first piece BERT produces for a word and discards the rest. However, what can one find searching for “Q” instead of “Qdrant”?

To solve the problems of DeepImpact’s architecture, the Term Independent Likelihood MoDEl (TILDEv2) model generates sparse encodings on a level of BERT’s representations, not on words level. Aside from that, its authors use the identical architecture to the DeepImpact model.
A single scalar importance score value might not be enough to capture all distinct meanings of a word. Homonyms (pizza, cocktail, flower, and female name “Margherita”) are one of the troublemakers in information retrieval.

**COIL**:
 If one value for the term importance score is insufficient, we could describe the term’s importance in a vector form! Authors of the COntextualized Inverted List (COIL) model based their work on this idea. Instead of squeezing 768-dimensional BERT’s contextualised embeddings into one value, they down-project them (through the similar “relevance” training objective) to 32 dimensions. Moreover, not to miss a detail, they also encode the query terms as vectors.

**Universal COntextualized Inverted List (UniCOIL)**:

Made by the authors of COIL as a follow-up, goes back to producing a scalar value as the importance score rather than a vector, leaving unchanged all other COIL design decisions


### Vocabulary Mismatch:

The retrieval based on the exact matching, however sophisticated the methods to predict term importance are, we can’t match relevant documents which have no query terms in them. If you’re searching for “pizza” in a book of recipes, you won’t find “Margherita”.
A way to solve this problem is through the so-called document expansion. Let’s append words which could be in a potential query searching for this document. So, the “Margherita” document becomes “Margherita pizza”. Now, exact matching on “pizza” will work.

There are two types of document expansion that are used in sparse neural retrieval: external (one model is responsible for expansion, another one for retrieval) and internal (all is done by a single model).

### External Document Expansion:

1. External Tools:


    - External Document Expansion with docT5query:

        External Document Expansion with docT5querydocT5query is the most used document expansion model. It is based on the Text-to-Text Transfer Transformer (T5) model trained to generate top-k possible queries for which the given document would be an answer. These predicted short queries (up to ~50-60 words) can have repetitions in them, so it also contributes to the frequency of the terms if the term frequency is considered by the retriever.

        The problem with docT5query expansion is a very long inference time, as with any generative model: it can generate only one token per run, and it spends a fair share of resources on it.

    - Term Independent Likelihood MODel (TILDE):

        Term Independent Likelihood MODel (TILDE) is an external expansion method that reduces the passage expansion time compared to docT5query by 98%. It uses the assumption that words in texts are independent of each other (as if we were inserting in our speech words without paying attention to their order), which allows for the parallelisation of document expansion.

        Instead of predicting queries, TILDE predicts the most likely terms to see next after reading a passage’s text (query likelihood paradigm). TILDE takes the probability distribution of all tokens in a BERT vocabulary based on the document’s text and appends top-k of them to the document without repetitions.

    Problems of external document expansion: External document expansion might not be feasible in many production scenarios where there’s not enough time or compute to expand each and every document you want to store in a database and then additionally do all the calculations needed for retrievers. To solve this problem, a generation of models was developed which do everything in one go, expanding documents “internally”.

2. Internal Document Expansion

     **SPARTA**
    Sparse Transformer Matching (SPARTA) model use BERT’s model and BERT’s vocabulary (around 30,000 tokens). For each token in BERT vocabulary, they find the maximum dot product between it and contextualized tokens in a document and learn a threshold of a considerable (non-zero) effect. Then, at the inference time, the only thing to be done is to sum up all scores of query tokens in that document.
    Trained on the MS MARCO dataset, many sparse neural retrievers, including SPARTA, show good results on MS MARCO test data, but when it comes to generalisation (working with other data), they could perform worse than BM25.

    The authors of the Sparse Lexical and Expansion Model (SPLADE)] family of models added dense model training tricks to the internal document expansion idea, which made the retrieval quality noticeably better.

    **SPLADE**
    The SPARTA model is not sparse enough by construction, so authors of the SPLADE family of models introduced explicit sparsity regularisation, preventing the model from producing too many non-zero values.
    The SPARTA model mostly uses the BERT model as-is, without any additional neural network to capture the specifity of Information Retrieval problem, so SPLADE models introduce a trainable neural network on top of BERT with a specific architecture choice to make it perfectly fit the task.
    SPLADE family of models, finally, uses knowledge distillation, which is learning from a bigger (and therefore much slower, not-so-fit for production tasks) model how to predict good representations.
    One of the last versions of the SPLADE family of models is SPLADE++.
    SPLADE++, opposed to SPARTA model, expands not only documents but also queries at inference time. We’ll demonstrate this in the next section.


### Dense Vs Sparse Retrievals

If n is number of documents:

 - Sparse Retrieval: 
    1. Memory is O(k * n) because:
    - Only store non-zero values
    - Each document has ~k unique terms
    - Don't store the zeros

    2. Search is O(log n) because:
    - Use inverted index for O(1) term lookup
    - Posting lists are sorted
    - Efficient merge operations with heap
    - Skip pointers for faster traversal

 - Dense Retrieval:
    - Memory: O(n * v) - full vector for each document
    - Search: O(n) - must compare with every document


    Example:
        dense_memory = {
            "structure": "Each document → Dense vector",
            "size": "n * v",  # Each doc has full vector
            "example": """
            Doc1: [0.1, 0.2, 0.3, ..., 0.8]  # v dimensions
            Doc2: [0.2, 0.5, 0.1, ..., 0.3]  # v dimensions
            ...
            DocN: [0.4, 0.1, 0.7, ..., 0.5]  # v dimensions
            """
        }
        
        # Sparse Retrieval Memory (Inverted Index)
        sparse_memory = {
            "structure": "Each term → List of (doc_id, score)",
            "size": "k * n",  # k terms per doc on average
            "example": """
            'apple': [(doc1, 0.8), (doc4, 0.3), ...]
            'banana': [(doc2, 0.5), (doc7, 0.9), ...]
            ...
            """
        }
    
### Conclusion

In areas where keyword matching is crucial but BM25 is insufficient for initial retrieval, semantic matching (e.g., synonyms, homonyms) adds significant value. This is especially true in fields such as medicine, academia, law, and e-commerce, where brand names and serial numbers play a critical role. Dense retrievers tend to return many false positives, while sparse neural retrieval helps narrow down these false positives.

Sparse neural retrieval can be a valuable option for scaling, especially when working with large datasets. It leverages exact matching using an inverted index, which can be fast depending on the nature of your data.

If you’re using traditional retrieval systems, sparse neural retrieval is compatible with them and helps bridge the semantic gap.