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
The training objective is to make this score reflect the relevance between the query and the document..
