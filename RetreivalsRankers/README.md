# Retrieval

Understanding the evolution from keyword-based to semantic search systems.

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
