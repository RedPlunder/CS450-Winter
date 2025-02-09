import faiss
import numpy as np
from embedding.py import get_embeddings, embed

def create_index():
    """Create FAISS index from embeddings."""
    doc_embeddings, documents = get_embeddings()
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return index, documents

def retrieve_documents(query, k=3):
    """Retrieve top-k most relevant documents for a query."""
    index, documents = create_index()
    query_embedding = embed([query.strip().lower()])[0]
    distances, indices = index.search(np.array([query_embedding]), k)
    return [documents[i] for i in indices[0]]

if __name__ == "__main__":
    query = "What is Kubernetes?"
    retrieved_docs = retrieve_documents(query)
    print("Retrieved Documents:", retrieved_docs)
