import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
import torch
from config import MODEL_NAME, EMBEDDINGS_PATH
from database import load_rag_data

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed(documents):
    """Generate embeddings for a list of documents."""
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def get_embeddings():
    """Load or generate document embeddings."""
    rag_data = load_rag_data()
    documents = rag_data['Concept'].tolist()

    if os.path.exists(EMBEDDINGS_PATH):
        print("Loading embeddings from file...")
        return np.load(EMBEDDINGS_PATH), documents
    else:
        print("Generating embeddings...")
        doc_embeddings = embed(documents)
        np.save(EMBEDDINGS_PATH, doc_embeddings)
        return doc_embeddings, documents

if __name__ == "__main__":
    embeddings, docs = get_embeddings()
    print(f"Generated {len(docs)} embeddings.")
