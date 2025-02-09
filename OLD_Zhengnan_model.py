import pandas as pd
import openai
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import os

# Set environment variable to prevent runtime issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set your OpenAI API key
# openai.api_key = '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'

# Load tokenizer and model for embedding
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

# Load RAG database
rag_db = pd.read_csv("/Users/zhengnan/Documents/CS450/kd/kd.csv")
rag_data = rag_db[['ID', 'Concept']].dropna()

# Convert RAG content to lowercase for better matching
rag_data['Concept'] = rag_data['Concept'].str.lower()

# Embed documents from RAG for retrieval
documents = rag_data['Concept'].tolist()

def embed(documents):
    inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

if os.path.exists('doc_embeddings.npy'):
    print("Loading embeddings from file...")
    doc_embeddings = np.load('doc_embeddings.npy')
else:
    print("Generating embeddings...")
    doc_embeddings = embed(documents)
    np.save('doc_embeddings.npy', doc_embeddings)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Retrieve and validate documents
def retrieve_documents(query, k=3):
    query_embedding = embed([query.strip().lower()])[0]
    distances, indices = index.search(np.array([query_embedding]), k)
    return [documents[i] for i in indices[0]]

# Function to generate a response using OpenAI API
def generate_response(query, context):
    print("\n\nContext passed: ")
    print(context)
    print("\n\n")

    messages = [
        {"role": "system", "content": (
            "You are recognized as a Kubernetes and NGINX ingress expert. Before providing an answer, validate the provided context for "
            "errors, deprecated features, or potential conflicts. Always adhere to the latest Kubernetes and NGINX standards. "
            "Identify and clearly explain any assumptions made based on the context, and provide necessary corrections or enhancements."
        )},
        {"role": "user", "content": (
            f"Given the following detailed context and choose what you think fit information for question:\n{context}\nCan you provide a validated and comprehensive response to this query:\n{query}\n"
            "Your response should:\n"
            "1. Include YAML configurations with accurate and effective annotations tailored to address the query.\n"
            "2. Explain the rationale behind each configuration and validate them against the provided context and current best practices.\n"
            "3. Highlight and discuss any potential issues or critical assumptions that could affect the implementation.\n"
            "4. Offer detailed debugging steps and troubleshooting advice to verify and refine the solution."
        )}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=4090,
            temperature=0
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error: {e}")
        return "Have Error."

# User query
query = input("Question: ")
retrieved_docs = retrieve_documents(query, k=3)
context = " ".join(retrieved_docs)

# GPT Response
response = generate_response(query, context)
print("GPT Answer:", response)