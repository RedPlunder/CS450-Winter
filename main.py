from retrieval import retrieve_documents
from openai_api import generate_response

def main():
    """Main function to run the RAG system."""
    query = input("Enter your query: ")

    # Retrieve top documents
    retrieved_docs = retrieve_documents(query, k=3)
    context = " ".join(retrieved_docs)

    # Generate response
    response = generate_response(query, context)
    
    print("\nGPT Answer:\n", response)

if __name__ == "__main__":
    main()
