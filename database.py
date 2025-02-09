import pandas as pd
from config import RAG_CSV_PATH

def load_rag_data():
    """Load and preprocess the RAG database."""
    rag_db = pd.read_csv(RAG_CSV_PATH)
    rag_data = rag_db[['ID', 'Concept']].dropna() #note that this was changed in the new version
    rag_data['Concept'] = rag_data['Concept'].str.lower()
    return rag_data

if __name__ == "__main__":
    rag_data = load_rag_data()
    print("RAG Data Loaded. Sample:")
    print(rag_data.head())
