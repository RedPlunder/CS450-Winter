import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")

# Set environment variable to prevent runtime issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# File paths that can be changed
RAG_CSV_PATH = "/Users/zhengnan/Documents/CS450/kd/kd.csv"
EMBEDDINGS_PATH = "doc_embeddings.npy"

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
