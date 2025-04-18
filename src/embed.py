from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
from pathlib import Path
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"  # Example HF embedding model
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_NAME = "all-MiniLM-L6-v2"
#VECTOR_STORE_PATH = PROJECT_ROOT / "vectorstore" / "embeddings.pkl"

VECTOR_STORE_PATH = "vectorstore/faiss_index.index"
TEXT_STORE_PATH = "vectorstore/faiss_texts.pkl"

def embed_documents(docs, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings, texts

def save_embeddings(embeddings, texts, save_path=VECTOR_STORE_PATH, text_path=TEXT_STORE_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Ensure embeddings are in float32 format for FAISS
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, save_path)

    # Save corresponding texts
    with open(text_path, "wb") as f:
        pickle.dump(texts, f)
