from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"  # Example HF embedding model
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = PROJECT_ROOT / "vectorstore" / "embeddings.pkl"



def embed_documents(docs, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings, texts


def save_embeddings(embeddings, texts, save_path=VECTOR_STORE_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump((embeddings, texts), f)
