from pathlib import Path

from ingest import ingest_documents
from embed import embed_documents, save_embeddings
from retriever import load_embeddings, retrieve

if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_NAME = "all-MiniLM-L6-v2"
    VECTOR_STORE_PATH = PROJECT_ROOT / "vectorstore" / "embeddings.pkl"

    chunks = ingest_documents(DATA_DIR)
    print(f"Loaded and split {len(chunks)} chunks.")

    embeddings, texts = embed_documents(chunks, MODEL_NAME)

    save_embeddings(embeddings, texts, VECTOR_STORE_PATH)

    
