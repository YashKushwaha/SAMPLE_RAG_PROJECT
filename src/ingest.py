### ingest.py
import os
from pathlib import Path
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embed import embed_documents, save_embeddings,VECTOR_STORE_PATH, MODEL_NAME
from retriever import load_embeddings, retrieve

def ingest_documents(data_path: str, chunk_size=500, chunk_overlap=100):
    loader = DirectoryLoader(data_path, glob='**/*.txt')
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"

    chunks = ingest_documents(DATA_DIR)
    print(f"Loaded and split {len(chunks)} chunks.")

    embeddings, texts = embed_documents(chunks, MODEL_NAME)

    save_embeddings(embeddings, texts, VECTOR_STORE_PATH)
