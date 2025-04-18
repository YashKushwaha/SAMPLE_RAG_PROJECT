import torch
from sentence_transformers import SentenceTransformer

class EmbeddingsModel:
    def __init__(self, model_name=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name  = "all-MiniLM-L6-v2" or model_name
        self.model = SentenceTransformer(model_name).to(device)

    def encode(self, sentence_list):
        embeddings = self.model.encode(sentence_list, device=device)
        return embeddings
    
    def get_similarities(self, embeddings):
        return self.model.similarity(embeddings, embeddings)