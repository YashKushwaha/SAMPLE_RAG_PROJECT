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


if torch.cuda.is_available():
    device = 'cuda'
    print('GPU found')
else:
    device = 'cpu'

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

model = model.to(device)

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences, device=device)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])