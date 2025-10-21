from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import torch

class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.model.encode(X)
    
