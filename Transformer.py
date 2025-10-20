from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device='cpu')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.model.encode(X)
    