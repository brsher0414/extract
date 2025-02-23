import faiss
import numpy as np

class SimilarityEngine:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        
    def build_index(self, vectors):
        self.index.add(vectors)
        
    def batch_search(self, query_vectors, k=5):
        distances, _ = self.index.search(query_vectors, k)
        return self._normalize(distances)
    
    def save_index(self, path):
        faiss.write_index(self.index, path)
    
    @staticmethod
    def load_index(path):
        return faiss.read_index(path)
    
    def _normalize(self, distances):
        min_val, max_val = np.min(distances), np.max(distances)
        return 1 - (distances - min_val) / (max_val - min_val) if max_val != min_val else distances