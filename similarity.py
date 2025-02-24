import faiss
import numpy as np
from tqdm import tqdm

class SimilarityEngine:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        
    def build_index(self, vectors):
        self.index.add(vectors)
        
    def batch_search(self, query_vectors, k=5, batch_size=512, show_progress=True):
        n = len(query_vectors)
        distances = []
        
        # 定义迭代器，根据show_progress决定是否使用tqdm
        if show_progress:
            try:
                iterator = tqdm(
                    range(0, n, batch_size),
                    desc="Processing batches",
                    unit="batch"
                )
            except ImportError:
                raise ImportError("tqdm is required to show progress. Install it with pip install tqdm.")
        else:
            iterator = range(0, n, batch_size)
        
        # 分批次处理查询向量
        for start_idx in iterator:
            end_idx = start_idx + batch_size
            batch = query_vectors[start_idx:end_idx]
            batch_distances, _ = self.index.search(batch, k)
            distances.append(batch_distances)
        
        # 合并结果并归一化
        distances = np.vstack(distances)
        return self._normalize(distances)
    
    def save_index(self, path):
        faiss.write_index(self.index, path)
    
    @staticmethod
    def load_index(path):
        return faiss.read_index(path)
    
    def _normalize(self, distances):
        min_val, max_val = np.min(distances), np.max(distances)
        return 1 - (distances - min_val) / (max_val - min_val) if max_val != min_val else distances