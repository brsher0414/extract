import os
import numpy as np
import pandas as pd
from typing import List, Tuple

class ReportProcessor:
    def __init__(self, 
                 group_columns: List[str] = ['PLATFORM', 'CATCODE'],
                 similarity_threshold: float = 0.6,
                 top_n: int = 20):
        """
        参数说明：
        - group_columns: 分组列名
        - similarity_threshold: 相似度阈值
        - top_n: 每组保留前N条记录
        """
        self.group_columns = group_columns
        self.threshold = similarity_threshold
        self.top_n = top_n
        self.mask_path = "similarity_mask.npy"
        
    def _generate_mask(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """生成全局筛选掩码（内存映射优化）"""
        mask = np.all(similarity_matrix <= self.threshold, axis=1)
        np.save(self.mask_path, mask)
        return np.load(self.mask_path, mmap_mode='r')
    
    def _process_group(self, 
                      group: pd.DataFrame, 
                      mask: np.ndarray) -> pd.DataFrame:
        """处理单个分组（内存优化版）"""
        group_indices = group.index.values
        valid_mask = mask[group_indices]
        
        filtered = group.query("index in @group_indices[@valid_mask]")
        
        if filtered.empty:
            return pd.DataFrame(columns=group.columns)
            
        return filtered.nlargest(self.top_n, 'SIMILARITY', keep='first')
    
    def process(self, 
               data: pd.DataFrame, 
               similarity_matrix: np.ndarray) -> pd.DataFrame:
        """主处理流程"""
        # 清理临时文件
        if os.path.exists(self.mask_path):
            os.remove(self.mask_path)
            
        # 添加相似度列
        data = data.assign(SIMILARITY=similarity_matrix.mean(axis=1))
        
        # 生成全局掩码
        mask = self._generate_mask(similarity_matrix)
        
        # 分块处理（内存优化）
        chunk_size = 50000
        chunks = []
        
        for chunk in np.array_split(data, len(data) // chunk_size + 1):
            grouped = chunk.groupby(self.group_columns, observed=True)
            results = [
                self._process_group(g, mask) 
                for _, g in grouped
            ]
            chunks.append(pd.concat(results, copy=False))
            
            # 及时释放内存
            del grouped, results
            if (len(chunks) % 10) == 0:
                pd.DataFrame().empty  
        
        # 最终合并
        result = pd.concat(chunks, copy=False)
        
        # 最终排序
        return result.sort_values(
            self.group_columns + ['SIMILARITY'],
            ascending=[True]*len(self.group_columns) + [False],
            kind='mergesort'
        ).reset_index(drop=True)