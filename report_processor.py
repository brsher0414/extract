import os
import numpy as np
import pandas as pd
from typing import List

class ReportProcessor:
    def __init__(self,
                 group_columns: List[str] = ['PLATFORM', 'CATCODE'],
                 similarity_threshold: float = 0.7,
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

    def process(self,
               data: pd.DataFrame,
               similarity_matrix: np.ndarray) -> pd.DataFrame:
        """主处理流程（优化后版本）"""
        # 清理临时文件
        if os.path.exists(self.mask_path):
            os.remove(self.mask_path)

        # 生成全局掩码
        mask = self._generate_mask(similarity_matrix)

        # 添加相似度列并应用掩码
        filtered_data = data.assign(SIMILARITY=similarity_matrix.mean(axis=1)) \
                            .loc[mask] \
                            .astype({
                                'SIMILARITY': 'float32',
                                'PLATFORM': 'category',
                                'CATCODE': 'category'
                            })

        # 全局分组筛选TopN
        result = filtered_data.groupby(
            self.group_columns,
            observed=True,
            group_keys=False
        ).apply(
            lambda x: x.nlargest(self.top_n, 'SIMILARITY', keep='first')
        )

        # 最终排序
        return result.sort_values(
            self.group_columns + ['SIMILARITY'],
            ascending=[True]*len(self.group_columns) + [False],
            kind='mergesort'
        ).reset_index(drop=True)