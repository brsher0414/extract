import os
import numpy as np
import pandas as pd
from typing import List, Tuple

class ReportProcessor:
    def __init__(self,
                 group_columns: List[str] = ['PLATFORM', 'CATCODE'],
                 similarity_threshold: float = 0.6,
                 top_n: int = 20,
                 summary_path: str = "processing_summary.csv"):  
        self.group_columns = group_columns
        self.threshold = similarity_threshold
        self.top_n = top_n
        self.mask_path = "similarity_mask.npy"
        self.summary_path = summary_path 

    def _generate_mask(self, similarity_matrix: np.ndarray) -> np.ndarray:
        mask = np.all(similarity_matrix <= self.threshold, axis=1)
        np.save(self.mask_path, mask)
        return np.load(self.mask_path, mmap_mode='r')

    def _generate_summary(self, raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> pd.DataFrame:
        raw_data = raw_data.copy()
        raw_data[self.group_columns] = raw_data[self.group_columns].astype('category')
        
        raw_counts = raw_data.groupby(
            self.group_columns, observed=True, dropna=False
        ).size().rename('RAW_COUNT').reset_index()

        processed_counts = processed_data.groupby(
            self.group_columns, observed=True, dropna=False
        ).size().rename('PROCESSED_COUNT').reset_index()

        raw_counts[self.group_columns] = raw_counts[self.group_columns].astype('str')
        processed_counts[self.group_columns] = processed_counts[self.group_columns].astype('str')
        
        summary = pd.merge(
            raw_counts,
            processed_counts,
            on=self.group_columns,
            how='outer'
        ).fillna(0)

        summary['FILTERED_COUNT'] = summary['RAW_COUNT'] - summary['PROCESSED_COUNT']
        summary = summary[summary['FILTERED_COUNT'] >= 0]  # 确保无负数

        summary['THRESHOLD'] = self.threshold
        summary['TOP_N'] = self.top_n
        return summary.sort_values('RAW_COUNT', ascending=False)

    def process(self,
               data: pd.DataFrame,
               similarity_matrix: np.ndarray,
               save_summary: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """返回处理结果和统计表"""
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

        result = filtered_data.groupby(
            self.group_columns,
            observed=True,
            dropna=False,  
            group_keys=False
        ).apply(
            lambda x: x.nlargest(self.top_n, 'SIMILARITY', keep='first')
        ).sort_values(
            self.group_columns + ['SIMILARITY'],
            ascending=[True]*len(self.group_columns) + [False],
            kind='mergesort'
        ).reset_index(drop=True)

        # 生成统计表
        summary_table = self._generate_summary(data, result)
        
        # 可选保存统计表
        if save_summary:
            summary_table.to_csv(self.summary_path, index=False)
            print(f"统计表已保存至: {self.summary_path}")

        return result, summary_table