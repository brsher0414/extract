import os
import numpy as np
import pandas as pd
from typing import List, Tuple

class ReportProcessor:
    def __init__(self,
                 group_columns: List[str] = ['PLATFORM', 'CATCODE'],
                 similarity_threshold: float = 0.7,
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

    def _generate_summary(self,
                         raw_data: pd.DataFrame,
                         processed_data: pd.DataFrame) -> pd.DataFrame:
        """生成处理统计表"""
        # 统计原始数据量
        raw_counts = raw_data.groupby(
            self.group_columns, observed=True
        ).size().rename('RAW_COUNT').reset_index()

        # 统计处理后数据量
        processed_counts = processed_data.groupby(
            self.group_columns, observed=True
        ).size().rename('PROCESSED_COUNT').reset_index()

        # 全外连接合并统计结果
        summary = pd.merge(
            raw_counts,
            processed_counts,
            on=self.group_columns,
            how='outer'
        ).fillna(0)

        # 计算过滤数量
        summary['FILTERED_COUNT'] = summary['RAW_COUNT'] - summary['PROCESSED_COUNT']
        
        # 添加处理元信息
        summary['THRESHOLD'] = self.threshold
        summary['TOP_N'] = self.top_n
        
        # 按原始数据量排序
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

        # 全局分组筛选TopN
        result = filtered_data.groupby(
            self.group_columns,
            observed=True,
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