import os
import numpy as np
import pandas as pd
from typing import List, Tuple

class ReportProcessor:
    def __init__(self,
                 group_columns: List[str] = ['PLATFORM', 'CATCODE'],
                 initial_similarity_threshold: float = 0.6,
                 top_n: int = 20,
                 summary_path: str = "processing_summary.csv"):  
        self.group_columns = group_columns
        self.initial_threshold = initial_similarity_threshold
        self.top_n = top_n
        self.mask_path = "similarity_mask.npy"
        self.summary_path = summary_path

    def process(self,
               data: pd.DataFrame,
               similarity_matrix: np.ndarray,
               save_summary: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if os.path.exists(self.mask_path):
            os.remove(self.mask_path)
        
        # 添加相似度列并处理空值
        data = data.copy()
        data['SIMILARITY'] = similarity_matrix.mean(axis=1)
        data[self.group_columns] = data[self.group_columns].replace({np.nan: pd.NA})
        
        # 按分组处理
        groups = data.groupby(self.group_columns, observed=True, dropna=False, group_keys=False)
        
        results = []
        threshold_records = []
        
        for group_key, group_data in groups:
            # 处理分组键中的空值
            group_keys = tuple(pd.NA if pd.isna(k) else k for k in 
                             (group_key if isinstance(group_key, tuple) else (group_key,)))
            group_dict = {
                col: (pd.NA if pd.isna(key) else key)
                for col, key in zip(self.group_columns, group_keys)
            }
            
            # --- 核心逻辑修改开始 ---
            # 第一阶段：阈值内样本降序抽取（最相似在前）
            threshold_samples = group_data[group_data['SIMILARITY'] <= self.initial_threshold]
            sorted_threshold = threshold_samples.sort_values('SIMILARITY', ascending=False)
            selected_initial = sorted_threshold.head(self.top_n)
            remaining = self.top_n - len(selected_initial)
            
            # 第二阶段：从未选中的样本中补足（升序抽取最不相似）
            if remaining > 0:
                # 获取未选中的样本（包括阈值外和阈值内未选中的）
                unselected_mask = ~group_data.index.isin(selected_initial.index)
                unselected_samples = group_data[unselected_mask]
                
                # 升序排列并补足
                sorted_unselected = unselected_samples.sort_values('SIMILARITY', ascending=True)
                selected_unselected = sorted_unselected.head(remaining)
                
                # 合并结果
                selected = pd.concat([selected_initial, selected_unselected], ignore_index=True)
                
                # 计算动态阈值（取补足样本中的最大相似度）
                used_threshold = max(
                    self.initial_threshold,
                    selected_unselected['SIMILARITY'].max() if not selected_unselected.empty else self.initial_threshold
                )
            else:
                selected = selected_initial
                used_threshold = self.initial_threshold
            # --- 核心逻辑修改结束 ---
            
            results.append(selected)
            threshold_records.append({**group_dict, 'DYNAMIC_THRESHOLD': used_threshold})
        
        # 合并所有分组结果
        result_df = pd.concat(results, ignore_index=True)
        
        # 生成统计表
        summary_table = self._generate_summary(data, result_df)
        
        # 合并动态阈值信息
        threshold_df = pd.DataFrame(threshold_records)
        if not threshold_df.empty:
            summary_table = pd.merge(
                summary_table,
                threshold_df,
                on=self.group_columns,
                how='left'
            )
        else:
            summary_table['DYNAMIC_THRESHOLD'] = 0.0
        
        # 空值一致性处理
        for col in self.group_columns:
            summary_table[col] = summary_table[col].replace({np.nan: pd.NA})
            summary_table[col] = summary_table[col].astype('object').where(
                summary_table[col].notna(), pd.NA
            )
        
        summary_table['INITIAL_THRESHOLD'] = self.initial_threshold
        
        if save_summary:
            summary_table.to_csv(self.summary_path, index=False)
        
        return result_df, summary_table

    def _generate_summary(self, raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> pd.DataFrame:
        # 原始数据分组计数
        raw_counts = raw_data.groupby(
            self.group_columns, observed=True, dropna=False
        ).size().rename('RAW_COUNT').reset_index()
        
        # 处理结果分组计数
        processed_counts = processed_data.groupby(
            self.group_columns, observed=True, dropna=False
        ).size().rename('PROCESSED_COUNT').reset_index()

        # 合并统计信息
        summary = pd.merge(
            raw_counts,
            processed_counts,
            on=self.group_columns,
            how='outer'
        ).fillna({'RAW_COUNT': 0, 'PROCESSED_COUNT': 0})
        
        # 恢复空值标识
        for col in self.group_columns:
            summary[col] = summary[col].replace({np.nan: pd.NA})
            summary[col] = summary[col].astype('object').where(
                summary[col].notna(), pd.NA
            )
        
        # 计算过滤数量
        summary['FILTERED_COUNT'] = summary['RAW_COUNT'] - summary['PROCESSED_COUNT']
        summary = summary[summary['FILTERED_COUNT'] >= 0]
        summary['TOP_N'] = self.top_n
        
        return summary.sort_values('RAW_COUNT', ascending=False)