import os
import numpy as np
import pandas as pd
from typing import List, Tuple

class ReportProcessor:
    def __init__(
        self,
        group_columns: List[str] = ['MKTID', 'CATCODE'],
        initial_similarity_threshold: float = 0.6,
        top_n: int = 20,
        summary_path: str = "processing_summary.csv"
    ):
        self.group_columns = group_columns
        self.initial_threshold = initial_similarity_threshold
        self.top_n = top_n
        self.mask_path = "similarity_mask.npy"
        self.summary_path = summary_path

    def process(
        self,
        data: pd.DataFrame,
        similarity_matrix: np.ndarray,
        save_summary: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if os.path.exists(self.mask_path):
            os.remove(self.mask_path)
        
        data = data.copy()
        data['SIMILARITY'] = similarity_matrix.mean(axis=1)
        data[self.group_columns] = data[self.group_columns].replace({np.nan: pd.NA})
        for col in self.group_columns:
            if data[col].dtype == 'object':
                data[col] = data[col].str.strip()
            data[col] = data[col].astype('string[pyarrow]') 
        
        groups = data.groupby(
            self.group_columns, 
            observed=True, 
            dropna=False, 
            group_keys=False
        )
        
        results = []
        threshold_records = []
        
        for group_key, group_data in groups:
            group_dict = {
                col: key if not pd.isna(key) else pd.NA
                for col, key in zip(self.group_columns, group_key)
            }
            
            threshold_samples = group_data[group_data['SIMILARITY'] <= self.initial_threshold]
            sorted_threshold = threshold_samples.sort_values('SIMILARITY', ascending=False)
            selected_initial = sorted_threshold.head(self.top_n)
            remaining = self.top_n - len(selected_initial)
            
            if remaining > 0:
                unselected_mask = ~group_data.index.isin(selected_initial.index)
                unselected_samples = group_data[unselected_mask]
                sorted_unselected = unselected_samples.sort_values('SIMILARITY', ascending=True)
                selected_unselected = sorted_unselected.head(remaining)
                selected = pd.concat([selected_initial, selected_unselected], ignore_index=True)
                used_threshold = max(
                    self.initial_threshold,
                    selected_unselected['SIMILARITY'].max() if not selected_unselected.empty else self.initial_threshold
                )
            else:
                selected = selected_initial
                used_threshold = self.initial_threshold
            
            results.append(selected)
            threshold_records.append({**group_dict, 'DYNAMIC_THRESHOLD': used_threshold})
        
        result_df = pd.concat(results, ignore_index=True)
        summary_table = self._generate_summary(data, result_df)
        
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
        
        summary_table['INITIAL_THRESHOLD'] = self.initial_threshold
        
        if save_summary:
            summary_table.to_csv(self.summary_path, index=False)
        
        return result_df, summary_table

    def _generate_summary(self, raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> pd.DataFrame:
        raw_counts = raw_data.groupby(
            self.group_columns, observed=True, dropna=False
        ).size().rename('RAW_COUNT').reset_index()
        
        processed_counts = processed_data.groupby(
            self.group_columns, observed=True, dropna=False
        ).size().rename('PROCESSED_COUNT').reset_index()

        summary = pd.merge(
            raw_counts,
            processed_counts,
            on=self.group_columns,
            how='outer'
        ).fillna({'RAW_COUNT': 0, 'PROCESSED_COUNT': 0})
        
        # 确保空值类型统一，避免后续重复处理
        for col in self.group_columns:
            summary[col] = summary[col].replace({np.nan: pd.NA}).astype('string[pyarrow]')
        
        summary['FILTERED_COUNT'] = summary['RAW_COUNT'] - summary['PROCESSED_COUNT']
        summary = summary[summary['FILTERED_COUNT'] >= 0]
        summary['TOP_N'] = self.top_n
        
        return summary.sort_values('RAW_COUNT', ascending=False)