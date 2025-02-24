from config import *
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from similarity import SimilarityEngine
import h5py
from storage_manager import StorageManager
from report_processor import ReportProcessor
import os
import re

def get_latest_timestamp_dir(parent_dir):
    """获取指定目录下最新时间戳文件夹"""
    subdirs = [d for d in os.listdir(parent_dir) 
               if os.path.isdir(os.path.join(parent_dir, d))]
    
    timestamp_dirs = [d for d in subdirs if re.match(r"\d{8}_\d{6}", d)]
    
    if not timestamp_dirs:
        raise ValueError(f"No timestamp directories found in {parent_dir}")
    
    return max(timestamp_dirs)

data_dir = os.path.join(BASE_DIR, "data")
vectors_dir = os.path.join(data_dir, "vectors")
indices_dir = os.path.join(data_dir, "indices")
report_dir = os.path.join(BASE_DIR, "output/{timestamp}")


latest_timestamp = get_latest_timestamp_dir(vectors_dir)

def generator():
    storage = StorageManager()

    
    print("加载新数据...")
    new_data = pd.read_csv(INPUT_DATA["new_csv"], low_memory=False)

    vector_dir = os.path.join(vectors_dir, latest_timestamp)
    old_vector_path = os.path.join(vector_dir, f"old_vectors_{latest_timestamp}.h5")
    new_vector_path = os.path.join(vector_dir, f"new_vectors_{latest_timestamp}.h5")

    index_dir = os.path.join(indices_dir, latest_timestamp)
    index_path = os.path.join(index_dir, f"index_{latest_timestamp}.faiss")
    
    with h5py.File(old_vector_path, 'r') as f:
        se = SimilarityEngine(f['vectors'].shape[1])
        se.build_index(f['vectors'][:])
        se.save_index(index_path) 

    with h5py.File(new_vector_path, 'r') as f:
        similarities = se.batch_search(f['vectors'][:])

    new_data['SIMILARITY'] = similarities.mean(axis=1)

    processor = ReportProcessor(
        group_columns=['PLATFORM', 'CATCODE'],
        similarity_threshold=0.7,
        top_n=20
    )

    result_df, summary_df = processor.process(new_data, similarities)  # 解包两个返回值

    # 导出结果数据
    result_df.to_excel(
        os.path.join(report_dir, f"low_similarity_{storage.timestamp}.xlsx"),
        index=False,
        engine='openpyxl'
    )
        
    summary_df.to_excel(
        os.path.join(report_dir, f"summary_{storage.timestamp}.xlsx"),
        index=False,
        engine='openpyxl'
    )


    plt.hist(similarities.flatten(), bins=50)
    plt.savefig(os.path.join(report_dir, f"similarity_dist_{storage.timestamp}.png"))

    print(f"处理完成！所有结果保存在版本目录：{storage.timestamp}")

if __name__ == "__main__":
    generator()