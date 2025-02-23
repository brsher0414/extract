from config import *
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_sheets, extract_descriptions
from vectorizer import Vectorizer
from similarity import SimilarityEngine
from utils import clean_text, optimize_memory
import h5py
from storage_manager import StorageManager

COLUMN_MAPPING = {
    "jd": 5,
    "taobao": 5,
    "kuaishou_new": 7,
    "douyin_new": 6
}

def main():
    # 初始化存储管理器
    storage = StorageManager()
    
    # --- 旧数据处理 ---
    print("加载旧数据...")
    old_data = load_sheets(INPUT_DATA["old_excel"])
    old_descs = [clean_text(d) for d in extract_descriptions(old_data, COLUMN_MAPPING)]
    
    # --- 新数据处理 ---
    print("加载新数据...")
    new_data = pd.read_csv(INPUT_DATA["new_csv"], low_memory=False)
    new_descs = [clean_text(d) for d in new_data['PROD_DESC_RAW']]
    
    # --- 向量化 ---
    vec_engine = Vectorizer()
    
    # 旧数据向量化
    old_vector_dir = storage.get_path("vectors")
    old_vector_path = os.path.join(old_vector_dir, f"old_vectors_{storage.timestamp}.h5")
    vec_engine.vectorize(old_descs, old_vector_path, resume=True)
    
    # 新数据向量化
    new_vector_dir = storage.get_path("vectors")
    new_vector_path = os.path.join(new_vector_dir, f"new_vectors_{storage.timestamp}.h5")
    vec_engine.vectorize(new_descs, new_vector_path, resume=True)
    
    # --- 构建索引 ---
    index_dir = storage.get_path("indices")
    index_path = os.path.join(index_dir, f"index_{storage.timestamp}.faiss")
    
    with h5py.File(old_vector_path, 'r') as f:
        se = SimilarityEngine(f['vectors'].shape[1])
        se.build_index(f['vectors'][:])
        se.save_index(index_path)  # 假设 SimilarityEngine 添加了 save_index 方法
    
    # --- 相似度分析 ---
    with h5py.File(new_vector_path, 'r') as f:
        similarities = se.batch_search(f['vectors'][:])
    
    # --- 结果处理 ---
    new_data['SIMILARITY'] = similarities.mean(axis=1)
    result = new_data[new_data['SIMILARITY'] <= 0.6]
    
    # --- 保存报告 ---
    report_dir = storage.get_path("reports")
    
    # 可视化报告
    plt.hist(similarities.flatten(), bins=50)
    plt.savefig(os.path.join(report_dir, f"similarity_dist_{storage.timestamp}.png"))
    
    # Excel报告
    result.to_excel(
        os.path.join(report_dir, f"low_similarity_{storage.timestamp}.xlsx"),
        index=False,
        engine='openpyxl'
    )

    print(f"处理完成！所有结果保存在版本目录：{storage.timestamp}")

if __name__ == "__main__":
    main()