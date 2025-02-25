from config import *
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_sheets, extract_descriptions
from vectorizer import Vectorizer
from similarity import SimilarityEngine
from utils import clean_text
import h5py
from storage_manager import StorageManager
from report_processor import ReportProcessor
import os
import re
import numpy as np

COLUMN_MAPPING = {
    "jd": 5,
    "taobao": 5,
    "kuaishou_new": 7,
    "douyin_new": 6
}

def get_valid_historical_dirs(parent_dir, exclude_timestamp):
    """获取有效的历史目录（排除当前运行目录）"""
    print(f"\n正在扫描目录: {parent_dir}")
    
    if not os.path.exists(parent_dir):
        print("目录不存在，跳过扫描")
        return []
    
    dirs = [d for d in os.listdir(parent_dir) 
           if os.path.isdir(os.path.join(parent_dir, d))]
    print(f"找到原始目录列表: {dirs}")
    
    valid_dirs = []
    for d in dirs:
        # 验证目录格式和时间戳排除
        if re.match(r"\d{8}_\d{6}", d):
            if d == exclude_timestamp:
                print(f"排除当前运行目录: {d}")
                continue
                
            # 验证必须包含结果文件
            report_path = os.path.join(parent_dir, d, f"low_similarity_{d}.xlsx")
            check_result = os.path.exists(report_path)
            
            status = "有效" if check_result else "缺少报告文件"
            print(f"检查目录 {d}: {status}")
            
            if check_result:
                valid_dirs.append(d)
    
    print(f"最终有效目录列表: {valid_dirs}")
    return sorted(valid_dirs, reverse=True)

def load_historical_data(storage):
    """加载最新有效历史数据"""
    current_timestamp = storage.timestamp
    print(f"\n当前运行时间戳: {current_timestamp}")

    # 获取所有output历史目录
    output_parent = os.path.dirname(storage.get_path("reports"))
    print(f"正在output父目录查找历史报告: {output_parent}")
    
    hist_output_dirs = get_valid_historical_dirs(output_parent, current_timestamp)
    
    if not hist_output_dirs:
        print("未找到有效历史目录")
        return None, None
    
    # 取最新历史目录
    latest_output_dir = hist_output_dirs[0]
    print(f"\n找到最新有效历史目录: {latest_output_dir}")
    
    # 构建对应向量路径
    vectors_parent = os.path.dirname(storage.get_path("vectors"))
    old_vector_path = os.path.join(
        vectors_parent,
        latest_output_dir,
        f"old_vectors_{latest_output_dir}.h5"
    )
    print(f"预期旧向量路径: {old_vector_path}")
    
    # 构建报告路径
    report_path = os.path.join(
        output_parent,
        latest_output_dir,
        f"low_similarity_{latest_output_dir}.xlsx"
    )
    print(f"预期报告路径: {report_path}")
    
    # 最终验证文件存在性
    file_check = []
    file_check.append(os.path.exists(old_vector_path))
    file_check.append(os.path.exists(report_path))
    
    if not all(file_check):
        print("文件验证失败:")
        print(f"旧向量存在: {file_check[0]}")
        print(f"报告存在: {file_check[1]}")
        return None, None
    
    print("所有必需文件验证通过")
    return old_vector_path, report_path

def main():
    storage = StorageManager()


    print("加载新数据...")
    new_data = pd.read_csv(INPUT_DATA["new_csv"], low_memory=False)
    new_descs = [clean_text(d) for d in new_data['PROD_DESC_RAW']]
    
    print("处理旧数据...")
    old_vector_path, hist_report_path = load_historical_data(storage)
    
    if old_vector_path and hist_report_path:
        print(f"发现历史数据：{os.path.basename(old_vector_path)}")
        
        # 加载历史向量
        with h5py.File(old_vector_path, 'r') as f:
            hist_vectors = f['vectors'][:]
        
        # 从历史报告提取新增描述
        hist_report = pd.read_excel(hist_report_path, engine='openpyxl')
        new_descs_from_report = [clean_text(d) for d in hist_report.iloc[:, 3]]
        
        # 向量化新增描述
        vec_engine = Vectorizer()
        temp_vector_path = os.path.join(storage.get_path("vectors"), "temp.h5")
        vec_engine.vectorize(new_descs_from_report, temp_vector_path, resume=True)
        
        # 合并历史向量
        with h5py.File(temp_vector_path, 'r') as f:
            new_vectors = f['vectors'][:]
            combined_vectors = np.vstack([hist_vectors, new_vectors])
        
        # 保存当前旧向量
        current_old_vector_path = os.path.join(
            storage.get_path("vectors"),
            f"old_vectors_{storage.timestamp}.h5"
        )
        with h5py.File(current_old_vector_path, 'w') as f:
            f.create_dataset('vectors', data=combined_vectors)
        
        os.remove(temp_vector_path)
    else:
        print("无历史数据，初始化旧向量...")
        old_data = load_sheets(INPUT_DATA["old_excel"])
        old_descs = [clean_text(d) for d in extract_descriptions(old_data, COLUMN_MAPPING)]
        
        vec_engine = Vectorizer()
        current_old_vector_path = os.path.join(
            storage.get_path("vectors"),
            f"old_vectors_{storage.timestamp}.h5"
        )
        vec_engine.vectorize(old_descs, current_old_vector_path, resume=True)


    # --- 构建索引 ---
    print("构建索引...")
    index_dir = storage.get_path("indices")
    index_path = os.path.join(index_dir, f"index_{storage.timestamp}.faiss")
    
    with h5py.File(old_vector_path, 'r') as f:
        se = SimilarityEngine(f['vectors'].shape[1])
        se.build_index(f['vectors'][:])
        se.save_index(index_path) 
        
    new_vector_dir = storage.get_path("vectors")
    new_vector_path = os.path.join(new_vector_dir, f"new_vectors_{storage.timestamp}.h5")
    vec_engine.vectorize(new_descs, new_vector_path, resume=True)
    

        
    # --- 相似度分析 ---
    report_dir = storage.get_path("reports")
    with h5py.File(new_vector_path, 'r') as f:
        similarities = se.batch_search(f['vectors'][:])

    # --- 结果处理 ---
    new_data['SIMILARITY'] = similarities.mean(axis=1)

    processor = ReportProcessor(
        group_columns=['PLATFORM', 'CATCODE'],
        similarity_threshold=0.6,
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
    main()