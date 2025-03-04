import os
from datetime import datetime

BASE_DIR = r"C:\Users\libi4006\Downloads\extract-main"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")

def get_latest_file(directory, extensions, pattern=None):
    """获取目录下符合扩展名和文件名模式的最新文件"""
    matched_files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            # 检查文件扩展名
            if isinstance(extensions, str):
                valid_extension = file.endswith(extensions)
            else:
                valid_extension = any(file.endswith(ext) for ext in extensions)
            # 检查文件名模式
            if valid_extension and (pattern is None or pattern in file):
                matched_files.append(file_path)
    
    if not matched_files:
        raise FileNotFoundError(f"No files found in {directory} matching criteria.")
    
    # 按修改时间排序，返回最新文件
    return max(matched_files, key=os.path.getmtime)

INPUT_DATA = {
    # 旧Excel文件：匹配.xlsx且文件名包含"new_model_samples"
    "old_excel": get_latest_file(RAW_DATA_DIR, '.xlsx', 'new_model_samples'),
    # 新CSV文件：匹配.csv且文件名包含"new_data"
    "new_csv": get_latest_file(RAW_DATA_DIR, '.csv', 'new_data')
}
OUTPUT_STRUCTURE = {
    "vectors": os.path.join(BASE_DIR, "data/vectors/{timestamp}"),
    "indices": os.path.join(BASE_DIR, "data/indices/{timestamp}"),
    "reports": os.path.join(BASE_DIR, "output/{timestamp}")
}

# 模型参数
BATCH_SIZE = 256
MAX_LENGTH = 128
MODEL_NAME = "hfl/chinese-bert-wwm-ext"
VECTOR_DTYPE = "float32"