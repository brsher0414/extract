import os
from datetime import datetime

# 基础路径
BASE_DIR = r"C:\Users\Administrator\Desktop\extract\1\text_similarity_processor"

# 输入路径
INPUT_DATA = {
    "old_excel": os.path.join(BASE_DIR, "data/raw/new_model_samples_1018.xlsx"),
    "new_csv": os.path.join(BASE_DIR, "data/raw/20241405_new_data.csv")
}

# 输出目录结构
OUTPUT_STRUCTURE = {
    "vectors": os.path.join(BASE_DIR, "data/vectors/{timestamp}"),
    "indices": os.path.join(BASE_DIR, "data/indices/{timestamp}"),
    "reports": os.path.join(BASE_DIR, "output/{timestamp}")
}

# 模型参数
BATCH_SIZE = 512
MAX_LENGTH = 64
MODEL_NAME = "hfl/chinese-bert-wwm-ext"
VECTOR_DTYPE = "float32"