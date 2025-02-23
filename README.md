#Text Similarity Processor
项目简介
这是一个用于计算文本相似度的工具。它可以从 Excel 和 CSV 文件加载文本数据，使用预训练的语言模型将文本转换为向量，并计算文本之间的相似度。支持断点续传，适合处理大规模数据。

快速开始
安装依赖
pip install -r requirements.txt
运行程序
python main.py
可配置参数
在 config.py 中修改以下参数：

1. 输入文件路径
INPUT_DATA = {
    "old_excel": "path/to/old_data.xlsx",  # 旧数据（Excel 文件）
    "new_csv": "path/to/new_data.csv"      # 新数据（CSV 文件）
}
2. 输出目录结构
OUTPUT_STRUCTURE = {
    "vectors": "output/vectors/{timestamp}",  # 向量数据
    "indices": "output/indices/{timestamp}",  # 相似度索引
    "reports": "output/reports/{timestamp}"   # 报告文件
}
3. 模型配置
MODEL_NAME = "bert-base-chinese"  # 预训练语言模型
BATCH_SIZE = 32                   # 批处理大小
MAX_LENGTH = 128                  # 文本最大长度
4. 列映射
在 main.py 中修改 COLUMN_MAPPING，指定文本所在的列：

COLUMN_MAPPING = {
    "jd": 5,          # 旧数据中文本所在的列
    "taobao": 5,      # 旧数据中文本所在的列
    "kuaishou_new": 7, # 旧数据中文本所在的列
    "douyin_new": 6    # 旧数据中文本所在的列
}
断点续传
如果程序在处理过程中中断（如按 Ctrl+C），可以重新运行程序，它会自动从断点继续处理。

如何工作？
程序会在每次处理完一个批次后，保存一个 .checkpoint 文件。
重新运行时，程序会检测 .checkpoint 文件，并从上次中断的位置继续处理。
注意事项
不要手动删除 .checkpoint 文件或未完成的 .h5 文件。
如果中断后修改了输入数据，可能会导致数据不一致。
输出文件
程序会生成以下文件：

向量数据：存储在 output/vectors/ 目录下，格式为 .h5。
相似度索引：存储在 output/indices/ 目录下，格式为 .faiss。
报告文件：存储在 output/reports/ 目录下，包括：
相似度分布图（similarity_dist_YYYYMMDD_HHMMSS.png）。
相似度较低的文本列表（low_similarity_YYYYMMDD_HHMMSS.xlsx）。
常见问题
1. 如何修改模型？
在 config.py 中修改 MODEL_NAME 参数，例如：

MODEL_NAME = "bert-base-multilingual-cased"
2. 如何处理其他数据格式？
在 data_loader.py 中添加新的数据加载函数，并在 main.py 中调用。

3. 如何强制重新开始？
删除 .checkpoint 文件和未完成的 .h5 文件，然后重新运行程序。

