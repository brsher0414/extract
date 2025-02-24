# Text Similarity Processor

## 项目简介
一个支持断点续传的文本相似度计算工具，适合处理大规模数据。  
- 从 Excel/CSV 加载数据
- 用 BERT 等模型生成文本向量
- 自动续传中断任务
- 生成可视化报告

---

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 运行程序
python main.py
```

## ⚙️ 配置指南
### 修改 `config.py`：
```python
# 输入文件路径
INPUT_DATA = {
    "old_excel": "data/old.xlsx",  # 📂 旧数据
    "new_csv": "data/new.csv"      # 📂 新数据
}

# 输出目录
OUTPUT_STRUCTURE = {
    "vectors": "output/vectors/{timestamp}",  # 💾 向量文件
    "indices": "output/indices/{timestamp}",  # 🔍 索引文件
    "reports": "output/reports/{timestamp}"   # 📊 报告文件
}

# 模型参数
MODEL_NAME = "hfl/chinese-bert-wwm-ext"  # 🤖 使用的中文模型
BATCH_SIZE = 512                  # 🔢 每批处理512条文本
MAX_LENGTH = 64                   # 📏 文本最大长度
```

## 修改列映射 (main.py)
```python
COLUMN_MAPPING = {
    "jd": 5,          # 📜 旧数据文本列
    "taobao": 5,      # 📜 旧数据文本列
    "kuaishou_new": 7,# 📜 旧数据文本列
    "douyin_new": 6   # 📜 旧数据文本列
}
```

## ⏯️ 断点续传
### 使用方法
```bash
# 中断后重新运行即可续传
python main.py
```
