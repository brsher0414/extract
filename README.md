# 文本相似度分析工具

## 项目概述
本项目用于分析新旧商品描述文本的相似度，自动识别低相似度样本并生成可视化报告。基于预训练语言模型构建文本向量，使用FAISS进行高效相似度检索。

## 主要功能
- 📂 数据加载：支持Excel多sheet和CSV格式输入
- 🔢 向量化处理：使用`hfl/chinese-bert-wwm-ext`模型生成文本向量
- 🔍 相似度分析：基于FAISS索引实现批量相似度检索
- 📊 报告生成：
  - 低相似度样本报告（Excel格式）
  - 相似度分布直方图（PNG格式）
  - 支持按平台/类目分组分析

## 快速开始

### 环境安装
```bash
pip install -r requirements.txt
```

### requirements.txt
```bash
pandas>=1.3.0
numpy>=1.21.0
transformers>=4.12.0
torch>=1.10.0
h5py>=3.6.0
faiss-cpu>=1.7.0
tqdm>=4.60.0
matplotlib>=3.5.0
openpyxl>=3.0.0
```
### 使用步骤
#### 1.准备数据：
- 旧数据：放入 data/raw/
- 新数据：放入 data/raw/
#### 2.运行主程序：
  ```bash
  python main.py
  ```
#### 3.生成报告（可选）：
  ```bash
  python report_generator.py
  ```
## 高级功能
- 断点续传：向量化过程支持中断后继续运行
- 内存优化：使用内存映射处理大规模数据
- 自动版本管理：按时间戳隔离每次运行结果
