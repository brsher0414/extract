import gc
import torch

def clean_text(text):
    """文本清洗函数"""
    if not isinstance(text, str):
        return ""
    return text.strip()

def optimize_memory():
    """内存优化"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()