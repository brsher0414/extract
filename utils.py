import gc
import torch

def clean_text(text):
    """文本清洗函数"""
    if not isinstance(text, str):
        return ""
    return text.strip()
