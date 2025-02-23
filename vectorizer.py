import os
import h5py
import json
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from config import MODEL_NAME, VECTOR_DTYPE, BATCH_SIZE, MAX_LENGTH


class Vectorizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def vectorize(self, texts, output_path, resume=False):
        """支持断点续传的向量化函数"""
        total = len(texts)
        start_idx = 0

        # 检查是否需要续传
        checkpoint_file = f"{output_path}.checkpoint"
        if resume and os.path.exists(output_path):
            with h5py.File(output_path, 'a') as hf:
                if 'vectors' in hf:
                    start_idx = hf['vectors'].shape[0]
                    print(f"检测到已有进度，从第 {start_idx} 条继续处理")

        # 初始化HDF5文件
        with h5py.File(output_path, 'a') as hf:
            if 'vectors' not in hf:
                # 创建可扩展数据集
                hf.create_dataset(
                    'vectors',
                    shape=(0, self.model.config.hidden_size),  # 初始为空
                    maxshape=(None, self.model.config.hidden_size),  # 第一维可扩展
                    dtype=VECTOR_DTYPE,
                    compression="gzip"
                )

            # 进度条和断点保存
            with tqdm(initial=start_idx, total=total, desc="向量化进度") as pbar:
                try:
                    for i in range(start_idx, total, BATCH_SIZE):
                        batch = texts[i:i + BATCH_SIZE]

                        inputs = self.tokenizer(
                            batch,
                            max_length=MAX_LENGTH,
                            truncation=True,
                            padding='longest',
                            return_tensors="pt"
                        ).to(self.device)

                        with torch.no_grad():
                            outputs = self.model(**inputs)

                        vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(VECTOR_DTYPE)

                        # 动态扩展数据集并写入数据
                        current_size = hf['vectors'].shape[0]
                        new_size = current_size + len(batch)
                        hf['vectors'].resize(new_size, axis=0)
                        hf['vectors'][current_size:new_size] = vectors

                        # 更新检查点
                        with open(checkpoint_file, 'w') as f:
                            json.dump({'last_index': i + len(batch)}, f)

                        pbar.update(len(batch))

                except KeyboardInterrupt:
                    print(f"\n手动中断！下次运行可从第 {i} 条继续")
                    exit(1)

        # 清理检查点
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return output_path