import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import glob
from datetime import datetime
from config import OUTPUT_STRUCTURE

class StorageManager:
    def __init__(self):
        # 先检测是否存在未完成任务
        self.timestamp = self._detect_incomplete_job() or self._generate_timestamp()
        
    def _generate_timestamp(self):
        """生成新时间戳"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _detect_incomplete_job(self):
        """检测未完成任务"""
        # 获取所有向量目录
        vector_base = OUTPUT_STRUCTURE["vectors"].split("{")[0]
        all_versions = sorted(glob.glob(os.path.join(vector_base, "*")))
        
        # 逆序检查最新版本
        for version_dir in reversed(all_versions):
            # 检查是否存在检查点文件
            checkpoint_files = glob.glob(os.path.join(version_dir, "*.checkpoint"))
            if checkpoint_files:
                return os.path.basename(version_dir)
        return None
    
    def get_path(self, category):
        """获取路径（复用未完成版本或新建）"""
        path_template = OUTPUT_STRUCTURE[category]
        path = path_template.format(timestamp=self.timestamp)
        os.makedirs(path, exist_ok=True)
        return path