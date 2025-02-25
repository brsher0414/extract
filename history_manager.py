# history_manager.py
import os
import re
from storage_manager import StorageManager

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