import pandas as pd

def load_sheets(excel_path):
    """加载Excel所有sheet"""
    sheets = {}
    with pd.ExcelFile(excel_path) as reader:
        for sheet_name in reader.sheet_names:
            sheets[sheet_name] = pd.read_excel(reader, sheet_name)
    return sheets

def extract_descriptions(data, column_mapping):
    """根据列映射提取描述字段（新增描述cate过滤）"""
    descriptions = []
    exclude_keywords = {"不确定", "套装产品不确定"}
    
    for sheet_name, df in data.items():
        col_idx = column_mapping.get(sheet_name)
        if col_idx is None:
            continue
        
        # 新增：获取描述cate列位置（prod_desc_raw右边一列）
        cate_col_idx = col_idx + 1
        
        # 新增：过滤逻辑（处理列越界情况）
        if cate_col_idx < len(df.columns):
            # 构建过滤掩码
            mask = ~df.iloc[:, cate_col_idx].astype(str).str.contains(
                '|'.join(exclude_keywords), 
                case=False, 
                na=False
            )
            df = df[mask]
        else:
            print(f"警告：{sheet_name} sheet缺少描述cate列（位置{cate_col_idx}），跳过过滤")
        
        # 提取处理后的描述
        for _, row in df.iterrows():
            desc = str(row.iloc[col_idx]).strip()  # 使用iloc按列位置索引
            if desc:
                descriptions.append(desc)
                
    return descriptions