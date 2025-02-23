import pandas as pd

def load_sheets(excel_path):
    """加载Excel所有sheet"""
    sheets = {}
    with pd.ExcelFile(excel_path) as reader:
        for sheet_name in reader.sheet_names:
            sheets[sheet_name] = pd.read_excel(reader, sheet_name)
    return sheets

def extract_descriptions(data, column_mapping):
    """根据列映射提取描述字段"""
    descriptions = []
    for sheet_name, df in data.items():
        col_idx = column_mapping.get(sheet_name)
        if col_idx is not None:
            for _, row in df.iterrows():
                descriptions.append(str(row[col_idx]).strip())
    return descriptions