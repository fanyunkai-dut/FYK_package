#查询某个站空值占比
import pandas as pd

def count_valid_and_nan(csv_file_path, column_name="总氮", encoding=None):
    # 读取CSV
    df = pd.read_csv(csv_file_path, encoding=encoding) if encoding else pd.read_csv(csv_file_path)

    if column_name not in df.columns:
        raise ValueError(f"列 '{column_name}' 不存在，当前列有：{list(df.columns)}")

    s = df[column_name]

    # 有值：非空且不是NaN
    valid_count = int(s.notna().sum())
    nan_count = int(s.isna().sum())
    total = valid_count + nan_count
    nan_ratio = nan_count / total if total > 0 else 0.0

    print(f"有值数量为：{valid_count}，空值数量为：{nan_count}，空值占比为：{nan_ratio:.4%}")

    return valid_count, nan_count, nan_ratio


# 示例：改成你的路径
csv_file_path = "/home/fanyunkai/FYK_data/processed_dataset2.5/羊尾.csv"
count_valid_and_nan(csv_file_path, column_name="总氮", encoding="utf-8-sig")  # 不确定编码就先试试utf-8-sig