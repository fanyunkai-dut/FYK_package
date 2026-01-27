import numpy as np
import pandas as pd

# CSV 转换为 NPY
def csv_to_npy(csv_file_path, npy_file_path):
    # 使用 pandas 读取 CSV 文件
    df = pd.read_csv(csv_file_path)
    
    # 将 DataFrame 转换为 NumPy 数组
    data = df.values
    
    # 保存为 NPY 文件
    np.save(npy_file_path, data)
    
    print(f"✅ CSV 文件已转换并保存为 {npy_file_path}")

# 调用示例
csv_file_path = "/home/fanyunkai/FYK_data/processed_dataset2.5/adj.csv"
npy_file_path = "/home/fanyunkai/FYK_data/processed_dataset2.5/adj.npy"

csv_to_npy(csv_file_path, npy_file_path)
