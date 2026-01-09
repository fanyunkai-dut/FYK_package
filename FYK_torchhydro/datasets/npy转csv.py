#把npy转换成csv以查看内容
import numpy as np

# NPY 转换为 CSV
def npy_to_csv(npy_file_path, csv_file_path):
    # 加载 NPY 文件
    data = np.load(npy_file_path)
    
    # 将 NumPy 数组保存为 CSV 文件
    np.savetxt(csv_file_path, data, delimiter=",", fmt="%.6f")
    
    print(f"✅ NPY 文件已转换并保存为 {csv_file_path}")

# 调用示例
npy_file_path = "/home/fanyunkai/FYK_data/processed_dataset2.5/WQ_hanjiang_true_train.npy"
csv_file_path = "/home/fanyunkai/FYK_data/processed_dataset2.5/WQ_hanjiang_true_train.csv"

npy_to_csv(npy_file_path, csv_file_path)

