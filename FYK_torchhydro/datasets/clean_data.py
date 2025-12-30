import pandas as pd
import numpy as np

def clean_data(csv_file_path, column_name, window_size=100, threshold=4, output_folder="./cleaned_data"):
    """
    清洗 CSV 数据，去除异常值。
    Parameters
    ----------
    csv_file_path : str
        CSV 文件的路径。
    column_name : list
        要清洗的列的名称，可以是一个字符串或一个包含多个列名的列表。
    window_size : int, optional, default=100
        滑动窗口的大小，前后各取多少个值进行比较。
    threshold : int, optional, default=4
        异常值的阈值倍数，当前值大于前后窗口平均值的 threshold 倍认为是异常值。
    output_folder : str, optional, default="./cleaned_data"
        清洗后的数据保存路径。
    Returns
    -------
    None
        该方法将清洗后的 CSV 文件保存到指定文件夹并打印删除的异常值数量。
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 检查每个列名是否存在
    for col in column_name:
        if col not in df.columns:
            raise ValueError(f"指定的列 '{col}' 在 CSV 文件中不存在。")

    # 数据清洗操作（滑动窗口）
    for col in column_name:
        anomaly_flags = []
        for idx in range(len(df)):
            # 获取当前值
            current_value = df[col].iloc[idx]
            
            # 如果当前值为缺失值，跳过
            if pd.isna(current_value):
                anomaly_flags.append(np.nan)
                continue

            # 确定前后窗口范围
            start_idx = max(0, idx - window_size)
            end_idx = min(len(df), idx + window_size + 1)

            # 获取前后窗口的数据
            window_data = df[col].iloc[start_idx:end_idx]

            # 排除缺失值
            valid_window_data = window_data.dropna()

            # 如果有效数据数量小于50个，跳过
            if len(valid_window_data) < 50:
                anomaly_flags.append(np.nan)
                continue

            # 计算前后窗口的平均值
            mean_value = valid_window_data.mean()

            # 判断当前值是否超过前后窗口平均值的 threshold 倍
            if current_value > mean_value * threshold:
                anomaly_flags.append(np.nan)
            else:
                anomaly_flags.append(current_value)

        # 将异常值标记到数据中
        df[col] = anomaly_flags

    # 计算并输出删除了多少个异常值
    deleted_count = df[column_name].isna().sum().sum()  # 对所有列计算 NaN 总数
    print(f"在文件 {csv_file_path} 中，删除了 {deleted_count} 个异常值。")


csv_file_path = "/home/fanyunkai/FYK_data/complete water quality/湖口.csv"
column_name = ["总氮"]
clean_data(csv_file_path, column_name)
