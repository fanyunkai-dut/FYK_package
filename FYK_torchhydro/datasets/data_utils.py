# data_utils.py
import os
import pandas as pd
import numpy as np
"""
    处理数据，补充缺失的时间点,从2020/11/09 00:00:00到2025/09/30 20:00:00,按4小时间隔补全时间点

    Parameters
    ----------
    input_csv : str
        输入的 CSV 文件路径
    start_time : str
        处理数据的开始时间
    end_time : str
        处理数据的结束时间

    Returns
    -------
    pd.DataFrame
"""
def complete_data(input_csv, start_time="2020/11/09 00:00:00", end_time="2025/09/30 20:00:00"):
    # 读取原始CSV数据
    df = pd.read_csv(input_csv)

    # 确保“监测时间”列是 datetime 格式
    df['监测时间'] = pd.to_datetime(df['监测时间'], format='%Y-%m-%d %H:%M:%S')

    # 筛选出从2020/11/09 00:00:00开始的数据
    df = df[df['监测时间'] >= pd.to_datetime(start_time)]

    # 创建一个从开始时间到结束时间，按4小时间隔的时间序列
    time_range = pd.date_range(start=start_time, end=end_time, freq='4H')

    # 设置监测时间为索引
    df.set_index('监测时间', inplace=True)

    # 创建一个完整的时间索引，并与原始数据合并
    full_df = pd.DataFrame(index=time_range)

    # 合并数据，原始数据会填充到对应时间点上，缺失的时间点会填充NaN
    merged_df = full_df.join(df)

    return merged_df
    
    """
    清洗 CSV 数据：
    1) 统计原始空白(NaN)数量
    2) 把“非正值”(<=0)全部置为 NaN(并统计非正值数量:包含0和负)
    3) 滑动窗口清洗异常高值： >= mean * threshold_high -> NaN(统计)(前100个点和后100个点组成的窗口的平均值,如果大于这个值的4倍或小于这个值的0.25倍，则标记为异常值，且取空值)
    4) 滑动窗口清洗异常低值： <= mean * threshold_low  -> NaN(统计)
       - mean 仅用窗口内非空值
       - 边界不足时自动用可用的前/后数据
       - 窗口内有效值少于 min_valid_neighbors 时不做异常判定(保留原值)
    5) 输出统计：原来的空白值、非正值、异常高值、异常低值、正常值
    """
def clean_data(
    csv_file_path,
    column_name,
    window_size=100,
    threshold_high=2.5,
    threshold_low=0.4,
    min_valid_neighbors=50,
    output_csv_path=None,
    encoding=None,  # 需要时可传 "utf-8-sig"/"gbk"
):
    df = pd.read_csv(csv_file_path, encoding=encoding) if encoding else pd.read_csv(csv_file_path)

    if isinstance(column_name, str):
        column_name = [column_name]

    for col in column_name:
        if col not in df.columns:
            raise ValueError(f"指定的列 '{col}' 在 CSV 文件中不存在。")

    stats = {
        "orig_nan": 0,
        "non_positive": 0,
        "high_anomaly": 0,
        "low_anomaly": 0,
        "normal": 0,
    }

    for col in column_name:
        # 强制转成数值；无法转换的当作 NaN
        s_num = pd.to_numeric(df[col], errors="coerce")

        # 1) 原始空白
        orig_nan_mask = s_num.isna()
        stats["orig_nan"] += int(orig_nan_mask.sum())

        # 2) 非正值（<=0），但不把原始 NaN 算进去
        non_pos_mask = (~orig_nan_mask) & (s_num <= 0)
        stats["non_positive"] += int(non_pos_mask.sum())

        cleaned = s_num.copy()
        cleaned[non_pos_mask] = np.nan

        # 3/4) 滑动窗口异常高/低
        values = cleaned.to_numpy(dtype=float)
        n = len(values)
        new_values = values.copy()

        high_cnt = 0
        low_cnt = 0

        for idx in range(n):
            cur = values[idx]
            if np.isnan(cur):
                continue

            start_idx = max(0, idx - window_size)
            end_idx = min(n, idx + window_size + 1)

            window_vals = values[start_idx:end_idx]
            valid = window_vals[~np.isnan(window_vals)]

            if valid.size < min_valid_neighbors:
                continue

            mean_val = valid.mean()
            if mean_val == 0:
                # 极端情况：均值为0时，低值不判；高值只要>0就算高（按阈值会变成 >0）
                if cur > 0:
                    # 若你不想在 mean=0 时做任何判定，直接把这一段删掉即可
                    pass
                continue

            if cur >= mean_val * threshold_high:
                new_values[idx] = np.nan
                high_cnt += 1
                continue

            if cur <= mean_val * threshold_low:
                new_values[idx] = np.nan
                low_cnt += 1
                continue

        df[col] = new_values
        stats["high_anomaly"] += high_cnt
        stats["low_anomaly"] += low_cnt
        stats["normal"] += int(pd.Series(new_values).notna().sum())

    if output_csv_path is not None:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    return df