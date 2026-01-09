#单个站点清洗异常值代码
import os
import pandas as pd
import numpy as np


def clean_data(
    csv_file_path,
    column_name,
    window_size=100,
    threshold_high=4,
    threshold_low=0.25,
    min_valid_neighbors=50,
    output_csv_path=None,
    encoding=None,  # 需要时可传 "utf-8-sig"/"gbk"
):
    """
    清洗 CSV 数据：
    1) 统计原始空白(NaN)数量
    2) 把“非正值”(<=0)全部置为 NaN（并统计非正值数量：包含0和负）
    3) 滑动窗口清洗异常高值： >= mean * threshold_high -> NaN（统计）
    4) 滑动窗口清洗异常低值： <= mean * threshold_low  -> NaN（统计）
       - mean 仅用窗口内非空值
       - 边界不足时自动用可用的前/后数据
       - 窗口内有效值少于 min_valid_neighbors 时不做异常判定（保留原值）
    5) 输出统计：原来的空白值、非正值、异常高值、异常低值、正常值
    """
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

    print(
        f"文件: {csv_file_path}\n"
        f"原来的空白值有多少: {stats['orig_nan']}\n"
        f"非正值(<=0)有多少: {stats['non_positive']}\n"
        f"异常高值有多少: {stats['high_anomaly']}\n"
        f"异常低值有多少: {stats['low_anomaly']}\n"
        f"正常值有多少: {stats['normal']}"
    )

    if output_csv_path is not None:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"清洗后的 CSV 已保存到: {output_csv_path}")

    return df


# ===== 你的调用 =====
csv_file_path = "/home/fanyunkai/FYK_data/complete water quality/小钢桥.csv"
column_name = ["总氮"]
output_csv_path = "/home/fanyunkai/FYK_data/cleaned_data/小钢桥.csv"

clean_data(
    csv_file_path,
    column_name,
    window_size=100,
    threshold_high=4,
    threshold_low=0.25,
    min_valid_neighbors=50,
    output_csv_path=output_csv_path,
)




