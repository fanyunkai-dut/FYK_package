# data_utils.py

import pandas as pd
import numpy as np

def complete_data(input_csv, start_time="2020/11/09 00:00:00", end_time="2025/09/30 20:00:00"):
    """
    处理数据，补充缺失的时间点,从2020/11/09 00:00:00到2025/09/30 20:00:00，按4小时间隔补全时间点

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
        处理后的数据
    """
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