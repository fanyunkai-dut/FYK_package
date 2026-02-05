import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate

# 设置文件夹路径
folder_path = "/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_swd"
output_file = "/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_swd_4hours.csv"

# 获取所有.nc文件
nc_files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]

# 时间和温度的列表
times = []
temperature_values = []

# 按文件排序，确保顺序正确
nc_files.sort()

# 循环读取所有.nc文件
for file in nc_files:
    # 跳过'hanjiang_'前缀并去掉.nc后缀
    file_name = file.replace('hanjiang_', '').rsplit('.', 1)[0]  # 去除前缀并去掉.nc后缀
    
    # 分割成日期（YYYYDDD）和小时（HH）
    date_str, hour_str = file_name.split(".", 1)  # 分割成YYYYDDD和小时部分
    
    # 获取年份-日序列（YYYYDDD）和小时（HH）
    year_day = date_str  # YYYYDDD
    hour = int(hour_str)  # HH

    # 读取.nc文件
    nc_path = os.path.join(folder_path, file)
    ds = xr.open_dataset(nc_path)
    
    # 假设温度字段名是'downward_shortwave_radiation'
    temperature = ds["downward_shortwave_radiation"].values
    
    # 计算温度的平均值（按经纬度求平均）
    avg_temperature = np.mean(temperature)
    
    # 记录当前时段的温度
    times.append(f"{year_day}.{hour:02d}")
    temperature_values.append(avg_temperature)

# 转换成DataFrame
df = pd.DataFrame({
    "time": times,
    "value": temperature_values
})

# 对每一天的8个数据点进行线性插值
interpolated_times = []
interpolated_temperature = []

for i in range(0, len(df), 8):
    day_data = df.iloc[i:i+8]
    
    if len(day_data) == 8:
        # 每一天的温度
        times_of_day = np.array([0, 3, 6, 9, 12, 15, 18, 21])
        temperature_of_day = day_data["temperature"].values
        
        # 线性插值：我们要插值到0, 4, 8, 12, 16, 20点
        interpolation_func = interpolate.interp1d(times_of_day, temperature_of_day, kind="linear")
        new_times = np.array([0, 4, 8, 12, 16, 20])
        new_temperature = interpolation_func(new_times)
        
        # 记录插值结果，保留一位小数
        for new_time, temp in zip(new_times, new_temperature):
            interpolated_times.append(f"{day_data['time'].iloc[0][0:7]}.{new_time:02d}")
            interpolated_temperature.append(round(temp, 1))  # 保留一位小数

# 创建最终的DataFrame
final_df = pd.DataFrame({
    "time": interpolated_times,
    "temperature": interpolated_temperature
})

# 保存为CSV文件
final_df.to_csv(output_file, index=False, encoding='utf-8')

print("CSV文件已生成。")
