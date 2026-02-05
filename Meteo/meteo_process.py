import numpy as np
import pandas as pd

def process_and_concat_weather_data(precip_csv, temp_csv, humidity_csv, windspeed_csv, lwd_csv, swd_csv, pressure_csv, output_npy_path):
    # 1. 读取每个气象数据的CSV，并检查数据是否为数值，若有问题则抛出异常并打印出错误行
    def read_and_check_csv(csv_file, column_name):
        df = pd.read_csv(csv_file, header=0)  # header=0 告诉 pandas 第一行是列名
        df.columns = ['time', column_name]  # 重命名第二列为传入的列名
        print(f"Columns in {csv_file}: {df.columns}")  # 打印列名
        try:
            # 确保数值列都可以被转换为数字，若不能转换就抛出异常
            df[column_name] = pd.to_numeric(df[column_name], errors='raise')  # 强制转换为数值，无法转换的值会变成 NaN
        except ValueError as e:
            print(f"Error in file: {csv_file}")
            print(f"Error message: {e}")
            # 输出有问题的数据行
            print(f"Problematic rows: {df[df[column_name].apply(lambda x: isinstance(x, str))]}")
            raise e  # 抛出异常中止程序
        return df
    
    # 读取每个气象数据的CSV并检查，并为每个 CSV 文件指定不同的列名
    precipitation_df = read_and_check_csv(precip_csv, 'precipitation')
    temperature_df = read_and_check_csv(temp_csv, 'temperature')
    humidity_df = read_and_check_csv(humidity_csv, 'temperature')
    windspeed_df = read_and_check_csv(windspeed_csv, 'temperature')
    lwd_df = read_and_check_csv(lwd_csv, 'temperature')
    swd_df = read_and_check_csv(swd_csv, 'temperature')
    pressure_df = read_and_check_csv(pressure_csv, 'temperature')

    # 2. 对降雨数据做对数变换 + 标准化
    epsilon = 1e-6  # 防止 log(0) 的问题
    precipitation_log = np.log1p(precipitation_df['precipitation'])  # 对数变换
    precipitation_mean = np.nanmean(precipitation_log)  # 计算均值时跳过 NaN
    precipitation_std = np.nanstd(precipitation_log)  # 计算标准差时跳过 NaN
    precipitation_normalized = (precipitation_log - precipitation_mean) / precipitation_std  # 标准化
    
    # 3. 其他气象数据（温度、湿度、风速、辐射、压强）做标准化
    def standardize(data):
        return (data - np.nanmean(data)) / np.nanstd(data)  # 计算时跳过 NaN
    
    temperature_normalized = standardize(temperature_df['temperature'])
    humidity_normalized = standardize(humidity_df['temperature'])
    windspeed_normalized = standardize(windspeed_df['temperature'])
    lwd_normalized = standardize(lwd_df['temperature'])
    swd_normalized = standardize(swd_df['temperature'])
    pressure_normalized = standardize(pressure_df['temperature'])
    
    # 4. 拼接处理后的气象数据
    # 这里每个气象特征的处理结果是一个 11274 长度的数组
    weather_data = np.stack([
        precipitation_normalized,
        temperature_normalized,
        humidity_normalized,
        windspeed_normalized,
        lwd_normalized,
        swd_normalized,
        pressure_normalized
    ], axis=0)  # shape: (7, 11274)
    
    # 5. 转置为 7*11274 的矩阵（包含降雨、温度、湿度、风速、辐射、压强）
    weather_data_transposed = weather_data.T  # shape: (11274, 7)
    
    # 6. 保存为 npy 文件
    np.save(output_npy_path, weather_data_transposed)
    print(f"气象数据已处理并保存为：{output_npy_path}")

# 使用示例
precip_csv = '/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_precipitation_4hours.csv'
temp_csv = '/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_temperature_4hours.csv'
humidity_csv = '/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_rh_4hours.csv'
windspeed_csv = '/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_wind_4hours.csv'
lwd_csv = '/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_lwd_4hours.csv'
swd_csv = '/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_swd_4hours.csv'
pressure_csv = '/home/fanyunkai/FYK_data/QX_hanjiang/hanjiang_pressure_4hours.csv'
output_npy_path = '/home/fanyunkai/FYK_data/QX_hanjiang/weather_data.npy'

process_and_concat_weather_data(precip_csv, temp_csv, humidity_csv, windspeed_csv, lwd_csv, swd_csv, pressure_csv, output_npy_path)




