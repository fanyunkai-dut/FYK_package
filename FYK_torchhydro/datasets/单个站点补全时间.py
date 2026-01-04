# process_files.py

import os
import yaml
from data_utils import complete_data

def process_files_from_config(config_path, output_dir, start_time="2020/11/09 00:00:00", end_time="2025/09/30 20:00:00"):
    """
    处理配置文件中的所有文件，并将结果保存到新的文件夹
    
    Parameters
    ----------
    config_path : str
        配置文件路径
    output_dir : str
        处理后输出的文件夹路径
    start_time : str
        开始时间
    end_time : str
        结束时间
    """
    
    # 1️⃣ 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 获取数据路径和文件名列表
    data_path = config["data"]["path"]
    files = config["data"]["files"]
    
    # 如果 files 为空，从目录中自动读取所有 CSV 文件
    if not files:
        files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

    # 创建输出目录（如果不存在的话）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2️⃣ 处理每个文件
    for file_name in files:
        input_csv = os.path.join(data_path, file_name)
        
        # 检查文件是否存在
        if not os.path.isfile(input_csv):
            print(f"文件 {input_csv} 不存在，跳过处理。")
            continue
        
        # 调用工具函数进行数据处理
        processed_df = complete_data(input_csv, start_time, end_time)

        # 生成新的CSV文件，保存到新的文件夹
        output_file = os.path.join(output_dir, file_name)  # 文件名不变
        processed_df.to_csv(output_file)

        print(f"处理完毕，输出文件为：{output_file}")

    print("所有文件处理完毕！")
    return "处理完所有文件！"

# 调用示例
config_path = '/home/fanyunkai/FYK_package/FYK_torchhydro/configs/example_config.yaml'  # 配置文件路径
output_dir = '/home/fanyunkai/FYK_data/complete water quality/'  # 输出文件夹路径
process_files_from_config(config_path, output_dir)