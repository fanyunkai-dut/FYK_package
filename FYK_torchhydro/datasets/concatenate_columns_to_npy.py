#第三步，把多个csv里的总氮列拼接成npy文件
import os
import numpy as np
import pandas as pd
import yaml

def concatenate_columns_to_npy(config_path):
    """
    读取配置文件，拼接多个 CSV 文件的某一列（如“总氮”）到一个 .npy 文件，并转置矩阵。
    """
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 获取文件夹路径和文件名列表
    data_path = config["dataset"]["output_path"]
    files = ["梁西渡.csv", "南柳渡.csv", "黄金峡.csv", "小钢桥.csv", "老君关.csv", "羊尾.csv"]

    # 存储拼接的总氮数据
    concatenated_data = []

    for file_name in files:
        # 构建每个文件的路径
        input_csv = os.path.join(data_path, file_name)

        if not os.path.isfile(input_csv):
            print(f"[跳过] 文件不存在: {file_name}")
            continue

        # 读取 CSV 文件
        df = pd.read_csv(input_csv)

        # 确保“总氮”列存在
        if "总氮" not in df.columns:
            print(f"[跳过] 文件 {file_name} 没有列: 总氮")
            continue

        # 提取“总氮”列并转换为 numpy 数组
        total_nitrogen_column = df["总氮"].values

        # 将这一列添加到拼接数据列表
        concatenated_data.append(total_nitrogen_column)

    # 如果没有有效的“总氮”数据，输出提示并退出
    if not concatenated_data:
        print("[错误] 没有有效的 '总氮' 数据，退出处理。")
        return

    # 拼接所有列，形成一个大的 numpy 数组
    # 按照列顺序拼接
    concatenated_data = np.array(concatenated_data)

    # 输出的 .npy 文件路径
    output_npy_path = os.path.join(data_path, "WQ_hanjiang.npy")

    # 保存为 .npy 文件
    np.save(output_npy_path, concatenated_data)

    print(f"[完成] 数据已保存到 {output_npy_path}")

# 调用示例
config_path = '/home/fanyunkai/FYK_package/FYK_torchhydro/configs/example_config.yaml'  # 配置文件路径
concatenate_columns_to_npy(config_path)

