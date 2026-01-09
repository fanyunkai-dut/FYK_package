#第六步，制作训练和验证集，叫做。
import numpy as np
import yaml
import os

def process_data(config_path):
    """
    加载原始数据和二进制掩膜，进行掩膜处理，然后根据训练集和测试集的比例切割数据。
    """
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 获取原始 .npy 文件路径和掩膜文件路径
    input_npy_path = config["WQ_data"]["input_file1"]  # 原始数据路径
    binary_npy_path = config["WQ_data"]["input_file3"]  # 二进制掩膜文件路径
    
    # 加载原始数据和二进制掩膜数据
    try:
        data = np.load(input_npy_path)
        binary_mask = np.load(binary_npy_path)
    except Exception as e:
        print(f"[错误] 加载 .npy 文件失败: {e}")
        return

    # 掩膜操作：原始数据与掩膜相乘，NaN 保持 NaN，非 NaN 值与掩膜相乘
    masked_data = np.where(np.isnan(data), np.nan, data * binary_mask)

    # 获取训练集和测试集的比例
    train_ratio = config["WQ_data"]["train_ratio"]
    test_ratio = config["WQ_data"]["test_ratio"]
    
    # 计算数据集的切割位置
    num_cols = masked_data.shape[1]  # 获取列数，即数据的总数（时间点数量）
    train_size = int(train_ratio * num_cols)
    test_size = num_cols - train_size

    # 切割数据：按照列进行切割（横向切割）
    train_data = masked_data[:, :train_size]
    test_data = masked_data[:, train_size:]

    # 输出训练集和测试集的形状
    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")

    # 保存训练集和测试集
    train_output_path = input_npy_path.replace(".npy", "_train.npy")
    test_output_path = input_npy_path.replace(".npy", "_test.npy")

    np.save(train_output_path, train_data)
    np.save(test_output_path, test_data)

    print(f"[完成] 训练集已保存到 {train_output_path}")
    print(f"[完成] 测试集已保存到 {test_output_path}")

# 调用示例
config_path = '/home/fanyunkai/FYK_package/FYK_torchhydro/configs/example_config.yaml'  # 配置文件路径
process_data(config_path)
