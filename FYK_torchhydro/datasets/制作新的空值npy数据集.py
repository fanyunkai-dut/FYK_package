#第五步，把原有的binary矩阵再mask20%的空值。
import numpy as np
import yaml

def apply_dropout_to_npy(config_path):
    """
    在二进制 .npy 文件中应用 dropout，随机将部分 '1' 值变为 '0'，根据配置文件中的 dropout_rate。
    """
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 获取 dropout_rate 参数
    dropout_rate = config["WQ_data"]["dropout_rate"]  # 遗忘率，指定随机变为 0 的比例
    input_npy_path = config["WQ_data"]["input_file2"]  # 原始的二进制 .npy 文件路径
    random_seed = config["WQ_data"]["random_seed"]

    # 加载原始二进制 .npy 数据
    try:
        data = np.load(input_npy_path)
    except Exception as e:
        print(f"[错误] 加载 .npy 文件失败: {e}")
        return

    # 获取数据的形状
    total_elements = data.size

    # 设置随机种子，确保结果可复现
    np.random.seed(random_seed)  # 可根据需要修改或通过配置文件传递随机种子

    # 找到所有值为 1 的位置
    ones_positions = np.where(data == 1)

    # 计算需要变为 0 的数量
    num_dropout = int(len(ones_positions[0]) * dropout_rate)

    # 随机选择位置
    dropout_indices = np.random.choice(len(ones_positions[0]), num_dropout, replace=False)

    # 将选中的位置的 1 设置为 0
    for idx in dropout_indices:
        data[ones_positions[0][idx], ones_positions[1][idx]] = 0

    # 生成新的输出路径，文件名加上 "_dropped" 后缀
    output_npy_path = input_npy_path.replace(".npy", "_dropped.npy")

    # 保存新的数据到新的 .npy 文件
    np.save(output_npy_path, data)

    print(f"[完成] 带有 dropout 的 .npy 文件已保存到: {output_npy_path}")


# 调用示例
config_path = '/home/fanyunkai/FYK_package/FYK_torchhydro/configs/example_config.yaml'  # 配置文件路径
apply_dropout_to_npy(config_path)