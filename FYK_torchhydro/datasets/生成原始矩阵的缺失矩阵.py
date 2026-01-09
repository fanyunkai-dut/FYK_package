#第四步，把原始缺失文件生成个新的只有1和0的npy，标记哪个位置有缺失值
import numpy as np
import yaml

def generate_binary_npy(config_path):
    """
    根据原始 .npy 文件生成一个新的二进制 .npy 文件，NaN 对应 0，非 NaN 对应 1。
    """
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 获取原始 .npy 文件路径
    input_npy_path = config["WQ_data"]["input_file"]

    # 加载原始 .npy 数据
    try:
        data = np.load(input_npy_path)
    except Exception as e:
        print(f"[错误] 加载 .npy 文件失败: {e}")
        return

    # 创建一个新的二进制 .npy 文件，非 NaN 对应 1，NaN 对应 0
    binary_data = np.where(np.isnan(data), 0, 1)

    # 生成输出文件路径
    output_npy_path = input_npy_path.replace(".npy", "_binary.npy")

    # 保存新的二进制 .npy 文件
    np.save(output_npy_path, binary_data)

    print(f"[完成] 二进制 .npy 文件已保存到 {output_npy_path}")

# 调用示例
config_path = '/home/fanyunkai/FYK_package/FYK_torchhydro/configs/example_config.yaml'  # 配置文件路径
generate_binary_npy(config_path)