import os
import yaml
import pandas as pd

# 1️⃣ 读取配置文件路径（相对 FYK_PACKAGE 的结构）
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "example_config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2️⃣ 获取配置中的路径和需要处理的指标
data_path = config["data"]["path"]
target_variables = config["data"]["target_variables"]

# 3️⃣ 自动读取所有 CSV 文件
file_list = [f for f in os.listdir(data_path) if f.endswith('.csv')]

# 4️⃣ 处理每个 CSV 文件
for f in file_list:
    file_path = os.path.join(data_path, f)
    df = pd.read_csv(file_path)

    # 5️⃣ 检查并提取总氮、总磷列
    for var in target_variables:
        if var in df.columns:
            print(f"在文件 {f} 中找到 {var} 列：")
            print(df[var].head())  # 输出总氮或总磷的前五行
        else:
            print(f"在文件 {f} 中未找到 {var} 列")