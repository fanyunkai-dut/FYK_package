#第三步，把多个csv里的总氮列拼接成npy文件，叫做WQ_hanjiang.npy
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
    data_path1 = config["dataset"]["output_path1"]
    files = ["云池(白洋).csv", "观音寺.csv", "柳口.csv", "调关.csv", "荆江口.csv", "旁海.csv", "浦市上游.csv",
              "侯家淇.csv", "五强溪.csv", "陈家河(四水厂).csv", "归阳镇.csv", "城北水厂.csv", "熬洲.csv",
                "霞湾.csv", "昭山.csv", "桔子洲.csv", "樟树港.csv", "岳阳楼.csv", "梁西渡.csv", "南柳渡.csv",
                  "黄金峡.csv", "小钢桥.csv", "老君关.csv", "羊尾.csv", "沈湾.csv", "白家湾.csv",
                    "余家湖.csv", "转斗.csv", "皇庄.csv", "罗汉闸.csv","岳口.csv", "宗关.csv", "城陵矶.csv",
                     "杨泗港.csv", "白浒山.csv", "燕矶.csv", "中官铺.csv", "姚港(河西水厂).csv", "梓坑.csv",
                      "峡山.csv", "赣县梅林.csv", "新庙前.csv", "通津.csv", "金滩.csv", "大洋洲.csv", "丰城小港口.csv",
                       "生米.csv", "蛤蟆石.csv", "香口(东至龙江水厂).csv", "皖河口.csv", "前江口.csv", "五步沟.csv",
                        "陈家墩.csv", "东西梁山.csv", "三兴村.csv", "小河口上游.csv","魏村.csv", "小湾.csv"]

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
    output_npy_path = os.path.join(data_path1, "WQ_changjiang.npy")

    # 保存为 .npy 文件
    np.save(output_npy_path, concatenated_data)

    print(f"[完成] 数据已保存到 {output_npy_path}")

# 调用示例
config_path = '/home/fanyunkai/FYK_package/FYK_LSTMNNMF_changjiang/configs/example_config.yaml'  # 配置文件路径
concatenate_columns_to_npy(config_path)

