import os
import yaml
from data_utils import complete_data, clean_data


def process_files_from_config(
    config_path,
    start_time="2020/11/09 00:00:00",
    end_time="2025/09/30 20:00:00",
    window_size=100,
    threshold_high=4,
    threshold_low=0.25,
    min_valid_neighbors=50,
    encoding=None,  # 需要时可填 "utf-8-sig"/"gbk"
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config["dataset"]
    input_dir = dataset_cfg["input_path"]
    output_dir = dataset_cfg["output_path"]
    files = dataset_cfg.get("files", [])
    variables = dataset_cfg.get("variables", [])

    if not files:
        files = [fn for fn in os.listdir(input_dir) if fn.lower().endswith(".csv")]

    os.makedirs(output_dir, exist_ok=True)

    for file_name in files:
        input_csv = os.path.join(input_dir, file_name)
        if not os.path.isfile(input_csv):
            print(f"文件不存在，跳过: {input_csv}")
            continue

        # 1) 补全时间
        df_completed = complete_data(input_csv, start_time=start_time, end_time=end_time)

        # ✅ 关键改动：把 index 还原成列，并命名为“监测时间”
        df_completed = df_completed.reset_index().rename(columns={"index": "监测时间"})

        # 2) 临时落盘（给 clean_data 读取）
        tmp_completed_path = os.path.join(output_dir, f"_tmp_completed_{file_name}")
        df_completed.to_csv(tmp_completed_path, index=False, encoding="utf-8-sig")

        # 3) 清洗异常值，输出到 output_dir（文件名不变）
        output_csv = os.path.join(output_dir, file_name)
        clean_data(
            csv_file_path=tmp_completed_path,
            column_name=variables,
            window_size=window_size,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            min_valid_neighbors=min_valid_neighbors,
            output_csv_path=output_csv,
            encoding=encoding,
        )

        # 4) 删除临时文件
        try:
            os.remove(tmp_completed_path)
        except Exception:
            pass

        print(f"处理完毕，输出文件为：{output_csv}")

    print("所有文件处理完毕！")
    return "处理完所有文件！"


# ===== 调用 =====
config_path = "/home/fanyunkai/FYK_package/FYK_torchhydro/configs/example_config.yaml"
process_files_from_config(config_path)