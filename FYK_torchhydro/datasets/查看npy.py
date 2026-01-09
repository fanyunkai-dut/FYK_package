#把npy转换成csv以查看内容
import numpy as np
###npy转换成csv
data = np.load("/home/fanyunkai/FYK_data/processed_dataset2.5/WQ_hanjiang.npy")

np.savetxt(
    "/home/fanyunkai/FYK_data/processed_dataset2.5/WQ_hanjiang.csv",
    data,
    delimiter=",",
    fmt="%.6f"
)

print("✅ 已保存为 Metr_ADJ.csv")
