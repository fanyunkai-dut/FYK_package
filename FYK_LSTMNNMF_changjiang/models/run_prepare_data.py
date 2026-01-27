# run_prepare_data.py
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# 添加prepare_data模块的路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prepare_data import prepare_data, save_results, load_and_inspect_results

def main():
    # 设置路径
    data_path = '/home/fanyunkai/FYK_data/WQ_changjiang/WQ_changjiang.npy'
    output_dir = '/home/fanyunkai/FYK_data/WQ_changjiang/'
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    print("=" * 60)
    print("开始数据预处理")
    print("=" * 60)
    
    # 1. 加载原始数据
    print(f"\n1. 加载原始数据: {data_path}")
    try:
        dense_mat = np.load(data_path)
        print(f"   数据加载成功，形状: {dense_mat.shape}")
        print(f"   数据类型: {dense_mat.dtype}")
    except Exception as e:
        print(f"   错误: 无法加载数据 - {e}")
        return
    
    # 2. 显示原始数据的基本统计
    print(f"\n2. 原始数据统计:")
    print(f"   最小值: {np.nanmin(dense_mat):.4f}")
    print(f"   最大值: {np.nanmax(dense_mat):.4f}")
    print(f"   平均值: {np.nanmean(dense_mat):.4f}")
    print(f"   标准差: {np.nanstd(dense_mat):.4f}")
    print(f"   NaN数量: {np.sum(np.isnan(dense_mat))} ({np.mean(np.isnan(dense_mat))*100:.2f}%)")
    
    # 3. 执行数据预处理
    print(f"\n3. 执行数据预处理...")
    sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask = prepare_data(
        dense_mat, 
        missing_rate=0.2
    )
    
    # 4. 保存结果
    print(f"\n4. 保存结果到: {output_dir}")
    save_results(output_dir, sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask)
    
    # 5. 加载并检查保存的结果
    print(f"\n5. 验证保存的结果...")
    load_and_inspect_results(output_dir)

if __name__ == "__main__":
    main()