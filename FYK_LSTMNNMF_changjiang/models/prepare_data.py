# prepare_data.py
import numpy as np
import os

def prepare_data(dense_mat, missing_rate=0.2):
    """
    准备数据，区分三种情况：
    1. 真实缺失（原始NaN）
    2. 人为缺失（我们设置的0）
    3. 真实观测值
    
    参数:
        dense_mat: 原始数据矩阵，包含NaN值
        missing_rate: 在非NaN位置设置人为缺失的比例
    
    返回:
        sparse_mat: 稀疏矩阵（原始数据，但真实NaN被替换为0，并且人为缺失的位置也被设为0）
        dense_mat_filled: 原始数据，但NaN被替换为0
        train_mask: 训练掩码（既不是原始NaN也不是人为缺失的位置）
        val_mask: 验证掩码（人为缺失的位置）
        original_nan_mask: 原始NaN的位置
    """
    # 记录原始缺失位置
    original_nan_mask = np.isnan(dense_mat)
    print(f"原始数据中NaN的比例: {np.mean(original_nan_mask):.2%}")
    print(f"原始数据中NaN的数量: {np.sum(original_nan_mask)}")
    
    # 将NaN替换为0以便计算（但我们会用掩码排除它们）
    dense_mat_filled = np.nan_to_num(dense_mat, nan=0.0)
    
    # 只在非NaN位置设置人为缺失
    non_nan_positions = np.where(~original_nan_mask)
    num_non_nan = len(non_nan_positions[0])
    
    print(f"非NaN位置数量: {num_non_nan}")
    print(f"非NaN位置占总位置的比例: {num_non_nan / dense_mat.size:.2%}")
    
    # 创建人为缺失掩码
    artificial_missing_mask = np.zeros_like(dense_mat, dtype=bool)
    
    if num_non_nan > 0:
        # 随机选择非NaN位置的20%作为人为缺失
        num_artificial_missing = int(num_non_nan * missing_rate)
        print(f"将设置 {num_artificial_missing} 个人为缺失位置 ({missing_rate*100:.0f}%)")
        
        # 随机选择索引
        indices = np.random.choice(num_non_nan, num_artificial_missing, replace=False)
        
        # 创建人为缺失掩码
        artificial_missing_mask[non_nan_positions[0][indices], non_nan_positions[1][indices]] = True
    else:
        print("警告：没有非NaN位置可用于设置人为缺失")
    
    # 创建稀疏矩阵
    sparse_mat = dense_mat_filled.copy()
    sparse_mat[artificial_missing_mask] = 0
    
    # 创建训练掩码：既不是原始缺失也不是人为缺失的位置
    train_mask = (~original_nan_mask) & (~artificial_missing_mask)
    
    # 创建验证掩码：人为缺失的位置
    val_mask = artificial_missing_mask
    
    # 打印统计信息
    print(f"\n=== 数据统计 ===")
    print(f"总位置数: {dense_mat.size}")
    print(f"原始NaN位置数: {np.sum(original_nan_mask)} ({np.mean(original_nan_mask):.2%})")
    print(f"人为缺失位置数: {np.sum(artificial_missing_mask)} ({np.mean(artificial_missing_mask):.2%})")
    print(f"训练位置数: {np.sum(train_mask)} ({np.mean(train_mask):.2%})")
    print(f"验证位置数: {np.sum(val_mask)} ({np.mean(val_mask):.2%})")
    
    # 验证没有重叠
    overlap = np.sum(train_mask & val_mask)
    if overlap > 0:
        print(f"警告：训练掩码和验证掩码有 {overlap} 个位置重叠！")
    
    # 验证覆盖完整性
    total_covered = np.sum(original_nan_mask) + np.sum(artificial_missing_mask) + np.sum(train_mask)
    if total_covered != dense_mat.size:
        print(f"警告：掩码没有完全覆盖所有位置！总位置数: {dense_mat.size}, 覆盖位置数: {total_covered}")
    
    return sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask


def save_results(output_dir, sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask):
    """保存所有结果到指定目录"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存各个文件
    np.save(os.path.join(output_dir, 'sparse_mat.npy'), sparse_mat)
    np.save(os.path.join(output_dir, 'dense_mat_filled.npy'), dense_mat_filled)
    np.save(os.path.join(output_dir, 'train_mask.npy'), train_mask)
    np.save(os.path.join(output_dir, 'val_mask.npy'), val_mask)
    np.save(os.path.join(output_dir, 'original_nan_mask.npy'), original_nan_mask)
    
    print(f"\n=== 文件保存到 {output_dir} ===")
    print(f"1. sparse_mat.npy - 形状: {sparse_mat.shape}")
    print(f"2. dense_mat_filled.npy - 形状: {dense_mat_filled.shape}")
    print(f"3. train_mask.npy - 形状: {train_mask.shape}")
    print(f"4. val_mask.npy - 形状: {val_mask.shape}")
    print(f"5. original_nan_mask.npy - 形状: {original_nan_mask.shape}")
    
    # 创建统计信息文件
    stats_file = os.path.join(output_dir, 'data_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("=== 数据预处理统计 ===\n")
        f.write(f"原始数据形状: {sparse_mat.shape}\n")
        f.write(f"总位置数: {sparse_mat.size}\n")
        f.write(f"原始NaN比例: {np.mean(original_nan_mask):.2%} ({np.sum(original_nan_mask)}个)\n")
        f.write(f"人为缺失比例: {np.mean(val_mask):.2%} ({np.sum(val_mask)}个)\n")
        f.write(f"训练数据比例: {np.mean(train_mask):.2%} ({np.sum(train_mask)}个)\n")
        f.write(f"验证数据比例: {np.mean(val_mask):.2%} ({np.sum(val_mask)}个)\n")
        
        # 检查稀疏矩阵中的0值分布
        zero_mask = sparse_mat == 0
        total_zeros = np.sum(zero_mask)
        f.write(f"\n稀疏矩阵中0值统计:\n")
        f.write(f"总0值数: {total_zeros} ({total_zeros/sparse_mat.size:.2%})\n")
        f.write(f"其中来自原始NaN: {np.sum(original_nan_mask)}\n")
        f.write(f"其中来自人为缺失: {np.sum(val_mask)}\n")
        f.write(f"实际观测值中的0值: {total_zeros - np.sum(original_nan_mask) - np.sum(val_mask)}\n")
    
    print(f"统计信息已保存到: {stats_file}")


def load_and_inspect_results(output_dir):
    """加载并检查保存的结果"""
    print(f"\n=== 加载并检查保存的结果 ===")
    
    sparse_mat = np.load(os.path.join(output_dir, 'sparse_mat.npy'))
    dense_mat_filled = np.load(os.path.join(output_dir, 'dense_mat_filled.npy'))
    train_mask = np.load(os.path.join(output_dir, 'train_mask.npy'))
    val_mask = np.load(os.path.join(output_dir, 'val_mask.npy'))
    original_nan_mask = np.load(os.path.join(output_dir, 'original_nan_mask.npy'))
    
    # 检查稀疏矩阵
    print(f"稀疏矩阵形状: {sparse_mat.shape}")
    print(f"稀疏矩阵中非零值数量: {np.sum(sparse_mat != 0)}")
    print(f"稀疏矩阵中0值数量: {np.sum(sparse_mat == 0)}")
    
    # 检查训练掩码
    print(f"\n训练掩码:")
    print(f"  True值数量: {np.sum(train_mask)}")
    print(f"  训练位置的值范围: [{np.min(sparse_mat[train_mask]):.4f}, {np.max(sparse_mat[train_mask]):.4f}]")
    
    # 检查验证掩码
    print(f"\n验证掩码:")
    print(f"  True值数量: {np.sum(val_mask)}")
    print(f"  验证位置的真实值范围: [{np.min(dense_mat_filled[val_mask]):.4f}, {np.max(dense_mat_filled[val_mask]):.4f}]")
    
    # 检查原始NaN掩码
    print(f"\n原始NaN掩码:")
    print(f"  True值数量: {np.sum(original_nan_mask)}")
    
    # 检查是否有重叠
    print(f"\n掩码重叠检查:")
    print(f"  训练∩验证: {np.sum(train_mask & val_mask)}")
    print(f"  训练∩原始NaN: {np.sum(train_mask & original_nan_mask)}")
    print(f"  验证∩原始NaN: {np.sum(val_mask & original_nan_mask)}")
    
    return sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask
