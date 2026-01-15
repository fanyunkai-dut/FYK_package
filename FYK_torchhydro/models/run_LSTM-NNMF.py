import numpy as np
import tensorflow as tf
import time
import os
import sys
import traceback  # 添加这个模块来追踪错误

# 禁用GPU，使用CPU（避免GPU相关问题）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 减少TensorFlow日志输出

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def load_preprocessed_data(data_dir):
    """
    加载预处理好的数据
    
    参数:
        data_dir: 数据目录路径
    
    返回:
        加载的所有数据
    """
    print("=" * 60)
    print("加载预处理数据")
    print("=" * 60)
    
    try:
        # 加载所有预处理文件
        sparse_mat = np.load(os.path.join(data_dir, 'sparse_mat.npy'))
        dense_mat_filled = np.load(os.path.join(data_dir, 'dense_mat_filled.npy'))
        train_mask = np.load(os.path.join(data_dir, 'train_mask.npy'))
        val_mask = np.load(os.path.join(data_dir, 'val_mask.npy'))
        original_nan_mask = np.load(os.path.join(data_dir, 'original_nan_mask.npy'))
        
        print(f"稀疏矩阵形状: {sparse_mat.shape}, 数据类型: {sparse_mat.dtype}")
        print(f"密集矩阵形状: {dense_mat_filled.shape}, 数据类型: {dense_mat_filled.dtype}")
        print(f"训练掩码形状: {train_mask.shape}, True值数量: {np.sum(train_mask)}")
        print(f"验证掩码形状: {val_mask.shape}, True值数量: {np.sum(val_mask)}")
        print(f"原始NaN掩码形状: {original_nan_mask.shape}, True值数量: {np.sum(original_nan_mask)}")
        
        # 检查数据范围
        print(f"稀疏矩阵值范围: [{np.nanmin(sparse_mat):.4f}, {np.nanmax(sparse_mat):.4f}]")
        print(f"密集矩阵值范围: [{np.nanmin(dense_mat_filled):.4f}, {np.nanmax(dense_mat_filled):.4f}]")
        
        return sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        traceback.print_exc()
        return None, None, None, None, None

def normalize_data(data, mask=None):
    """
    标准化数据
    
    参数:
        data: 原始数据
        mask: 掩码，如果提供，只使用掩码为True的位置计算统计量
    
    返回:
        normalized_data: 标准化后的数据
        mean: 均值
        std: 标准差
    """
    print("执行数据标准化...")
    
    try:
        if mask is not None:
            # 只使用掩码为True的位置计算统计量
            valid_data = data[mask]
            print(f"使用掩码，有效数据数量: {len(valid_data)}")
        else:
            # 过滤掉NaN和0值（人为缺失）
            valid_mask = ~np.isnan(data) & (data != 0)
            valid_data = data[valid_mask]
            print(f"不使用掩码，有效数据数量: {len(valid_data)}")
        
        if len(valid_data) == 0:
            print("警告: 没有有效数据用于标准化，使用全零数据")
            return np.zeros_like(data), 0.0, 1.0
        
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        
        if std < 1e-8:  # 避免除零
            std = 1.0
            print(f"警告: 标准差太小 ({std:.6f})，设为1.0")
        
        normalized_data = (data - mean) / std
        
        print(f"数据标准化完成: 均值={mean:.4f}, 标准差={std:.4f}")
        print(f"标准化后值范围: [{np.min(normalized_data):.4f}, {np.max(normalized_data):.4f}]")
        
        return normalized_data, mean, std
        
    except Exception as e:
        print(f"数据标准化时出错: {e}")
        traceback.print_exc()
        return data, 0.0, 1.0  # 返回原始数据

def main():
    """主函数：运行LSTMNNMF模型"""
    
    print("=" * 60)
    print("LSTMNNMF模型 - 水质预测 - 调试版本")
    print("=" * 60)
    
    try:
        # ==================== 参数设置 ====================
        data_dir = '/home/fanyunkai/FYK_data/WQ_hanjiang/'  # 预处理数据目录
        save_dir = '/home/fanyunkai/FYK_data/WQ_hanjiang_results/'  # 结果保存目录
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 模型参数
        rank = 60  # 从60减小到10，因为数据维度较小（6个站点）
        time_lags = np.array([1, 2, 3])  # 先使用小的时间滞后进行测试
        test_len = 100  # 使用非常小的测试集长度进行测试
        
        # 正则化参数（进一步减小）
        lambda_w = 100  # 进一步减小
        lambda_x = 100  # 进一步减小
        eta = 0.1  # 进一步减小
        
        # 训练参数
        epochs = 500  # 先试10轮
        learning_rate = 0.001  # 减小学习率
        log_interval = 1  # 每个epoch都打印
        
        # 初始化参数
        latent_normal_init_params = {'mean': 0.0, 'stddev': 0.01}  # 使用较小的初始化标准差
        
        # ==================== 加载数据 ====================
        print("\n1. 加载数据...")
        sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask = \
            load_preprocessed_data(data_dir)
        
        if sparse_mat is None:
            print("数据加载失败，退出程序")
            return
        
        # ==================== 数据标准化 ====================
        print("\n2. 数据标准化...")
        normalized_data, data_mean, data_std = normalize_data(dense_mat_filled, train_mask)
        
        # 更新数据
        dense_mat_filled = normalized_data
        sparse_mat = normalized_data.copy()
        sparse_mat[val_mask] = 0  # 保持验证位置为0
        
        # ==================== 创建邻接矩阵 ====================
        print("\n3. 加载邻接矩阵...")
        adj_path = os.path.join(data_dir, 'adj.npy')
        A = np.load(adj_path).astype(np.float32)
        
        # ==================== 按时间分割数据 ====================
        print("\n4. 按时间分割数据...")
        num_times = sparse_mat.shape[1]
        train_len = num_times - test_len
        
        if train_len <= 0:
            print(f"错误: 训练集长度 {train_len} <= 0，请减小 test_len")
            return
        
        print(f"总时间点数: {num_times}")
        print(f"训练集长度: {train_len}")
        print(f"测试集长度: {test_len}")
        
        # 分割稀疏矩阵
        training_set = sparse_mat[:, :train_len]
        test_set = sparse_mat[:, train_len:]
        
        # 分割密集矩阵
        training_ground_truth = dense_mat_filled[:, :train_len]
        test_ground_truth = dense_mat_filled[:, train_len:]
        
        # 分割掩码
        train_mask_train = train_mask[:, :train_len]
        val_mask_train = val_mask[:, :train_len]
        
        print(f"训练集形状: {training_set.shape}")
        print(f"测试集形状: {test_set.shape}")
        print(f"训练掩码形状: {train_mask_train.shape}, True值数量: {np.sum(train_mask_train)}")
        print(f"验证掩码形状: {val_mask_train.shape}, True值数量: {np.sum(val_mask_train)}")
        
        # ==================== 导入模型类 ====================
        print("\n5. 导入LSTMNNMF模型类...")
        try:
            from LSTMNNMF import LSTMNNMF
            print("模型类导入成功")
        except ImportError as e:
            print(f"导入模型类时出错: {e}")
            traceback.print_exc()
            return
        
        # ==================== 创建模型 ====================
        print("\n6. 创建LSTMNNMF模型...")
        try:
            model = LSTMNNMF(
                training_set=training_set,
                training_ground_truth=training_ground_truth,
                train_mask=train_mask_train,
                val_mask=val_mask_train,
                A=A,
                rank=rank,
                time_lags=time_lags,
                lambda_w=lambda_w,
                lambda_x=lambda_x,
                eta=eta,
                latent_normal_init_params=latent_normal_init_params
            )
            print("模型创建成功!")
            print(f"传感器数量: {model.num_sensors}")
            print(f"训练时间点数: {model.num_times}")
            print(f"训练位置数量: {np.sum(train_mask_train)}")
            print(f"验证位置数量: {np.sum(val_mask_train)}")
        except Exception as e:
            print(f"创建模型时出错: {e}")
            traceback.print_exc()
            return
        
        # ==================== 创建优化器 ====================
        print("\n7. 创建优化器...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=0.5  # 梯度裁剪
        )
        
        # ==================== 测试单次前向传播 ====================
        print("\n8. 测试单次前向传播...")
        try:
            # 测试重建
            reconstructed = model.call()
            print(f"重建矩阵形状: {reconstructed.shape}")
            
            # 测试损失计算
            loss, train_residual_error, W_F_norm, X_F_norm, X_norm, W_norm = model.loss_cal()
            print(f"初始损失: {loss.numpy():.4f}")
            print(f"重建误差: {train_residual_error.numpy():.4f}")
            
            # 测试指标计算
            val_mape, val_rmse = model.metrics_cal()
            print(f"初始验证MAPE: {val_mape.numpy():.4f}%")
            print(f"初始验证RMSE: {val_rmse.numpy():.4f}")
        except Exception as e:
            print(f"前向传播测试时出错: {e}")
            traceback.print_exc()
            return
        
        # ==================== 训练模型 ====================
        print("\n9. 开始训练模型...")
        print("=" * 60)
        
        # 简化训练循环
        train_loss_history = []
        
        for epoch in range(epochs):
            try:
                with tf.GradientTape() as tape:
                    loss, train_residual_error, W_F_norm, X_F_norm, X_norm, W_norm = model.loss_cal()
                
                if tf.math.is_nan(loss):
                    print(f"Epoch {epoch+1}: 损失为NaN，跳过")
                    continue
                
                gradients = tape.gradient(loss, model.trainable_variables)
                
                # 检查梯度
                grad_valid = True
                for i, g in enumerate(gradients):
                    if g is None:
                        print(f"Epoch {epoch+1}: 梯度 {i} 为None")
                        grad_valid = False
                    elif tf.reduce_any(tf.math.is_nan(g)):
                        print(f"Epoch {epoch+1}: 梯度 {i} 包含NaN")
                        grad_valid = False
                
                if not grad_valid:
                    print(f"Epoch {epoch+1}: 梯度无效，跳过")
                    continue
                
                # 应用梯度
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # 记录损失
                train_loss_history.append(loss.numpy())
                
                # 打印进度
                if (epoch + 1) % log_interval == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}/{epochs}: 损失 = {loss.numpy():.4f}")
                    
            except Exception as e:
                print(f"Epoch {epoch+1} 训练时出错: {e}")
                traceback.print_exc()
                continue
        
        print("\n训练完成!")
        print(f"最终损失: {train_loss_history[-1] if train_loss_history else 'N/A':.4f}")
        
        # ==================== 保存结果 ====================
        print("\n10. 保存结果...")
        try:
            # 获取重建矩阵
            reconstructed = model.get_reconstructed_matrix()
            np.save(os.path.join(save_dir, 'reconstructed_matrix.npy'), reconstructed)
            print(f"重建矩阵已保存，形状: {reconstructed.shape}")
            
            # 获取潜在变量
            W, X = model.get_latent_variables()
            np.save(os.path.join(save_dir, 'W_matrix.npy'), W)
            np.save(os.path.join(save_dir, 'X_matrix.npy'), X)
            print(f"W矩阵形状: {W.shape}, X矩阵形状: {X.shape}")
            
            # 保存损失历史
            np.save(os.path.join(save_dir, 'loss_history.npy'), np.array(train_loss_history))
            print(f"损失历史已保存，长度: {len(train_loss_history)}")
            
            # 绘制损失曲线
            if len(train_loss_history) > 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(train_loss_history, label='训练损失')
                plt.xlabel('Epoch')
                plt.ylabel('损失')
                plt.title('训练损失曲线')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"损失曲线图已保存")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("LSTMNNMF模型运行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"主程序运行时出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()