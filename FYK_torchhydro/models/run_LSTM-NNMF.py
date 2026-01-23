import numpy as np
import tensorflow as tf
import time
import os
import sys

# 禁用GPU，使用CPU（避免GPU相关问题）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        # 加载所有预处理文件
    sparse_mat = np.load(os.path.join(data_dir, 'sparse_mat.npy'))
    dense_mat_filled = np.load(os.path.join(data_dir, 'dense_mat_filled.npy'))
    train_mask = np.load(os.path.join(data_dir, 'train_mask.npy'))
    val_mask = np.load(os.path.join(data_dir, 'val_mask.npy'))
    original_nan_mask = np.load(os.path.join(data_dir, 'original_nan_mask.npy'))
    return sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask

def split_data_by_time(sparse_mat, dense_mat_filled, train_mask, val_mask, test_len):
    """
    按时间维度分割数据
    """
    num_times = sparse_mat.shape[1]
    train_len = num_times - test_len
    
    print(f"\n按时间分割数据:")
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

    
    return (training_set, training_ground_truth, train_mask_train, val_mask_train,
            test_set, test_ground_truth)

def train_model(model, optimizer, epochs=500, log_interval=50):
    """
    训练模型
    """
    
    train_loss_history = []
    train_rmse_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss, train_residual_error, W_F_norm, X_F_norm, X_norm, W_norm = model.loss_cal()
            
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # 应用梯度
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # 计算训练RMSE
            train_rmse = tf.sqrt(train_residual_error / tf.cast(tf.size(model.train_ground_truth_vec), tf.float32))
            
            # 计算验证指标
            val_mape, val_rmse = model.metrics_cal()
            
            # 记录历史
            train_loss_history.append(loss.numpy())
            train_rmse_history.append(train_rmse.numpy())
            
            # 打印日志
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  时间: {epoch_time:.2f}s, 总损失: {loss.numpy():.2f}")
                print(f"  训练RMSE: {train_rmse.numpy():.4f}, 验证MAPE: {val_mape.numpy():.4f}, 验证RMSE: {val_rmse.numpy():.4f}")
                print(f"  损失分量 - 重建误差: {train_residual_error.numpy():.2f}, "
                      f"W正则化: {W_F_norm.numpy():.2f}, X正则化: {X_F_norm.numpy():.2f}")
                start_time = time.time()
    
    return train_loss_history, train_rmse_history

def save_results(model, save_dir):
    """
    保存模型结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("保存模型结果")
    print("=" * 60)
    
    # 获取重建矩阵
    reconstructed = model.get_reconstructed_matrix()
    np.save(os.path.join(save_dir, 'reconstructed_matrix.npy'), reconstructed)
    print(f"重建矩阵已保存，形状: {reconstructed.shape}")
    
    # 检查重建矩阵的范围
    print(f"重建矩阵值范围: [{np.min(reconstructed):.4f}, {np.max(reconstructed):.4f}]")
    print(f"重建矩阵均值: {np.mean(reconstructed):.4f}")
    
    # 检查训练位置的重建值
    train_reconstructed = reconstructed[model.train_pos_mask.numpy()]
    print(f"训练位置重建值范围: [{np.min(train_reconstructed):.4f}, {np.max(train_reconstructed):.4f}]")
    
    # 获取潜在变量
    W, X = model.get_latent_variables()
    np.save(os.path.join(save_dir, 'W_matrix.npy'), W)
    np.save(os.path.join(save_dir, 'X_matrix.npy'), X)
    print(f"W矩阵形状: {W.shape}, X矩阵形状: {X.shape}")
    
    # 计算最终性能
    train_mape, train_rmse = model.metrics_cal()
    print(f"\n最终验证性能:")
    print(f"验证MAPE: {train_mape.numpy():.4f}%")
    print(f"验证RMSE: {train_rmse.numpy():.4f}")

def main():
    """主函数：运行LSTMNNMF模型"""
    
    print("=" * 60)
    print("LSTMNNMF模型 - 水质预测")
    print("=" * 60)
    
    try:
        # ==================== 参数设置 ====================
        data_dir = '/home/fanyunkai/FYK_data/WQ_hanjiang/'
        save_dir = '/home/fanyunkai/FYK_data/WQ_hanjiang_results/'
        os.makedirs(save_dir, exist_ok=True)
        
        # 模型参数 - 使用原交通数据的参数
        rank = 60
        time_lags = np.array([1, 2, 3, 4, 182, 2192])
        test_len = 2055  # 10274*0.2
        
        # 正则化参数 - 使用原交通数据的参数
        lambda_w = 100
        lambda_x = 10
        eta = 0.1
        
        # 训练参数 - 使用原交通数据的参数
        epochs = 400
        learning_rate = 0.001
        log_interval = 50
        
        # 初始化参数 - 使用原交通数据的参数
        latent_normal_init_params = {'mean': 0.0, 'stddev': 0.1}
        
        # ==================== 加载数据 ====================
        print("\n1. 加载数据...")
        sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask = \
            load_preprocessed_data(data_dir)
        
        if sparse_mat is None:
            return
        
        # ==================== 重要：不进行标准化 ====================
        print("\n2. 使用原始数据（不进行标准化）")
        print("注意：保持原始数据范围，与交通数据代码保持一致")
        
        # 重要：直接使用原始数据，不进行标准化
        # sparse_mat 已经是正确的（验证位置为0）
        
        # ==================== 加载邻接矩阵 ====================
        print("\n3. 加载邻接矩阵...")
        adj_path = os.path.join(data_dir, 'adj.npy')
        A = np.load(adj_path).astype(np.float32)
        print(f"邻接矩阵形状: {A.shape}")
        
        # ==================== 按时间分割数据 ====================
        (training_set, training_ground_truth, train_mask_train, val_mask_train,
         test_set, test_ground_truth) = split_data_by_time(
             sparse_mat, dense_mat_filled, train_mask, val_mask, test_len
         )
        
        # ==================== 导入并创建模型 ====================
        print("\n4. 创建LSTMNNMF模型...")
        try:
            from LSTMNNMF import LSTMNNMF
        except ImportError as e:
            print(f"导入模型类时出错: {e}")
            return
        
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
        
        print(f"模型创建成功!")
        print(f"传感器数量: {model.num_sensors}")
        print(f"训练时间点数: {model.num_times}")
        
        # ==================== 创建优化器 ====================
        print("\n5. 创建优化器...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # ==================== 初始测试 ====================
        print("\n6. 初始测试...")
        reconstructed = model.call()
        print(f"初始重建矩阵形状: {reconstructed.shape}")
        print(f"初始重建矩阵值范围: [{np.min(reconstructed.numpy()):.4f}, {np.max(reconstructed.numpy()):.4f}]")
        
        loss, train_residual_error, W_F_norm, X_F_norm, X_norm, W_norm = model.loss_cal()
        print(f"初始损失: {loss.numpy():.2f}")
        print(f"初始重建误差: {train_residual_error.numpy():.2f}")
        
        # ==================== 训练模型 ====================
        train_loss_history, train_rmse_history = train_model(
            model=model,
            optimizer=optimizer,
            epochs=epochs,
            log_interval=log_interval
        )
        
        # ==================== 保存结果 ====================
        save_results(model, save_dir)
        
        # ==================== 可视化损失曲线 ====================
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
            print(f"损失曲线图已保存到: {os.path.join(save_dir, 'loss_curve.png')}")
        
        print("\n" + "=" * 60)
        print("LSTMNNMF模型训练完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"主程序运行时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()