import os
# 禁用GPU，必须放在 import tensorflow 之前更稳
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import tensorflow as tf

from FYK_Transformer_NNMF_hanjiang.models.TransNNMF import LSTMNNMF


def load_preprocessed_data(data_dir):
    sparse_mat = np.load(os.path.join(data_dir, 'sparse_mat.npy'))
    dense_mat_filled = np.load(os.path.join(data_dir, 'dense_mat_filled.npy'))
    train_mask = np.load(os.path.join(data_dir, 'train_mask.npy'))
    val_mask = np.load(os.path.join(data_dir, 'val_mask.npy'))
    original_nan_mask = np.load(os.path.join(data_dir, 'original_nan_mask.npy'))
    return sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask


def split_data_by_time(sparse_mat, dense_mat_filled, train_mask, val_mask, test_len):
    num_times = sparse_mat.shape[1]
    train_len = num_times - test_len

    print(f"\n按时间分割数据:")
    print(f"总时间点数: {num_times}")
    print(f"训练集长度: {train_len}")
    print(f"测试集长度: {test_len}")

    training_set = sparse_mat[:, :train_len]
    test_set = sparse_mat[:, train_len:]

    training_ground_truth = dense_mat_filled[:, :train_len]
    test_ground_truth = dense_mat_filled[:, train_len:]

    train_mask_train = train_mask[:, :train_len]
    val_mask_train = val_mask[:, :train_len]

    return (training_set, training_ground_truth, train_mask_train, val_mask_train,
            test_set, test_ground_truth)


def train_model(model, optimizer, epochs=400, log_interval=50):
    train_loss_history = []
    train_rmse_history = []

    t0 = time.time()

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss, train_residual_error, W_F_norm, X_F_norm, X_time, W_graph = model.loss_cal()

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 训练RMSE（基于训练mask上的残差）
        denom = tf.cast(tf.size(model.train_ground_truth_vec), tf.float32)
        train_rmse = tf.sqrt(train_residual_error / denom)

        val_mape, val_rmse = model.metrics_cal()

        train_loss_history.append(float(loss.numpy()))
        train_rmse_history.append(float(train_rmse.numpy()))

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            dt = time.time() - t0
            print(f"Epoch {epoch+1}/{epochs}: time={dt:.2f}s")
            print(f"  loss={loss.numpy():.2f} | train_RMSE={train_rmse.numpy():.4f} | val_MAPE={val_mape.numpy():.4f} | val_RMSE={val_rmse.numpy():.4f}")
            print(f"  parts: recon={train_residual_error.numpy():.2f}, W_L2={W_F_norm.numpy():.2f}, X_L2={X_F_norm.numpy():.2f}, X_time={X_time.numpy():.2f}, W_graph={W_graph.numpy():.2f}")
            t0 = time.time()

    return train_loss_history, train_rmse_history


def save_results(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("保存模型结果")
    print("=" * 60)

    reconstructed = model.get_reconstructed_matrix()
    np.save(os.path.join(save_dir, 'reconstructed_matrix.npy'), reconstructed)
    print(f"重建矩阵已保存，形状: {reconstructed.shape}")
    print(f"重建矩阵值范围: [{np.min(reconstructed):.4f}, {np.max(reconstructed):.4f}]")

    train_reconstructed = reconstructed[model.train_pos_mask.numpy()]
    print(f"训练位置重建值范围: [{np.min(train_reconstructed):.4f}, {np.max(train_reconstructed):.4f}]")

    W, X = model.get_latent_variables()
    np.save(os.path.join(save_dir, 'W_matrix.npy'), W)
    np.save(os.path.join(save_dir, 'X_matrix.npy'), X)
    print(f"W矩阵形状: {W.shape}, X矩阵形状: {X.shape}")

    val_mape, val_rmse = model.metrics_cal()
    print(f"\n最终验证性能:")
    print(f"验证MAPE: {val_mape.numpy():.4f}")
    print(f"验证RMSE: {val_rmse.numpy():.4f}")


def main():
    print("=" * 60)
    print("NNMF + Transformer时间正则（替代LSTM）")
    print("=" * 60)

    # ==================== 参数设置 ====================
    data_dir = '/home/fanyunkai/FYK_data/WQ_hanjiang/'
    save_dir = '/home/fanyunkai/FYK_data/WQ_hanjiang_results/result_transformer/'
    os.makedirs(save_dir, exist_ok=True)

    rank = 60

    # ⚠️ 你原来是 1..99，Transformer注意力是 O(L^2)，CPU会慢；
    # 建议先试 1..24 或 1..48
    time_lags = np.array(range(1, 48))  # 先简单点，跑得快

    test_len = 2055

    lambda_w = 100
    lambda_x = 5
    eta = 0.1

    epochs = 1000
    learning_rate = 0.001
    log_interval = 50

    latent_normal_init_params = {'mean': 0.0, 'stddev': 0.1}

    # Transformer 超参（先简单）
    n_heads = 6
    dropout = 0.1

    # ==================== 加载数据 ====================
    print("\n1. 加载数据...")
    sparse_mat, dense_mat_filled, train_mask, val_mask, original_nan_mask = load_preprocessed_data(data_dir)

    # ==================== 加载邻接矩阵 ====================
    print("\n2. 加载邻接矩阵...")
    adj_path = os.path.join(data_dir, 'adj.npy')
    A = np.load(adj_path).astype(np.float32)
    print(f"邻接矩阵形状: {A.shape}")

    # ==================== 按时间分割 ====================
    (training_set, training_ground_truth, train_mask_train, val_mask_train,
     test_set, test_ground_truth) = split_data_by_time(
        sparse_mat, dense_mat_filled, train_mask, val_mask, test_len
    )

    # ==================== 创建模型 ====================
    print("\n3. 创建模型...")
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
        latent_normal_init_params=latent_normal_init_params,
        n_heads=n_heads,
        dropout=dropout,
        debug=False
    )

    # ==================== 优化器 ====================
    print("\n4. 创建优化器...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # ==================== 初始测试 ====================
    print("\n5. 初始测试...")
    reconstructed = model.call()
    print(f"初始重建矩阵形状: {reconstructed.shape}")
    print(f"初始重建矩阵值范围: [{np.min(reconstructed.numpy()):.4f}, {np.max(reconstructed.numpy()):.4f}]")

    loss, train_residual_error, W_F_norm, X_F_norm, X_time, W_graph = model.loss_cal()
    print(f"初始loss: {loss.numpy():.2f} | recon: {train_residual_error.numpy():.2f} | X_time: {X_time.numpy():.2f}")

    # ==================== 训练 ====================
    print("\n6. 开始训练...")
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
        plt.plot(train_loss_history, label='train_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"损失曲线图已保存到: {os.path.join(save_dir, 'loss_curve.png')}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
