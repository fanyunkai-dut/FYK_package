import tensorflow as tf
from tensorflow.python.keras import Model, layers
import numpy as np


class LSTMNNMF(Model):
    """
    NNMF + (Transformer时间正则) + (图拉普拉斯正则) + (L2正则)

    说明：
    - 重建：Y_hat = W @ X^T
    - 时间正则：让 X[t] 接近 Transformer(过去若干滞后窗口)
    - 空间正则：图拉普拉斯平滑 W（可选，取决于你是否传入 A、lambda_w）
    """

    def __init__(
        self,
        training_set,
        training_ground_truth,
        train_mask,
        val_mask,
        A,
        rank,
        time_lags,
        lambda_w=100,
        lambda_x=10,
        eta=0.1,
        latent_normal_init_params={'mean': 0.0, 'stddev': 0.1},
        # Transformer相关
        n_heads=4,
        dropout=0.1,
        debug=False
    ):
        super().__init__()

        self.debug = debug

        # 基本参数
        self.rank = int(rank)
        self.time_lags = tf.constant(time_lags[::-1], dtype=tf.int64)  # 反转后得到 [t-maxlag,...,t-1]
        self.lag_len = int(len(time_lags))
        self.max_lag = int(np.max(time_lags))

        # 掩码
        self.train_pos_mask = tf.constant(train_mask, dtype=tf.bool)
        self.val_pos_mask = tf.constant(val_mask, dtype=tf.bool)

        # 正则化参数
        self.lambda_w = float(lambda_w)
        self.lambda_x = float(lambda_x)
        self.eta = float(eta)

        # 邻接矩阵（用于图拉普拉斯正则）
        self.adj = tf.constant(A, dtype=tf.float32) if A is not None else None

        # 数据
        self.ground_truth_tf = tf.constant(training_ground_truth, dtype=tf.float32)
        self.train_ground_truth_vec = tf.boolean_mask(self.ground_truth_tf, self.train_pos_mask)
        self.val_ground_truth_vec = tf.boolean_mask(self.ground_truth_tf, self.val_pos_mask)

        self.num_sensors = int(training_ground_truth.shape[0])
        self.num_times = int(training_ground_truth.shape[1])

        # 初始化W和X（潜在变量）
        self.W_tf = tf.Variable(
            tf.random.normal(
                shape=[self.num_sensors, self.rank],
                mean=latent_normal_init_params['mean'],
                stddev=latent_normal_init_params['stddev']
            ),
            dtype=tf.float32
        )
        self.X_tf = tf.Variable(
            tf.random.normal(
                shape=[self.num_times, self.rank],
                mean=latent_normal_init_params['mean'],
                stddev=latent_normal_init_params['stddev']
            ),
            dtype=tf.float32
        )

        # ========= Transformer Encoder（单块） =========
        # 说明：输入 [B, L, rank]，输出取最后一步 [B, rank] 作为 X_pred
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)

        # 给 Transformer 加一个“可学习位置嵌入”，不然 self-attn 对顺序不敏感
        self.pos_emb = layers.Embedding(input_dim=self.lag_len, output_dim=self.rank)

        # Multi-Head Attention
        # key_dim * n_heads 一般 ~ rank（这里取 rank//heads，保证>=1）
        key_dim = max(1, self.rank // self.n_heads)
        self.mha = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=key_dim,
            dropout=self.dropout
        )

        # FFN
        d_ff = 4 * self.rank
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(self.rank),
        ])

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(self.dropout)
        self.drop2 = layers.Dropout(self.dropout)
        self.out_proj = layers.Dense(self.rank)

        # 中间结果
        self.recovered_tensor = None
        self.X_predicted = None
        self.loss_val = None

        print(f"模型初始化完成: rank={rank}, max_lag={self.max_lag}, lag_len={self.lag_len}, heads={self.n_heads}")

    def call(self, inputs=None, training=False):
        # 重建矩阵： [num_sensors, rank] x [rank, num_times] => [num_sensors, num_times]
        self.recovered_tensor = tf.matmul(self.W_tf, tf.transpose(self.X_tf))
        return self.recovered_tensor

    def create_lstm_inputs_vectorized(self):
        """
        这里名字沿用你原来的（避免你其他地方改动），实际是“序列窗口输入”：
        返回 seq_inputs: [data_len, lag_len, rank]
        """
        data_len = self.num_times - self.max_lag
        if data_len <= 0:
            if self.debug:
                print(f"[WARN] data_len={data_len} <= 0, 跳过时间正则")
            return None

        start_idx = tf.constant(self.max_lag, dtype=tf.int64)
        end_idx = tf.constant(self.num_times, dtype=tf.int64)

        # indices: [data_len, lag_len]，每一行是该样本的历史窗口索引
        indices = tf.range(start_idx, end_idx, dtype=tf.int64)[:, tf.newaxis] - self.time_lags[tf.newaxis, :]
        flat_indices = tf.reshape(indices, [-1])

        gathered = tf.gather(self.X_tf, flat_indices, axis=0)
        seq_inputs = tf.reshape(gathered, [data_len, self.lag_len, self.rank])
        return seq_inputs

    def compute_graph_laplacian(self):
        """
        图拉普拉斯平滑项：sum_{(i,j) in edges} ||W_i - W_j||^2
        这里沿用你原来的“上三角去重”写法
        """
        if self.adj is None or self.lambda_w == 0:
            return tf.constant(0.0, dtype=tf.float32)

        adj_upper = tf.linalg.band_part(self.adj, 0, -1) - tf.linalg.band_part(self.adj, 0, 0)
        indices = tf.where(adj_upper > 0)

        if indices.shape[0] == 0:
            return tf.constant(0.0, dtype=tf.float32)

        i_indices = indices[:, 0]
        j_indices = indices[:, 1]

        W_i = tf.gather(self.W_tf, i_indices)
        W_j = tf.gather(self.W_tf, j_indices)

        diff = W_i - W_j
        squared_norms = tf.reduce_sum(tf.square(diff), axis=1)
        total_norm = tf.reduce_sum(squared_norms)

        return total_norm * self.lambda_w

    def Transformer_X_predict(self, X_sub, training=True):
        """
        X_sub: [B, L, rank]
        return: [B, rank]
        """
        # 可学习位置编码
        pos = self.pos_emb(tf.range(self.lag_len)[tf.newaxis, :])  # [1, L, rank]
        x = X_sub + pos

        # Self-Attention
        attn_out = self.mha(x, x, training=training)  # [B, L, rank(近似)]
        x = self.ln1(x + self.drop1(attn_out, training=training))

        # FFN
        ffn_out = self.ffn(x, training=training)  # [B, L, rank]
        x = self.ln2(x + self.drop2(ffn_out, training=training))

        # 取最后一个时间步作为预测
        last = x[:, -1, :]  # [B, rank]
        return self.out_proj(last)  # [B, rank]

    def loss_cal(self):
        """
        总损失：
        - 重建误差（训练mask）
        - L2正则（W、X）
        - 图拉普拉斯正则（W）
        - Transformer时间正则（X）
        """
        # 1) 重建误差
        self.recovered_tensor = self.call(training=True)
        train_recovered_vec = tf.boolean_mask(self.recovered_tensor, self.train_pos_mask)

        # 理论上两个 mask 来自同一个矩阵，长度应该一致；这里做个保险
        gt = self.train_ground_truth_vec
        if tf.shape(train_recovered_vec)[0] != tf.shape(gt)[0]:
            min_len = tf.minimum(tf.shape(train_recovered_vec)[0], tf.shape(gt)[0])
            train_recovered_vec = train_recovered_vec[:min_len]
            gt = gt[:min_len]

        train_residual_vec = train_recovered_vec - gt
        train_residual_vec = tf.where(tf.math.is_nan(train_residual_vec),
                                      tf.zeros_like(train_residual_vec),
                                      train_residual_vec)
        train_residual_error = tf.math.square(tf.norm(train_residual_vec, ord='euclidean'))

        # 2) L2 正则
        W_F_norm = tf.math.square(tf.norm(self.W_tf, ord='fro', axis=(0, 1))) * self.lambda_w * self.eta
        X_F_norm = tf.math.square(tf.norm(self.X_tf, ord='fro', axis=(0, 1))) * self.lambda_x * self.eta

        # 3) 图拉普拉斯正则
        W_graph = self.compute_graph_laplacian()

        # 4) Transformer 时间正则
        seq_inputs = self.create_lstm_inputs_vectorized()
        X_time = tf.constant(0.0, dtype=tf.float32)

        if seq_inputs is not None and self.lambda_x != 0:
            try:
                self.X_predicted = self.Transformer_X_predict(seq_inputs, training=True)  # [data_len, rank]
                X_tf_slice = self.X_tf[self.max_lag:, :]  # [data_len, rank]

                # 形状匹配检查
                if X_tf_slice.shape[0] == self.X_predicted.shape[0]:
                    X_time = tf.math.square(
                        tf.norm(X_tf_slice - self.X_predicted, ord='fro', axis=(0, 1))
                    ) * self.lambda_x
            except Exception as e:
                if self.debug:
                    print(f"[WARN] Transformer预测失败，跳过时间正则: {e}")
                X_time = tf.constant(0.0, dtype=tf.float32)

        # 5) 总损失
        self.loss_val = train_residual_error + W_F_norm + X_F_norm + W_graph + X_time

        # 防 NaN
        if tf.math.is_nan(self.loss_val):
            self.loss_val = tf.constant(1.0, dtype=tf.float32)

        return self.loss_val, train_residual_error, W_F_norm, X_F_norm, X_time, W_graph

    def metrics_cal(self):
        """
        返回标量 MAPE / RMSE（验证mask）
        """
        self.recovered_tensor = self.call(training=False)
        val_recovered_vec = tf.boolean_mask(self.recovered_tensor, self.val_pos_mask)

        # 防止形状不一致
        gt = self.val_ground_truth_vec
        if tf.shape(val_recovered_vec)[0] != tf.shape(gt)[0]:
            min_len = tf.minimum(tf.shape(val_recovered_vec)[0], tf.shape(gt)[0])
            val_recovered_vec = val_recovered_vec[:min_len]
            gt = gt[:min_len]

        # 标量指标
        mape_vec = tf.keras.losses.MAPE(gt, val_recovered_vec)  # 可能是向量
        mse_vec = tf.keras.losses.MSE(gt, val_recovered_vec)

        mape = tf.reduce_mean(mape_vec)
        rmse = tf.sqrt(tf.reduce_mean(mse_vec))
        return mape, rmse

    def get_reconstructed_matrix(self):
        self.recovered_tensor = self.call(training=False)
        return self.recovered_tensor.numpy()

    def get_latent_variables(self):
        return self.W_tf.numpy(), self.X_tf.numpy()