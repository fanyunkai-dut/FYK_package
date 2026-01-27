import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
import numpy as np

class LSTMNNMF(Model):
    """
    仅矩阵分解 + 重建误差（无LSTM、无图拉普拉斯、无L2正则）
    """
    def __init__(self, training_set, training_ground_truth, train_mask, val_mask,
                 rank, latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}):
        super(LSTMNNMF, self).__init__()

        self.rank = rank

        # 掩码
        self.train_pos_mask = tf.constant(train_mask, dtype=tf.bool)
        self.val_pos_mask = tf.constant(val_mask, dtype=tf.bool)

        # 数据
        self.ground_truth_tf = tf.constant(training_ground_truth, dtype=tf.float32)
        self.train_ground_truth_vec = tf.boolean_mask(self.ground_truth_tf, self.train_pos_mask)
        self.val_ground_truth_vec = tf.boolean_mask(self.ground_truth_tf, self.val_pos_mask)

        self.num_sensors = training_ground_truth.shape[0]
        self.num_times = training_ground_truth.shape[1]

        # 潜在变量 W, X
        self.W_tf = tf.Variable(
            tf.random.normal([self.num_sensors, self.rank],
                             mean=latent_normal_init_params['mean'],
                             stddev=latent_normal_init_params['stddev']),
            dtype=tf.float32
        )
        self.X_tf = tf.Variable(
            tf.random.normal([self.num_times, self.rank],
                             mean=latent_normal_init_params['mean'],
                             stddev=latent_normal_init_params['stddev']),
            dtype=tf.float32
        )

        self.recovered_tensor = None
        self.loss_val = None

        print(f"模型初始化完成(仅重建误差): rank={rank}")

    def call(self):
        self.recovered_tensor = tf.matmul(self.W_tf, tf.transpose(self.X_tf))
        return self.recovered_tensor

    def loss_cal(self):
        """
        只计算：训练位置上的重建误差（sum of squares）
        """
        self.recovered_tensor = self.call()
        train_recovered_vec = tf.boolean_mask(self.recovered_tensor, self.train_pos_mask)

        # residual
        train_residual_vec = train_recovered_vec - self.train_ground_truth_vec

        # 避免 NaN
        train_residual_vec = tf.where(tf.math.is_nan(train_residual_vec),
                                      tf.zeros_like(train_residual_vec),
                                      train_residual_vec)

        train_residual_error = tf.math.square(tf.norm(train_residual_vec, ord='euclidean'))
        self.loss_val = train_residual_error
        return self.loss_val, train_residual_error

    def metrics_cal(self):
        self.recovered_tensor = self.call()
        val_recovered_vec = tf.boolean_mask(self.recovered_tensor, self.val_pos_mask)
        mape = tf.keras.losses.MAPE(self.val_ground_truth_vec, val_recovered_vec)
        rmse = tf.sqrt(tf.keras.losses.MSE(self.val_ground_truth_vec, val_recovered_vec))
        return mape, rmse

    def get_reconstructed_matrix(self):
        self.recovered_tensor = self.call()
        return self.recovered_tensor.numpy()

    def get_latent_variables(self):
        return self.W_tf.numpy(), self.X_tf.numpy()