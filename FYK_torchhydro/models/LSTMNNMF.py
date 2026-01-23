import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
import numpy as np

class LSTMNNMF(Model):
    """
    LSTM正则化神经网络矩阵分解模型
    
    适用于处理具有三种缺失情况的数据：
    1. 真实观测值（用于训练）
    2. 人为缺失值（用于验证）
    3. 原始缺失值（最终填补目标）
    """
    
    def __init__(self, training_set, training_ground_truth, train_mask, val_mask, 
                 A, rank, time_lags, lambda_w=100, lambda_x=100, eta=0.02, 
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}):
        super(LSTMNNMF, self).__init__()
        
        # 基本参数
        self.rank = rank
        self.time_lags = tf.constant(time_lags[::-1], dtype=tf.int64)  # constant为超参数，不参与反向传播和调参，[::-1]是调转方向，统一int64
        self.lag_len = len(time_lags)
        self.max_lag = np.max(time_lags)
        
        # 掩码设置（使用传入的掩码）
        self.train_pos_mask = tf.constant(train_mask, dtype=tf.bool)
        self.val_pos_mask = tf.constant(val_mask, dtype=tf.bool)
        
        # 正则化参数
        self.lambda_w = lambda_w
        self.lambda_x = lambda_x
        self.eta = eta
        
        # 邻接矩阵
        self.adj = tf.constant(A, dtype=tf.float32)
        
        # 训练数据
        self.ground_truth_tf = tf.constant(training_ground_truth, dtype=tf.float32)
        self.train_ground_truth_vec = tf.boolean_mask(self.ground_truth_tf, self.train_pos_mask) #boolean的作用是只提取有值的并展开为一维
        self.val_ground_truth_vec = tf.boolean_mask(self.ground_truth_tf, self.val_pos_mask)
        
        # 初始化参数
        self.latent_normal_init_params = latent_normal_init_params
        self.num_sensors = training_ground_truth.shape[0]
        self.num_times = training_ground_truth.shape[1]
        
        # 初始化W和X矩阵（潜在变量）
        # 避免使用过时的initializer
        self.W_tf = tf.Variable(
            tf.random.normal(shape=[self.num_sensors, self.rank], 
                           mean=latent_normal_init_params['mean'], 
                           stddev=latent_normal_init_params['stddev']),
            dtype=tf.float32
        )
        self.X_tf = tf.Variable(
            tf.random.normal(shape=[self.num_times, self.rank], 
                           mean=latent_normal_init_params['mean'], 
                           stddev=latent_normal_init_params['stddev']),
            dtype=tf.float32
        )

        # LSTM和全连接层
        self.LSTM = layers.LSTM(self.rank, input_shape=(self.lag_len, self.rank), unroll=True)
        self.Dense = layers.Dense(self.rank)
        
        # 用于存储中间结果
        self.recovered_tensor = None
        self.X_predicted = None
        self.loss_val = None
        
        print(f"模型初始化完成: rank={rank}, time_lags={time_lags}, max_lag={self.max_lag}")
    
    def call(self):
        """
        前向传播：重建数据矩阵
        W_tf: [num_sensors, rank]
        X_tf: [num_times, rank]
        输出: [num_sensors, num_times]
        """
        self.recovered_tensor = tf.matmul(self.W_tf, tf.transpose(self.X_tf))
        return self.recovered_tensor
    
    def create_lstm_inputs_vectorized(self):
        """
        向量化创建LSTM输入序列，避免图构建问题
        
        返回:
            lstm_inputs: LSTM输入，形状为[data_len, lag_len, rank]
        """
        # 计算数据长度
        data_len = self.num_times - self.max_lag
        
        if data_len <= 0:
            print(f"警告: data_len={data_len}, max_lag={self.max_lag}, num_times={self.num_times}")
            print(f"跳过LSTM正则化项")
            return None
        
        print(f"创建LSTM输入: data_len={data_len}, max_lag={self.max_lag}, num_times={self.num_times}")
        
        # 创建所有样本的索引矩阵
        # 形状: (data_len, lag_len)
        # 使用tf.range并指定dtype为int64，与time_lags保持一致
        start_idx = tf.constant(self.max_lag, dtype=tf.int64)
        end_idx = tf.constant(self.num_times, dtype=tf.int64)
        
        indices = tf.range(start_idx, end_idx, dtype=tf.int64)[:, tf.newaxis] - self.time_lags[tf.newaxis, :]
        
        # 展平索引
        flat_indices = tf.reshape(indices, [-1])
        
        # 收集数据
        gathered = tf.gather(self.X_tf, flat_indices, axis=0)
        
        # 重新整形为 (data_len, lag_len, rank)，即[样本1：[时间步3*特征数60的矩阵], 样本2:[时间步3*特征数60的矩阵], ...],样本即是data_len
        lstm_inputs = tf.reshape(gathered, [data_len, self.lag_len, self.rank])
        
        print(f"LSTM输入创建完成: 形状={lstm_inputs.shape}")
        
        return lstm_inputs
    
    def compute_graph_laplacian(self):
    # 使用上三角矩阵避免重复计算
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

    def loss_cal(self):
        """
        计算总损失
        """
        print("开始计算损失...")
        
        try:
            # 1. 重建损失
            self.recovered_tensor = self.call()
            train_recovered_vec = tf.boolean_mask(self.recovered_tensor, self.train_pos_mask)
            
            # 检查形状
            print(f"重建数据形状: {self.recovered_tensor.shape}")
            print(f"训练重建向量形状: {train_recovered_vec.shape}")
            print(f"训练真实值向量形状: {self.train_ground_truth_vec.shape}")
            
            # 如果形状不匹配，调整
            if train_recovered_vec.shape != self.train_ground_truth_vec.shape:
                print("警告: 形状不匹配，进行调整")
                min_len = min(train_recovered_vec.shape[0], self.train_ground_truth_vec.shape[0])
                train_recovered_vec = train_recovered_vec[:min_len]
                train_ground_truth_vec = self.train_ground_truth_vec[:min_len]
            else:
                train_ground_truth_vec = self.train_ground_truth_vec
            
            train_residual_vec = train_recovered_vec - train_ground_truth_vec
            
            # 检查是否有NaN
            nan_in_residual = tf.reduce_any(tf.math.is_nan(train_residual_vec))
            
            if tf.equal(nan_in_residual, True):
                print("警告: train_residual_vec中包含NaN，使用替代值")
                # 使用一个小值替代NaN
                train_residual_vec = tf.where(tf.math.is_nan(train_residual_vec), 
                                             tf.zeros_like(train_residual_vec), 
                                             train_residual_vec)
            
            train_residual_error = tf.math.square(tf.norm(train_residual_vec, ord='euclidean'))
            print(f"重建误差: {train_residual_error.numpy():.4f}")
            
            # 2. L2正则化项
            W_F_norm = tf.math.square(tf.norm(self.W_tf, ord='fro', axis=(0, 1))) * self.lambda_w * self.eta
            X_F_norm = tf.math.square(tf.norm(self.X_tf, ord='fro', axis=(0, 1))) * self.lambda_x * self.eta
            print(f"W正则化: {W_F_norm.numpy():.4f}, X正则化: {X_F_norm.numpy():.4f}")
            
            # 3. 图拉普拉斯正则化（空间平滑）
            W_norm = self.compute_graph_laplacian()
            print(f"图拉普拉斯正则化: {W_norm.numpy():.4f}")

            # 4. LSTM时间正则化
            lstm_inputs = self.create_lstm_inputs_vectorized()
            X_norm = tf.constant(0.0, dtype=tf.float32)  # 先设为0，避免LSTM问题
            
            if lstm_inputs is not None:
                try:
                    self.X_predicted = self.LSTM_X_predict(lstm_inputs)
                    
                    # 获取对应的X_tf部分
                    X_tf_slice = self.X_tf[self.max_lag:, :]
                    
                    # 确保形状匹配
                    if X_tf_slice.shape[0] == self.X_predicted.shape[0]:
                        X_norm = tf.math.square(
                            tf.norm(X_tf_slice - self.X_predicted, ord='fro', axis=(0, 1))
                        ) * self.lambda_x
                        print(f"LSTM时间正则化: {X_norm.numpy():.4f}")
                    else:
                        print(f"形状不匹配: X_tf_slice={X_tf_slice.shape}, X_predicted={self.X_predicted.shape}")
                except Exception as e:
                    print(f"LSTM预测时出错: {e}")
                    X_norm = tf.constant(0.0, dtype=tf.float32)
            
            # 5. 总损失
            self.loss_val = train_residual_error + W_F_norm + X_F_norm + X_norm + W_norm
            
            # 检查损失是否为NaN
            if tf.math.is_nan(self.loss_val):
                print(f"警告: 损失中包含NaN! 各分量: train_residual_error={train_residual_error}, "
                      f"W_F_norm={W_F_norm}, X_F_norm={X_F_norm}, X_norm={X_norm}, W_norm={W_norm}")
                # 返回一个有限的损失值
                self.loss_val = tf.constant(1.0, dtype=tf.float32)
            
            print(f"总损失: {self.loss_val.numpy():.4f}")
            
            return self.loss_val, train_residual_error, W_F_norm, X_F_norm, X_norm, W_norm
            
        except Exception as e:
            print(f"损失计算时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个默认损失
            return (tf.constant(1.0, dtype=tf.float32), 
                    tf.constant(1.0, dtype=tf.float32), 
                    tf.constant(0.0, dtype=tf.float32), 
                    tf.constant(0.0, dtype=tf.float32), 
                    tf.constant(0.0, dtype=tf.float32), 
                    tf.constant(0.0, dtype=tf.float32))
    
    def LSTM_X_predict(self, X_sub):
        """
        LSTM预测下一时刻的X
        """
        try:
            lstm_out = self.LSTM(X_sub)
            self.X_predicted = self.Dense(lstm_out)
            return self.X_predicted
        except Exception as e:
            print(f"LSTM预测时出错: {e}")
            # 返回一个与输入batch大小相同的零矩阵
            batch_size = X_sub.shape[0]
            return tf.zeros([batch_size, self.rank], dtype=tf.float32)
    
    def metrics_cal(self):
        self.recovered_tensor = self.call()
        val_recovered_vec = tf.boolean_mask(self.recovered_tensor, self.val_pos_mask)  
        mape = tf.keras.losses.MAPE(self.val_ground_truth_vec, val_recovered_vec)
        rmse = tf.sqrt(tf.keras.losses.MSE(self.val_ground_truth_vec, val_recovered_vec))
        return mape, rmse
    
    
    def get_reconstructed_matrix(self):
        """获取重建的完整矩阵"""
        self.recovered_tensor = self.call()
        return self.recovered_tensor.numpy()
    
    def get_latent_variables(self):
        """获取潜在变量W和X"""
        return self.W_tf.numpy(), self.X_tf.numpy()