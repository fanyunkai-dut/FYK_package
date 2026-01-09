import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
import numpy as np
import time
import numpy as np
import time
from ipywidgets import IntProgress
from IPython.display import display

class LSTMNNMF(Model):
    def __init__(self, training_set, ground_truth, A, rank, time_lags, lambda_w = 100, lambda_x = 100, eta = 0.02, latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}):
        super(LSTMNNMF, self).__init__()
        self.rank = tf.constant(rank)
        self.time_lags = tf.constant(time_lags[::-1])
        self.lag_len = tf.constant(len(self.time_lags))
        self.max_lag = tf.constant(np.max(time_lags))
        self.train_pos_mask = (training_set != 0)
        self.test_pos_mask = ((training_set == 0) & (ground_truth != 0))
        self.lambda_w = lambda_w
        self.lambda_x = lambda_x
        self.eta = eta
        self.adj = tf.constant(A, dtype=tf.float32)
        
        self.ground_truth_tf = tf.constant(ground_truth, dtype=tf.float32)
        self.train_ground_truth_vec = tf.boolean_mask(self.ground_truth_tf, self.train_pos_mask)
        self.test_ground_truth_vec = tf.boolean_mask(self.ground_truth_tf, self.test_pos_mask)
        
        self.latent_normal_init_params = latent_normal_init_params
        self.num_sensors = training_ground_truth.shape[0]
        self.num_times = training_ground_truth.shape[1]
        
        initializer = tf.compat.v1.truncated_normal_initializer(**self.latent_normal_init_params)
        self.W_tf = tf.Variable(initializer(shape=[self.num_sensors, self.rank], dtype=tf.float32))
        self.X_tf = tf.Variable(initializer(shape=[self.num_times, self.rank], dtype=tf.float32))

        self.LSTM = layers.LSTM(self.rank, input_shape = (self.lag_len, self.rank), unroll=True)
        self.Dense = layers.Dense(self.rank)


    def call(self):
        self.recovered_tensor = tf.matmul(self.W_tf, tf.transpose(self.X_tf))
        return self.recovered_tensor

    def LSTM_X_predict(self, X_sub):
        lstm_out = self.LSTM(X_sub)
        self.X_predicted = self.Dense(lstm_out)
        return self.X_predicted
    
    def create_lstm_inputs(self):
        lstm_inputs = tf.gather(self.X_tf, 0 + self.max_lag - self.time_lags, axis=0)
        lstm_inputs = tf.expand_dims(lstm_inputs, axis=0)
        data_len = self.X_tf.shape[0] - self.max_lag
        for t in tf.range(1, data_len):
            ipt = tf.gather(self.X_tf, t + self.max_lag - self.time_lags, axis=0)
            ipt_exp = tf.expand_dims(ipt, axis=0)
            e = tf.concat([lstm_inputs, ipt_exp], 0)
        return e
    
    def loss_cal(self):
        self.recovered_tensor = self.call()
        train_recovered_vec = tf.boolean_mask(self.recovered_tensor, self.train_pos_mask)
        train_residual_vec = train_recovered_vec - self.train_ground_truth_vec
        train_residual_error = tf.math.square(tf.norm(train_residual_vec, ord='euclidean'))
        W_F_norm = tf.math.square(tf.norm(self.W_tf, ord='fro', axis=(0,1))) * self.lambda_w * self.eta
        X_F_norm = tf.math.square(tf.norm(self.X_tf, ord='fro', axis=(0,1))) * self.lambda_x * self.eta
        
        W_norm = tf.constant(0, dtype=tf.float32)
        for i in tf.range(self.num_sensors):
            for j in tf.range(i + 1, self.num_sensors):
                if self.adj[i, j] == 1:
                    W_norm = W_norm + tf.math.square(tf.norm(self.W_tf[i, :] - self.W_tf[j, :], ord='euclidean'))
        W_norm = W_norm * self.lambda_w
        
        lstm_inputs = self.create_lstm_inputs()
        self.X_predicted = self.LSTM_X_predict(lstm_inputs)

        X_norm = tf.math.square(tf.norm(self.X_tf[self.max_lag:, :] - self.X_predicted, ord='fro', axis=(0,1))) * self.lambda_x
        
        self.loss_val = train_residual_error + W_F_norm  + X_F_norm + X_norm + W_norm
        return self.loss_val, train_residual_error, W_F_norm, X_F_norm, X_norm
    
    def metrics_cal(self):
        self.recovered_tensor = self.call()
        test_recovered_vec = tf.boolean_mask(self.recovered_tensor, self.test_pos_mask)
        self.mape = tf.keras.losses.MAPE(self.test_ground_truth_vec, test_recovered_vec)
        self.rmse = tf.math.sqrt(tf.keras.losses.MSE(self.test_ground_truth_vec, test_recovered_vec))
        return self.mape, self.rmse
        
training_set = np.load('/home/fanyunkai/FYK_data/processed_dataset2.5/WQ_hanjiang_train.npy')
training_ground_truth = np.load('/home/fanyunkai/FYK_data/processed_dataset2.5/WQ_hanjiang_true_train.npy')
A = np.load('/home/fanyunkai/FYK_data/processed_dataset2.5/adj.npy')

# 配置参数
rank = 60
time_lags = np.array([1, 2, 288])
lambda_w = 100
lambda_x = 100
eta = 0.1

# 创建并训练模型
model = LSTMNNMF(training_set, training_ground_truth, A, rank, time_lags, lambda_w=lambda_w, lambda_x=lambda_x, eta=eta)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        loss = model.loss_cal()
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

EPOCHS = 500
start_time = time.time()
f = IntProgress(min = 0, max = EPOCHS) # instantiate the bar
display(f) # display the bar
rmse_list = []
mape_list = []
with tf.device('/CPU:0'):
    for epoch in range(EPOCHS):
        f.value = epoch
        loss, train_residual_error, W_F_norm, X_F_norm, X_norm = train_step()
        mape, rmse = model.metrics_cal()
        rmse_list.append(rmse)
        mape_list.append(mape)
        if (epoch + 1) %50 == 0:
            print('Epoch: %d, loss: %.2f, time cost: %f, test MAPE = %.2f, RMSE = %.2f'%((epoch+1, loss, time.time() - start_time, mape, rmse)))
            print('Residual error: %f, W l2 norm: %f, X l2 norm: %f, X norm: %f'%(train_residual_error, W_F_norm, X_F_norm, X_norm))
            print()
            start_time = time.time()
