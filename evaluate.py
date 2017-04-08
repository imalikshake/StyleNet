from __future__ import print_function
import numpy as np
import h5py
from data_util import BatchGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

from midi_util import *
import os
import sys
from sklearn.preprocessing import MinMaxScaler

model_path = "./runs/third_run/model"
x_path = "./runs/third_run/inputs"
y_path = "./runs/third_run/velocities"
pred_path = "./runs/third_run/predictions"

scaler =  MinMaxScaler(feature_range=(-1,1))

# input_size = lstm_size = 128
# num_layers = 2
#
# inputs = tf.placeholder(tf.float32, [None, None, input_size])
# outputs = tf.placeholder(tf.float32, [None, None, input_size])
#
# batch_size = tf.shape(inputs)[0]
#
# def single_cell(forget_bias=1.0, use_LSTM=True):
#     if use_LSTM:
#         cell = tf.contrib.rnn.BasicLSTMCell(lstm_size,forget_bias=forget_bias, activation=tf.nn.relu)
#     else:
#         cell = tf.contrib.rnn.GRUCell(lstm_size,forget_bias=forget_bias, activation=tf.nn.relu)
#     return cell
#
# if num_layers == 1:
#     cell = single_cell
# else:
#     cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
#
# initial_state = cell.zero_state(batch_size, dtype=tf.float32)
#
# rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
#     cell, inputs, initial_state=initial_state)
#
# int_rnn_outputs = tf.round(rnn_outputs)
#
# error_l2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(outputs, rnn_outputs))))
#
# tf.summary.scalar("error", error_l2)
# summary_op = tf.summary.merge_all()
#
# optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
# gradients = optimizer.compute_gradients(error_l2)
#
# def ClipIfNotNone(grad):
#     if grad is None:
#         return grad
#
#     return tf.clip_by_value(grad, -1, 1)
#
# clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
# opt = optimizer.apply_gradients(clipped_gradients)

input_size = lstm_size = 128
num_layers = 2

inputs = tf.placeholder(tf.float32, [None, None, input_size])
outputs = tf.placeholder(tf.float32, [None, None, input_size])

batch_size = tf.shape(inputs)[0]

def single_cell(forget_bias=1.0, use_LSTM=True):
    if use_LSTM:
        cell = tf.contrib.rnn.LSTMCell(lstm_size,forget_bias=forget_bias,initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.tanh)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
    else:
        cell = tf.contrib.rnn.GRUCell(lstm_size,forget_bias=forget_bias)
    return cell

if num_layers == 1:
    cell = single_cell()
else:
    cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])


initial_state = cell.zero_state(batch_size, dtype=tf.float32)

rnn_outputs, rnn_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)


W = tf.get_variable("W", shape=[input_size, input_size],
           initializer=tf.contrib.layers.xavier_initializer())

# W = tf.Variable(tf.random_normal([input_size,input_size]))

B = tf.Variable(tf.random_normal([input_size]))

def mul_fn(current_input):
    return tf.matmul(current_input, W) + B

pred = tf.map_fn(mul_fn, rnn_outputs)

int_rnn_outputs = tf.round(pred)

# feed_train = {inputs: X_list,
#         outputs: Y_list}

# rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(outputs, rnn_outputs)))

error_l2 =  tf.reduce_mean(tf.square(tf.subtract(outputs, pred)))

tf.summary.scalar("error", error_l2)
summary_op = tf.summary.merge_all()

# optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(error_l2)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
gradients = optimizer.compute_gradients(error_l2)

def ClipIfNotNone(grad):
    if grad is None:
        return grad

    return tf.clip_by_value(grad, -1, 1)

clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
opt = optimizer.apply_gradients(clipped_gradients)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for i, filename in enumerate(os.listdir(model_path)):
#         if filename.split('.')[-1] == 'meta':
new_saver = tf.train.import_meta_graph(os.path.join(model_path,"model-e500b3.ckpt.meta"))
new_saver.restore(sess, tf.train.latest_checkpoint(model_path))




for i, filename in enumerate(os.listdir(x_path)):
    if filename.split('.')[-1] == 'npy':
        true_path = os.path.join(y_path, filename)
        # print(filename)
        in_list = []
        out_list = []
        abs_path = os.path.join(x_path,filename)
        loaded = np.load(abs_path)
        loaded = scaler.fit_transform(loaded)
        true_vel = np.load(true_path)
        # loaded = scaler.fit_transform(loaded)
        # clipped = loaded[:128]
        X_list.append(loaded)
        Y_list.append(true_vel)
        error, out = sess.run([error_l2, int_rnn_outputs],feed_dict={inputs:in_list, outputs:out_list})
        prediction = out[-1]
        fig = plt.figure(figsize=(10, 10), dpi=150)

        fig.add_subplot(2,1,1)
        plt.imshow(Y_list[-1])

        fig.add_subplot(2,2,1)
        plt.imshow(prediction)

        # fig.add_subplot()
        # plt.imshow(Y_list[-1])
        # plt.show()
        # plt.show()
        # out_path = os.path.join(pred_path, filename)
        np.save(os.path.join(png_path, "e%d" %(epoch) , prediction)

# x = np.array(x,dtype=float)
# print(x.shape)
# in_x = []
# in_x.append(x)
