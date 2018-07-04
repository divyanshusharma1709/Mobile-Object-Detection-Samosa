#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:13:14 2018

@author: divyanshu
"""
import tensorflow as tf
import numpy as np
tf.reset_default_graph()
import pickle
'''def getImage(start, end):
    with open("final_dataset.p", "rb") as f:
        dictname = pickle.load(f)
    images = []
    labels = []
    for i in range(end):
        if(i < start):
            continue
        else:
            images.append(dictname[0][i])
            labels.append(dictname[1][i])
    return images, labels, (end - start + 1)'''
with open("final_dataset.p", "rb") as f:
        dictname = pickle.load(f)
features = dictname[0]
labels = dictname[1][0:86]
label = []
for i in range(86):
    label.append(labels[i])
#labels = np.array(labels, dtype=np.int32)
n_classes = 1
x = tf.placeholder("float", [1, 256,256,3])
y = tf.placeholder(tf.int32, 1)

def neuron_layer(X, n_neurons, n_inputs, name, activation = None):
    stddev = 2/np.sqrt(n_inputs)
    init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
    W = tf.Variable(init, name = "Weights")
    b = tf.Variable(tf.zeros([n_neurons]), name = "Biases")
    z = tf.matmul(X,W) + b
    if activation == "relu":
        return tf.nn.relu(z)
    else:
        return z






def conv_layer(input_data, weights, bias, strides):
    x = tf.nn.conv2d(input_data, weights, strides, padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)

def max_pool_layer(input_layer, filter_shape, strides):
    ksize = [1, filter_shape[0], filter_shape[1], 1]
    out_layer = tf.nn.max_pool(input_layer, ksize, strides, padding = 'SAME')
    return out_layer

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
}
bias = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
}


#Build the network


conv1 = conv_layer(x, weights['wc1'], bias['bc1'], strides = [1,2,2,1])
conv2 = conv_layer(conv1, weights['wc2'], bias['bc2'], strides = [1,2,2,1])
pool1 = max_pool_layer(conv2,[7,7], [1,2,2,1])
pool1 = tf.nn.relu(pool1)
conv3 = conv_layer(pool1, weights['wc3'], bias['bc3'],strides = [1,2,2,1])
pool2 = max_pool_layer(conv3,[4,4],[1,2,2,1])
pool2 = tf.nn.relu(pool2)
#Fully connected layer
fc1 = tf.reshape(pool2, [1, -1])
fc1 = tf.nn.relu(fc1)
fc2 = neuron_layer(fc1, 100,8192, name = "FCL2", activation = "relu")

output = neuron_layer(fc2, 1,100, name = "Output", activation = "relu")
opt = tf.nn.softmax(output)

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = opt)
loss = tf.reduce_mean(xentropy, name = "Loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(output, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()

op = []
feat = features
for i in range(96):
    feat[i] = np.expand_dims(feat[i], axis=0)
with tf.Session() as sess:
    sess.run(init)
#    out = sess.run(pool2, feed_dict = {x: features})
    for i in range(86):
        out = sess.run(opt, feed_dict = {x: feat[i], y: label[i]})
        op.append(out)
        
    
   