#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:13:14 2018

@author: divyanshu
"""
import tensorflow as tf
import numpy as np



import pickle
def getImage(start, end):
    with open("final_data.p", "rb") as f:
        dictname = pickle.load(f)
    images = []
    labels = []
    for i in range(end):
        if(i < start):
            continue
        else:
            images.append(dictname[0][i])
            labels.append(dictname[1][i])
    return images, labels, (end - start + 1)

features, labels, num = getImage(0,5)
n_classes = 2
#X = tf.placeholder(tf.float32,shape = (None,256,256,3))
x = tf.placeholder("float", [None, 256,256,3])
y = tf.placeholder("float", [None, 2])

def conv_layer(input_data, weights, bias, strides):
    x = tf.nn.conv2d(input_data, weights, strides, padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)
print(np.ones(num))


def max_pool_layer(input_layer, filter_shape, strides):
    ksize = [1, filter_shape[0], filter_shape[1], 1]
    out_layer = tf.nn.max_pool(input_layer, ksize, strides, padding = 'SAME')
    return out_layer

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
bias = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}


#Build the network


conv1 = conv_layer(x, weights['wc1'], bias['bc1'], strides = [1,2,2,1])
conv2 = conv_layer(conv1, weights['wc2'], bias['bc2'], strides = [1,2,2,1])
pool1 = max_pool_layer(conv2,[7,7], [1,2,2,1])
conv3 = conv_layer(pool1, weights['wc3'], bias['bc3'],strides = [1,2,2,1])
pool2 = max_pool_layer(conv3,[4,4],[1,2,2,1])
init = tf.global_variables_initializer()
#Fully connected layer
fc1 = tf.reshape(pool2, [-1, x = weights['wd1'].get_shape().as_list()[0]])




with tf.Session() as sess:
    sess.run(init)
    out = sess.run(conv1, feed_dict = {x: features})