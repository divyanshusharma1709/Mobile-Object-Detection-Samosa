#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:13:14 2018

@author: divyanshu
"""
import tensorflow as tf,cv2
import numpy as np
tf.reset_default_graph()
directory = "/home/divyanshu/Documents/samosa"
import pickle
with open("new_data.p", "rb") as f:
        dictname = pickle.load(f)
features = dictname[0]
for i in range(225):
    features[i] = cv2.normalize(features[i],features[i], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

labels = dictname[1]
'''dictname = [features, labels]
pickle.dump(dictname, open("new_data.p", "wb"))
for i in range(225):
    if(np.shape(features[i]) != (256,256,3)):
        del(features[i])
        del(labels[i])
'''
for i in range(len(features)):
    features[i] = np.expand_dims(features[i], axis=0)
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)
#labels = np.array(labels, dtype=np.int32)
n_classes = 2
x = tf.placeholder("float", [1, 256,256,3], name = "Input")
y = tf.placeholder(tf.int32, None)
keep_prob = tf.placeholder(tf.float32)

def neuron_layer(X, n_neurons, n_inputs, name, activation = None):
    stddev = 2/np.sqrt(n_inputs)
    init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
    W = tf.Variable(init, name = "Weights")
    b = tf.Variable(tf.zeros([n_neurons]), name = "Biases")
    z = tf.matmul(X,W) + b
    z = tf.identity(z, name)
    if activation == "relu":
        return tf.nn.relu(z)
    elif activation =="softmax":
        return tf.nn.softmax(z)
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
fc1 = tf.reshape(pool2, [1, -1]) #Check this also
fc1 = tf.nn.relu(fc1)
fc2 = neuron_layer(fc1, 100,8192, name = "FCL2", activation = "relu")
drop = tf.nn.dropout(fc2, keep_prob)
opt = neuron_layer(drop, 2,100, name = "Output")
labels_max = tf.reduce_max(labels)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = opt)
loss = tf.reduce_mean(xentropy, name = "Loss")
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_op = optimizer.minimize(loss)
correct = tf.equal(tf.argmax(y),tf.argmax(opt))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

accurate = []
oplist = []
net_loss = []
MODEL_NAME = 'convnet'

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
    
    for i in range(len(features_train)):
        out = sess.run(opt, feed_dict = {x: features_train[i], keep_prob:0.5})
        oplist.append(out)
        
    for epochs in range(300):
        losslist = []
        for i in range(len(features_train)):
            optimize = sess.run(training_op, feed_dict = {x: features_train[i], y: [labels_train[i]], keep_prob:0.5})
            loss_ = sess.run(loss, feed_dict = {x: features_train[i], y: [labels_train[i]], keep_prob:0.5})   
            losslist.append(loss_)
            print(loss_)

        net_loss.append(losslist)
    '''for i in range(len(features_test)):
            out = sess.run(opt, feed_dict = {x: features_test[i], y: [labels_test[i]], test_train: "test", keep_prob:1.0})
            out = tf.identity(out, name="Output")
            acc = sess.run(accuracy, feed_dict = {x: features_test[i], y: [labels_test[i]], test_train: "train", keep_prob:1.0})
            accurate.append(acc)'''
    writer.close()
    saver.save(sess, directory)
            
            
ctr = 0
for i in range(len(losslist)):
    if(losslist[i] != 0.0):
        ctr = ctr + 1
perzero = (ctr/len(losslist)) * 100







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:42:51 2018

@author: divyanshu
"""

import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

directory = "/home/divyanshu/Documents/samosa/saved_without_softmax"

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(directory, "Output")