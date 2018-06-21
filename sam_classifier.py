#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:31:29 2018

@author: divyanshu
"""

import tensorflow as tf
import cv2, os
import numpy as np
from matplotlib import pyplot as plt
directory = "/home/divyanshu/Documents/resized_samosa/" #Change to pickle file
def getImage(start, end, directory):
    images = []
    i = start
    for image in os.listdir(directory):
        if(i > end):
            break
        if(i < start):
            continue
        else:
            img = cv2.imread(os.path.join(directory, image))
            images.append(img)
            i = i + 1
    return images

im = getImage(5,10,directory)
filters = np.zeros(shape = (7,7,3,2),dtype = np.float32)
filters[:,3,:,0] = 1 #Vertical Line
filters[3,:,:,1] = 1 #Horizontal Line
X = tf.placeholder(tf.float32,shape = (None,256,256,3))
l1 = tf.nn.conv2d(X, filters, strides = [1,2,2,1], padding = "SAME")
with tf.Session() as sess:
    output = sess.run(l1,feed_dict = {X:im})
plt.imshow(output[3,:,:,0])