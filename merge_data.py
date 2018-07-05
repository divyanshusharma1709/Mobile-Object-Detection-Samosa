#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:10:32 2018

@author: divyanshu
"""
import pickle
import numpy
with open("final_dataset.p", "rb") as f:
        dictname = pickle.load(f)
with open("data.p", "rb") as f:
        rand_data = pickle.load(f)
images = []
for i in range(136):
    pix = numpy.array(rand_data[0][i])
    images.append(pix)
random_data = [images, rand_data[1]]
pickle.dump(random_data, open("data.p","wb"))

with open("final_dataset.p", "rb") as f:
        dictname = pickle.load(f)
with open("data.p", "rb") as f:
        rand_data = pickle.load(f)

x = list(rand_data[1])
y = list(dictname[1])
for i in range(136):        
    y.append(x[i])
y.append(0.0)
data = [dictname[0], y]
pickle.dump(data, open("new_data.p","wb"))
