#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 10:36:22 2018

@author: divyanshu
"""
import os, pickle, numpy as np
from PIL import Image
imlist = []
directory = "/home/divyanshu/Documents/random_images_resized"
for image in os.listdir(directory):
    img = Image.open(os.path.join(directory, image))
    imlist.append(img)
data = [imlist, np.zeros(136)]
pickle.dump(data, open("data.p",'wb'))
    