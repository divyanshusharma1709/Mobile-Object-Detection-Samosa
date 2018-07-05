#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:28:52 2018

@author: divyanshu
"""
from PIL import Image
import os, sys
width = height = 256
directory = "/home/divyanshu/Downloads/random/"
for image in os.listdir(directory):
    print('Resizing image ' + image)
 
    # Open the image file.
    img = Image.open(os.path.join(directory, image)).convert('RGB').save(os.path.join(directory, image))
    img = Image.open(os.path.join(directory, image))

 
    # Resize it.
    img = img.resize((width, height), Image.BILINEAR)
 
    # Save it back to disk.
    img.save(os.path.join(directory, 'resized-' + image))
 
print('Batch processing complete.')