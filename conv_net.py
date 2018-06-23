#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 11:31:23 2018

@author: divyanshu
"""
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
    return images, labels

features, labels = getImage(0,5)

