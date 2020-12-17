#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 17:46:23 2018

@author: scott
"""
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--inFile")
parser.add_argument("--mean", action="store_true")
parser.add_argument("--median", action="store_true")
args = parser.parse_args()

v = []
with open(args.inFile, 'r') as stats:
    next(stats)
    for line in stats:
        x = line.split()
        xx = np.array(x[4:], dtype=np.float)
        if any(np.isinf(xx)):
            continue
        else:
            v.append(xx)
p = np.nanmean(np.vstack(v), axis=0)
m = np.nanmedian(np.vstack(v), axis=0)
#r = np.std(np.vstack(v), axis=0)
#print("{}".format(" ".join(map(str, r))))
if args.mean:
    print("{}\n".format(" ".join(map(str, p))))
if args.median:
    print("{}\n".format(" ".join(map(str, m))))


