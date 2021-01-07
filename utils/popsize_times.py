#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:17:43 2020
@author: Scott T. Small

Creation of bins for calculating LD based on time segments.
Parameters:
     nb_times: int, number of time windows
     r: float,recomb rate per generation per bp
     L: int, size of each segment, in bp.
     Tmax: int, the oldest time window will start at Tmax
     per_err: int, the length of each interval, as a per of the target
         distance
     a: float, the length of time windows increases when time increases,
             at a speed that is determined by this coefficient
 Returns:
     intervals_list: list
     times: list

usage: python popszie_times.py 10 13000 3.89e-8 1e6
"""
import sys
import numpy as np

nb_times = int(sys.argv[1])
tmax = int(sys.argv[2])
r = float(sys.argv[3])
length_bp = float(sys.argv[4])
per_err = 5
a = 0.06

times = -np.ones(shape=nb_times, dtype='float')
for i in range(nb_times):
    times[i] = (np.exp(np.log(1+a*tmax)*i/(nb_times-1))-1)/a
print(f"Recommended times for popsize changes using Boitard alg (in gens): {np.rint(times)}")

# bins for LD
interval_list = []
for i in range(nb_times - 1):
    t = (times[i + 1] + times[i])/2
    d = 1/(2 * r * t)
    if d <= length_bp:
        interval_list.append([d - per_err * d/100.0, d + per_err * d/100.0])
t = tmax + times[nb_times-1] - times[nb_times-2]
# d = 10.0**8/(2.0 * t)
d = 1/(2*r*t)
interval_list.append([d-per_err * d/100.0, d + per_err * d/100.0])
print(f"Average LD will be computed for the following distance bins (in bp): {np.rint(interval_list)}")
