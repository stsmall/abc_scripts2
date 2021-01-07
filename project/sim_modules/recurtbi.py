# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:31:12 2020
@author: Scott T. Small

"""
from .drawdist import DrawDist

cn = 0

def itdict(d, size):
    global cn
    while cn < len(d):
        for k, v in d.items():
            if len(v) == 3:
                dist, low, high = v
                if "tbi" in str(low):
                    if type(d[low]) is int:
                        d[k][1] = d[low]
                elif "tbi" in str(high):
                    if type(d[high]) is int:
                        d[k][2] = d[high]
                else:
                    draw = getattr(DrawDist(), dist[1:])
                    d[k] = draw(float(low), float(high), size)
                    cn += 1
                    return itdict(d, size)
    return d

# d={"tbi1": [0, "tbi2"], "tbi2": ["tbi3", "tbi4"], "tbi3":[100, "tbi4"],"tbi4": [1000, 10000]}

# def itdict(d):
#     global cn
#     while cn < len(d):
#         for k, v in d.items():
#             if type(v) is list:
#                 if "tbi" in str(v[0]):
#                     if type(d[v[0]]) is int:
#                         d[k][0] = d[v[0]]
#                 elif "tbi" in str(v[1]):
#                     if type(d[v[1]]) is int:
#                         d[k][1] = d[v[1]]
#                 else:
#                     d[k] = np.random.randint(v[0], v[1])
#                     cn += 1
#                     return itdict(d)
#     return d
