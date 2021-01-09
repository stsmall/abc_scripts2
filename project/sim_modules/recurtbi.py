# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:31:12 2020
@author: Scott T. Small

"""
from project.sim_modules.drawdist import DrawDist

avail_dist = {"unif": ("low", "high"), "log_unif": ("low", "high"),
              "unif_int": ("low", "high"), "log_unif_int": ("low", "high"),
              "log_normal": ("mu", "sigma"), "norm_int": ("mu", "sigma"),
              "log_norm_int": ("mu", "sigma"), "exp": ("scale", "scale"),
              "beta": ("a", "b"), "constant": ("c", "c")}

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
                    assert dist[1:] in avail_dist,"dist not recognized"
                    d[k] = draw(float(low), float(high), size)
                    cn += 1
                    return itdict(d, size)
    return d

# d={"tbi1": [0, "tbi2"], "tbi2": ["tbi3", "tbi4"], "tbi3":[100, "tbi4"],"tbi4": [1000, 10000]}
