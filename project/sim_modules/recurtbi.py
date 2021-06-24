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
              "beta": ("a", "b"), "constant": ("c", "c"), "posterior":("file", "col")}

cn = 0


def itdict(dt, size):
    global cn
    while cn < len(dt):
        for k, v in dt.items():
            if len(v) == 3:
                dist, low, high = v
                if "tbi" in str(low):
                    if len(dt[low]) == size:
                        dt[k][1] = dt[low]
                elif "tbi" in str(high):
                    if len(dt[high]) == size:
                        dt[k][2] = dt[high]
                else:
                    draw = getattr(DrawDist(), dist[1:])
                    assert dist[1:] in avail_dist, "dist not recognized"
                    if type(low) is str and type(high) is str:
                        # low is str, high is str   '0' '123'
                        dt[k] = draw(float(low), float(high), size)
                    elif type(low) is str and len(high) == size:
                        # low is str, high is array '0' ([1,2,3])
                        dt[k] = draw([float(low)]*size, high, size)
                    elif len(low) == size and type(high) is str:
                        # low is array, high is str ([1,2,3]) '10'
                        dt[k] = draw(low, [float(high)]*size, size)
                    elif len(low) == size and len(high) == size:
                        # low is array, high is array ([1,2,3]) ([1,2,3])
                        dt[k] = draw(low, high, size)
                    cn += 1
                    return itdict(dt, size)
    return dt

# d={"tbi1": [0, "tbi2"], "tbi2": ["tbi3", "tbi4"], "tbi3":[100, "tbi4"],"tbi4": [1000, 10000]}
