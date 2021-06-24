#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:31:12 2020
@author: Scott T. Small

module for generating draws from priors for filet_sims.py
"""

import pandas as pd
import numpy as np
from project.sim_modules.drawdist import DrawDist
from project.sim_modules.recurtbi import itdict

avail_dist = {"unif": ("low", "high"), "log_unif": ("low", "high"),
              "unif_int": ("low", "high"), "log_unif_int": ("low", "high"),
              "log_normal": ("mu", "sigma"), "norm_int": ("mu", "sigma"),
              "log_norm_int": ("mu", "sigma"), "exp": ("scale", "scale"),
              "beta": ("a", "b"), "constant": ("c", "c"), "posterior":("file", "col")}


def draw_params(param_dt, size: int, condition_ls):
    """Draws parameters from dist of lenght size.

    avail_dist = {"unif":("low", "high"), "log_unif": ("low", "high"),
                  "unif_int":("low", "high"), "log_unif_int:("low", "high"),
                  "log_normal":("mu", "sigma"), "norm_int":("mu", "sigma"),
                  "log_norm_int":("mu", "sigma"), "exp":("scale", "scale"),
                  "beta":("a", "b"), "constant":("c", "c"), "posterior":("file", "col")}

    Parameters
    ----------
    param_dt : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    param_dt : TYPE
        DESCRIPTION.

    """
    time_ls = []
    value_ls = []
    tbi_dt = {}

    # set up tbi dict
    for i, tbi in enumerate(param_dt["tbi"]):
        tbi_dt[tbi] = param_dt["time"][i]
    tbi_dt = itdict(tbi_dt, size)

    # # set up last w/ not tbi in low/high
    # dist, low, high = tbi_dt[f"tbi{i}"]
    # draw = getattr(DrawDist(), dist[1:])
    # tbi_dt[f"tbi{i}"] = draw(float(low), float(high), size)
    # # fill in remaining
    # i -= 1  # interate through in reverse
    # while i >= 0:
    #     dist, low, high = tbi_dt[f"tbi{i}"]
    #     draw = getattr(DrawDist(), dist[1:])
    #     if "tbi" in low or "tbi" in high:
    #         if "tbi" in low and "tbi" in high:
    #             low = tbi_dt[low]
    #             high = tbi_dt[high]
    #             size_n = 1
    #         elif "tbi" in low:
    #             low = tbi_dt[low]
    #             high = np.full(size, float(high))
    #             size_n = 1
    #         elif "tbi" in high:
    #             high = tbi_dt[high]
    #             low = np.full(size, float(low))
    #             size_n = 1
    #         try:
    #             tbi_dt[f"tbi{i}"] = draw(low, high, size_n)
    #         except ValueError:
    #             assert low == high
    #             # low == high, ensure both events happen at the same time
    #             tbi_dt[f"tbi{i}"] = tbi_dt[low]
    #     else:
    #         tbi_dt[f"tbi{i}"] = draw(float(low), float(high), size)
    #     i -= 1

    # check conditions
    if condition_ls:
        for condition in condition_ls:
            tbi_1, cond, tbi_2 = condition
            if cond == "lt":
                assert all(tbi_dt[tbi_1] < tbi_dt[tbi_2])
            elif cond == "gt":
                assert all(tbi_dt[tbi_1] > tbi_dt[tbi_2])
    # fill to one list
    time_ls = [tbi_dt[tbi] for tbi in param_dt["tbi"]]
    param_dt["time"] = time_ls
    # value should have no tbi
    for tvalue in param_dt["value"]:
        if tvalue:
            dist, low, high = tvalue
            assert dist[1:] in avail_dist, "dist not recognized"
            draw = getattr(DrawDist(), dist[1:])
            value_ls.append(draw(float(low), float(high), size))
        else:
            value_ls.append([np.nan]*size)
    param_dt["value"] = value_ls

    return param_dt


def parse_model(in_file, size):
    """Parse rows in model.

    Parameters
    ----------
    in_file : TYPE
        DESCRIPTION.

    Returns
    -------
    event_df : TYPE
        DESCRIPTION.
    param_df : TYPE
        DESCRIPTION.

    """
    event_dt = {"time": [], "event": [], "pops": [], "value": []}
    param_dt = {"tbi": [], "time": [], "event": [], "pops": [], "value": []}
    condition_ls = []
    with open(in_file, 'r') as model:
        for line in model:
            if not line.startswith("#"):  # comment line
                if line.strip():
                    if line.startswith("tbi"):
                        x_lin = line.split()
                        if "#" in x_lin:  # comment in line
                            ix = x_lin.index("#")
                            x_lin = x_lin[:ix]
                        param_dt["tbi"].append(x_lin[0])
                        param_dt["event"].append(x_lin[1])
                        param_dt["pops"].append(x_lin[2].split("_"))
                        ix = [i for i, param in enumerate(x_lin) if "r" in param]
                        if len(ix) > 1:
                            param_dt["time"].append(x_lin[ix[0]:ix[0]+3])
                            param_dt["value"].append(x_lin[ix[1]:])
                        else:
                            param_dt["time"].append(x_lin[ix[0]:ix[0]+3])
                            param_dt["value"].append([])
                    elif line.startswith("cond"):
                        c_lin = line.split()
                        if "#" in c_lin:
                            ix = c_lin.index("#")
                            c_lin = c_lin[:ix]
                        condition_ls.append(c_lin[1:])
                    else:
                        y_lin = line.split()
                        if "#" in y_lin:
                            ix = y_lin.index("#")
                            y_lin = y_lin[:ix]
                        assert y_lin[0].isdigit(), "no tbi_, line must start with integer/float"
                        event_dt["time"].append(int(y_lin[0]))
                        event_dt["event"].append(y_lin[1])
                        event_dt["pops"].append(y_lin[2].split("_"))
                        event_dt["value"].append(list(map(float, y_lin[3:])))
    event_df = pd.DataFrame(event_dt, index=range(len(event_dt["time"])))
    # sort and draw tbi events
    param_dt = draw_params(param_dt, size, condition_ls)
    param_df = pd.DataFrame(param_dt, index=param_dt["tbi"])
    return event_df, param_df
