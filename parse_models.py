#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:31:12 2020
@author: Scott T. Small

module for generating draws from priors for filet_sims.py
"""

import pandas as pd
import numpy as np
import Ddist


def draw_params(param_dt, size, condition_ls):
    """Draws parameters from dist of lenght size.

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
    # TODO: recursion
    time_ls = []
    value_ls = []
    tbi_dt = {}

    # set up tbi dict
    for i, tbi in enumerate(param_dt["tbi"]):
        tbi_dt[tbi] = param_dt["time"][i]
    # set up last w/ not tbi in low/high
    dist, low, high = tbi_dt[f"tbi{i}"]
    draw = getattr(Ddist, dist[1:])
    tbi_dt[f"tbi{i}"] = draw(float(low), float(high), size)
    # fill in remaining
    i -= 1  # interate through in reverse
    while i >= 0:
        dist, low, high = tbi_dt[f"tbi{i}"]
        draw = getattr(Ddist, dist[1:])
        if "tbi" in low or "tbi" in high:
            if "tbi" in low and "tbi" in high:
                low = tbi_dt[low]
                high = tbi_dt[high]
                size_n = 1
            elif "tbi" in low:
                low = tbi_dt[low]
                high = np.full(size, float(high))
                size_n = 1
            elif "tbi" in high:
                high = tbi_dt[high]
                low = np.full(size, float(low))
                size_n = 1
            try:
                tbi_dt[f"tbi{i}"] = draw(low, high, size_n)
            except ValueError:
                assert low == high
                # low == high, ensure both events happen at the same time
                tbi_dt[f"tbi{i}"] = tbi_dt[low]
        else:
            tbi_dt[f"tbi{i}"] = draw(float(low), float(high), size)
        i -= 1
    # check conditions
    if condition_ls:
        for condition in condition_ls:
            tbi_1, cond, tbi_2 = condition
            for t1, t2 in zip(tbi_dt[tbi_1], tbi_dt[tbi_2]):
                if cond == "lt":
                    assert t1 < t2
                elif cond == "gt":
                    assert t1 > t2
    # fill to one list
    time_ls = [tbi_dt[tbi] for tbi in param_dt["index"]]
    param_dt["time"] = time_ls
    # value should have no tbi
    for tvalue in param_dt["value"]:
        dist, low, high = tvalue
        draw = getattr(Ddist, dist[1:])
        value_ls.append(draw(float(low), float(high), size))
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
    event_dt = {"time": [], "event": [], "pop": [], "value": []}
    param_dt = {"index": [], "time": [], "event": [], "pop": [], "value": []}
    condition_ls = []
    with open(in_file, 'r') as model:
        for line in model:
            if not line.startswith("#"):
                if line.strip():
                    if line.startswith("tbi"):
                        x_lin = line.split()
                        param_dt["tbi"].append(x_lin[0])
                        param_dt["event"].append(x_lin[1])
                        param_dt["pop"].append(list(x_lin[2]))
                        if x_lin[1].startswith("t"):  # more than one dist keyword
                            ix = [i for i, param in enumerate(x_lin) if "r" in param]
                            param_dt["time"].append(x_lin[ix[0]:ix[0]+3])  # this is distribution keyword
                            param_dt["value"].append(x_lin[ix[1]:])  # this is low and high
                        else:
                            param_dt["time"].append(x_lin[3:])  # this is distribution keyword
                            param_dt["value"].append(None)  # this is low and high
                    elif line.startswith("cond"):
                        c_lin = line.split()
                        condition_ls.append(c_lin[1:])
                    else:
                        y_lin = line.split()
                        assert y_lin[0].isdigit(), "no tbi_, line must start with integer/float"
                        event_dt["time"].append(int(y_lin[0]))
                        event_dt["event"].append(y_lin[1])
                        event_dt["pop"].append(list(y_lin[2]))
                        event_dt["value"].append(map(float, y_lin[3:]))

    event_df = pd.DataFrame(event_dt, index=range(len(event_dt["time"])))
    # sort and draw tbi events
    param_dt = draw_params(param_dt, size, condition_ls)
    param_df = pd.DataFrame(param_dt, index=param_dt["tbi"])

    return event_df, param_df
