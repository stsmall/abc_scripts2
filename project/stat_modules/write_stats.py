#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:36:26 2020
@author: Scott T. Small

"""
from itertools import combinations
from math import floor


def stats_out(stats_arr, out_file, nprocs):
    """Write stats to out file."""
    if nprocs > 1:
        for stat in stats_arr:
            rd = [round(num, 5) for num in stat]
            out_file.write("{}\n".format("\t".join(map(str, rd))))
    else:
        rd = [round(num, 5) for num in stats_arr]
        out_file.write("{}\n".format("\t".join(map(str, rd))))

    return out_file


def headers(out_file, stats_dt):
    stats_list = stats_dt["calc_stats"]

    try:
        prob, sizes = [len(i) for i in stats_dt["ibs_params"]]
    except KeyError:
        prob, sizes = 0, 0, 0
    try:
        intervals = len(stats_dt["ld_params"])
    except KeyError:
        intervals = 0

    sub_pops = [len(i) for i in stats_dt["pop_config"]]
    n_pops = len(sub_pops)
    quants = len(stats_dt["pw_quants"])
    sfsfold, sfsagg = stats_dt["sfs_fold_agg"]
    spatfold, spatagg = stats_dt["spat_fold_agg"]
    afibsfold = stats_dt["afibs_fold"]
    sfs_n = [sub-1 for sub in sub_pops]
    if sfsfold:
        sfs_n = [floor((sub)/2) for sub in sub_pops]
    if sfsagg:
        sfs_n = [6] * n_pops
    spat_n = [sub-1 for sub in sub_pops]
    if spatfold:
        spat_n = [floor((sub)/2) for sub in sub_pops]
    if spatagg:
        spat_n = [6] * n_pops
    afibs_n = [sub-2 for sub in sub_pops]
    if afibsfold:
        afibs_n = [floor((sub-1)/2) for sub in sub_pops]
    pops_dt = {"pi": [1]*n_pops,
               "tajd": [1]*n_pops,
               "hap_het": [1]*n_pops,
               "afibs": afibs_n,
               "spatial_sfs": spat_n,
               "ld": [intervals]*n_pops,
               "sfs": sfs_n,
               "ibs": [prob*sizes]*n_pops
               }
    cross_dt = {"dXY": quants,
                "dmin": quants,
                "FST": quants,
                "gmin": quants,
                "IBS_maxXY": 1,
                "Zx": quants,
                "jsfs": 23,
                "ld_2pop": quants,
                "ddRank12": quants,
                "dd12": quants,
                "delta_tajD": quants
                }
    header = []
    for p in stats_list:
        if p in pops_dt:
            for p_ix in range(0, n_pops):
                s_ix = pops_dt[p][p_ix]
                if "pi" in p or "tajd" in p or "hap_het" in p:
                    header.append(f"{p}_pop{p_ix}")
                    header.append(f"{p}std_pop{p_ix}")
                else:
                    if "afibs" in p:
                        header.extend([f"{p}{i}_pop{p_ix}" for i in range(1, s_ix+1)])
                        header.extend([f"{p}std{i}_pop{p_ix}" for i in range(1, s_ix+1)])
                    else:
                        header.extend([f"{p}{i}_pop{p_ix}" for i in range(1, s_ix+1)])
        elif p in cross_dt:
            s_ix = cross_dt[p]
            for i, j in combinations(range(n_pops), 2):
                if s_ix == 1:
                    header.append(f"{p}_pop{i}{j}")
                elif "dd12" in p or "ddRank12" in p:
                    header.extend([f"{p}_{k}_pop{i}" for k in range(1, s_ix+1)])
                    header.extend([f"{p}_{k}_pop{j}" for k in range(1, s_ix+1)])
                else:
                    header.extend([f"{p}{k}_pop{i}{j}" for k in range(1, s_ix+1)])
        else:
            breakpoint()
            print("stat not found")
    out_file.write("{}\n".format("\t".join(header)))
    return out_file, len(header), header
