#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:35:46 2021
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

"""
import numpy as np
import moments.LD as mold
from project.stat_modules.sequtils import h2gt, pop2seg
import scipy.spatial.distance
import warnings


def ld_intervals(nb_times, tmax, r, length_bp):
    """Creation of the bins of physical distance for LD stats.

    Windows are choosen based on the time windows defined above.

    Parameters
    ----------
    nb_times : List
        number of time windows
    tmax : int
        the oldest time window will start at Tmax
    r : float
        recomb rate per generation per bp
    length_bp : int
        size of each segment, in bp

    Returns
    -------
    interval_list: List
        DESCRIPTION.

    """
    a = 0.06
    per_err = 5

    times = -np.ones(shape=nb_times, dtype='float')
    for i in range(nb_times):
        times[i] = (np.exp(np.log(1+a*tmax)*i/(nb_times-1))-1)/a
    interval_list = []
    for i in range(nb_times - 1):
        t = (times[i + 1] + times[i])/2
        d = 1/(2 * r * t)
        if d <= length_bp:
            interval_list.append([d - per_err * d/100.0, d + per_err * d/100.0])
    t = tmax + times[nb_times - 1] - times[nb_times - 2]
    # d = 10.0**8/(2.0 * t)
    d = 1/(2*r*t)
    interval_list.append([d-per_err * d/100.0, d + per_err * d/100.0])
    print(f"Average LD distance bins (in bp): {np.rint(interval_list)}")
    intervals = np.rint(interval_list)
    np.savetxt("ldintervals.txt", intervals)
    # np.loadtxt()
    return intervals


def ld_pop2(p1, pos, hap, quants, maf=0.05):
    """Calculate the LD statistics in intervals between 2 pops.

    Parameters
    ----------
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    intervals : TYPE
        DESCRIPTION.
    ld : TYPE
        DESCRIPTION.

    Returns
    -------
    ld_ls : TYPE
        DESCRIPTION.

    """
    D2 = 0
    pos, gt = h2gt(pos, hap, maf=0.05)
    p1_ = range(int(p1/2))
    p2_ = range(int(p1/2), gt.shape[0])
    pos, gt1, gt2 = pop2seg(p1_, p2_, pos, gt)
    # L1*L2 where Li is number of snps in gtpop{i}_seg
    pw_ld = mold.Parsing.compute_pairwise_stats_between(gt1.T, gt2.T)[D2]
    ld_q = np.quantile(pw_ld, quants)
    return ld_q


def ld_pop2_win(p1, pos, hap, win_size, length_bp, maf=0.05):
    """Calculate the LD statistics in intervals between 2 pops.

    Parameters
    ----------
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    intervals : TYPE
        DESCRIPTION.
    ld : TYPE
        DESCRIPTION.

    Returns
    -------
    ld_ls : TYPE
        DESCRIPTION.

    """
    D2 = 0
    pos, gt = h2gt(pos, hap, maf=0.05)
    p1_ = range(int(p1/2))
    p2_ = range(int(p1/2), gt.shape[0])
    pos, gt1, gt2 = pop2seg(p1_, p2_, pos, gt)
    # divide up windows evenly
    ld_wins = []
    coords = (range(0, length_bp + win_size, win_size))
    windows = list(zip(coords[0::1], coords[1::1]))
    for s, e in windows:
        gt1_win = gt1[:, (pos > s) & (pos <= e)]
        gt2_win = gt2[:, (pos > s) & (pos <= e)]
        ld_win = mold.Parsing.compute_average_stats_between(gt1_win.T, gt2_win.T)[D2]
        ld_wins.append(ld_win)
    return ld_wins


def ld_pop_mp(pos, hap, intervals, maf=0.05):
    """Calculate the LD statistics in intervals for 1 pop.

    If there are too many pw comparisons:
        1) randomly thin SNPs
        2) choose fewer individuals that will have fewer seg positions
        3)

    Parameters
    ----------
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    intervals : TYPE
        DESCRIPTION.
    ld : TYPE
        DESCRIPTION.

    Returns
    -------
    ld_ls : TYPE
        DESCRIPTION.

    """
    D2 = 0  # D2
    pos, gt = h2gt(pos, hap, maf=maf)
    ld_ls = []
    c2 = pos[:, None]
    pw_dist = scipy.spatial.distance.pdist(c2, 'cityblock')
    pw_ld = mold.Parsing.compute_pairwise_stats(gt.T)[D2]
    for dmin, dmax in intervals:
        ld_interval = pw_ld[(pw_dist >= dmin) & (pw_dist <= dmax)]
        ld_mean = np.mean(ld_interval)
        ld_ls.append(ld_mean)
    return ld_ls


def ld_pop_mb(pos, hap, intervals, maf=0.05):
    """Calculate D2 in intervals.

    Uses the method in Boitard 2015 to get positions
    """
    D2 = 0
    pos, gt = h2gt(pos, hap, maf=maf)
    ld_ls = []
    p = len(intervals)
    for i in range(0, p):
        r2_list = []
        dmin = intervals[i][0]
        dmax = intervals[i][1]
        # looks for snp pairs with the good distance
        nb_snp = len(pos)
        if nb_snp > 0:
            i_deb = 0
            i_fin = 1
            while i_fin < nb_snp:
                while i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) < dmin:
                    i_fin += 1
                if i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) <= dmax:
                    # compute D2
                    gtld = gt[:, [i_deb, i_fin]]
                    r2_list.append(mold.Parsing.compute_average_stats(gtld.T)[D2])
                i_deb = i_fin+1  # =+1
                i_fin = i_deb+1
        ld_ls.append(np.mean(np.array(r2_list, dtype='float')))
    return ld_ls


def r2(u, v):
    """Return the r2 value for two haplotype vectors.

    (numpy arrays with alleles coded 1 and 0)

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    fcross = np.mean(u*v)
    fu = np.mean(u)
    fv = np.mean(v)
    return (fcross-fu*fv)**2/(fu*(1-fu)*fv*(1-fv))


def distrib_r2(pos, hap, interval_list):
    """Return the mean and the variance of r2 for a list of distance intervals.


    Parameters
    ----------
    pos : TYPE
        pos_list is a list of 1 dim arrays
    hap : TYPE
        hap_list is a list of 2 dim arrays
    interval_list : TYPE
        a subset of non overlapping pairs is used for each interval

    Returns
    -------
    moy : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.

    """
    p = len(interval_list)
    moy = -np.ones(shape=p, dtype='float')
    var = -np.ones(shape=p, dtype='float')
    for i in range(0, p):
        r2_list = []
        dmin = interval_list[i][0]
        dmax = interval_list[i][1]
        # looks for snp pairs with the good distance
        nb_snp = len(pos)
        if nb_snp > 0:
            i_deb = 0
            i_fin = 1
            while i_fin < nb_snp:
                while i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) < dmin:
                    i_fin += 1
                if i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) <= dmax:
                    # compute r2
                    breakpoint()
                    u_deb = hap[:, i_deb]
                    u_fin = hap[:, i_fin]
                    r2_list.append(r2(u_deb, u_fin))
                i_deb = i_fin+1
                i_fin = i_deb+1
        if len(r2_list) < 2:
            # try a more exhaustive screening of SNP pairs
            r2_list = []
            dmin = interval_list[i][0]
            dmax = interval_list[i][1]
            nb_snp = len(pos)
            if nb_snp > 0:
                i_deb = 0
                i_fin = 1
                while i_fin < nb_snp:
                    while i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) < dmin:
                        i_fin += 1
                    if i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) <= dmax:
                        # compute r2
                        u_deb = hap[:, i_deb]
                        u_fin = hap[:, i_fin]
                        r2_list.append(r2(u_deb, u_fin))
                    i_deb += 1
                    i_fin = i_deb+1
        # computes the stat
        if len(r2_list) >= 2:
            moy[i] = np.mean(np.array(r2_list, dtype='float'))
            var[i] = np.std(np.array(r2_list, dtype='float'))
    return moy, var


def zyg_r2(u, v):
    """Return the zygotic r2 value for two genotype vectors.

    (numpy arrays with genotypes coded 0, 1 and 2)

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return (np.corrcoef(u, v)[0, 1])**2


def distrib_zyg_r2(pos, geno, interval_list):
    """Return the mean and the variance of zygotic r2 for a list of distance intervals.

    Parameters
    ----------
    pos : TYPE
        pos_list is a list of 1 dim arrays
    geno : TYPE
        geno_list is a list of 2 dim arrays
    interval_list : TYPE
        a subset of non overlapping pairs is used for each interval

    Returns
    -------
    moy : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.

    """
    warnings.simplefilter("error", RuntimeWarning)
    p = len(interval_list)
    moy = -np.ones(shape=p, dtype='float')
    var = -np.ones(shape=p, dtype='float')
    for i in range(0, p):
        r2_list = []
        dmin = interval_list[i][0]
        dmax = interval_list[i][1]
        nb_snp = len(pos)
        if nb_snp > 0:
            i_deb = 0
            i_fin = 1
            while i_fin < nb_snp:
                while i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) < dmin:
                    i_fin += 1
                if i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) <= dmax:
                    # compute r2
                    u_deb = geno[:, i_deb]
                    u_fin = geno[:, i_fin]
                    try:
                        r2_list.append(zyg_r2(u_deb, u_fin))
                    except RuntimeWarning:
                        pass
                i_deb = i_fin+1
                i_fin = i_deb+1
        if len(r2_list) < 2:
            # try a more exhaustive screening of SNP pairs
            r2_list = []
            dmin = interval_list[i][0]
            dmax = interval_list[i][1]
            nb_snp = len(pos)
            if nb_snp > 0:
                i_deb = 0
                i_fin = 1
                while i_fin < nb_snp:
                    while i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) < dmin:
                        i_fin += 1
                    if i_fin < nb_snp and (pos[i_fin]-pos[i_deb]) <= dmax:
                        # compute r2
                        u_deb = geno[:, i_deb]
                        u_fin = geno[:, i_fin]
                        try:
                            r2_list.append(zyg_r2(u_deb, u_fin))
                        except RuntimeWarning:
                            pass
                    i_deb += 1
                    i_fin = i_deb+1
        if len(r2_list) >= 2:
            moy[i] = np.mean(np.array(r2_list, dtype='float'))
            var[i] = np.std(np.array(r2_list, dtype='float'))
    return moy, var
