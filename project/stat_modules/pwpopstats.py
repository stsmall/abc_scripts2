#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 19:24:35 2020
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.
"""
import numpy as np
import allel
import moments.LD as mold
from itertools import product
from scipy import stats
from project.stat_modules.sequtils import get_seg, get_ac_seg, h2gt, pop2seg


def d_tajD(p1, pos, gt, win_size, length_bp):
    """Compute the difference in Tajima’s D between two populations in moving windows.

    Parameters
    ----------
    p1 : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    gt : TYPE
        DESCRIPTION.
    size : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    dtajd_mean : TYPE
        DESCRIPTION.
    dtajd_std : TYPE
        DESCRIPTION.

    """
    ac1, ac2, pos_s = get_ac_seg(p1, pos, gt)
    # segregating in both pops
    loc_asc = ac1.is_segregating() & ac2.is_segregating()
    ac1_seg = ac1.compress(loc_asc, axis=0)
    ac2_seg = ac2.compress(loc_asc, axis=0)
    pos_seg = pos_s[loc_asc]  # sites
    windows = int(length_bp/win_size)
    size = int(len(pos_seg) / windows)
    if size < 5:
        size = 5
    dtajd_ = allel.moving_delta_tajima_d(ac1_seg, ac2_seg, size=size)
    return dtajd_


def fst(p1, pos, gt, quants):
    """Calculate Hudson's FST.

    Hudson’s FST estimator as the ratio of averages computed following
    Bhatia et al. (2013).

    Parameters
    ----------
    p1 : TYPE
        DESCRIPTION.
    p2 : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    gt : TYPE
        DESCRIPTION.
    win_size : TYPE
        DESCRIPTION.
    length_bp : TYPE
        DESCRIPTION.

    Returns
    -------
    fst : TYPE
        DESCRIPTION.

    """
    ac1, ac2, pos_s = get_ac_seg(p1, pos, gt)
    # segregating in both pops
    loc_asc = ac1.is_segregating() & ac2.is_segregating()
    ac1_seg = ac1.compress(loc_asc, axis=0)
    ac2_seg = ac2.compress(loc_asc, axis=0)
    num, den = allel.hudson_fst(ac1_seg, ac2_seg)
    fst_snp = num / den
    fst_ = np.quantile(fst_snp, quants)
    return fst_


def dxy(p1, pos, gt, win_size, length_bp):
    """Calculate pairwise divergence between two populations.

    Parameters
    ----------
    p1 : int
        size of subpop1.
    pos : TYPE
        DESCRIPTION.
    gt : TYPE
        DESCRIPTION.
    win_size : TYPE
        DESCRIPTION.
    length_bp : TYPE
        DESCRIPTION.

    Returns
    -------
    dxy_win : TYPE
        DESCRIPTION.

    """
    ac1, ac2, pos_s = get_ac_seg(p1, pos, gt)
    dxy_win = allel.windowed_divergence(pos_s, ac1, ac2, size=win_size, start=1, stop=length_bp)

    return dxy_win[0]


def dmin(p1, pos, gt, win_size, length_bp):
    """Minimum of the pairwise divergence between pops."""
    gtseg, pos_s = get_seg(gt, pos)
    p1_ = range(p1)
    p2_ = range(p1, gtseg.shape[1])
    dmin_ls = []
    # method 2
    coords = (range(0, length_bp + win_size, win_size))
    windows = list(zip(coords[0::1], coords[1::1]))
    for s, e in windows:
        ix = pos_s.locate_range(s, e)
        dm = []
        gtpop2 = gtseg[ix, p2_]
        for i1 in p1_:
            p1_i = gtseg[ix, i1]
            dxy_ = p1_i[:, None] != gtpop2
            dm.append(min(np.count_nonzero(dxy_, axis=0)))
        dmin_ls.append(min(dm)/win_size)
    dmin_ = np.array(dmin_ls)

    return dmin_


def gmin(p1, pos, gt, win_size, length_bp):
    """Calculate minimum pairwise difference between two populations.

    dmin / dxy
    """
    dmin_ = dmin(p1, pos, gt, win_size, length_bp)
    dxy_ = dxy(p1, pos, gt, win_size, length_bp)
    gmin_win = dmin_/dxy_
    return gmin_win


def pi_fx(pos, gt, win_size, length_bp):
    """Diversity in one pop in a window."""
    ac = gt.count_alleles()
    pi_ = allel.windowed_diversity(pos, ac, size=win_size, start=1, stop=length_bp)

    return pi_[0]


def dd1_2(p1, pos, gt, win_size, length_bp, quants):
    """dmin/pi for each population.

    dd1=dmin/pi1
    dd2=dmin/pi2
    """
    dd12_ls = []
    dmin_ = dmin(p1, pos, gt, win_size, length_bp)
    gtseg, pos_s = get_seg(gt, pos)
    p1_ = range(p1)
    p2_ = range(p1, gtseg.shape[1])
    for p in [p1_, p2_]:
        gtpop = gt.take(p, axis=1)
        pi_ = pi_fx(pos, gtpop, win_size, length_bp)
        dd12_ls.extend(np.quantile((dmin_/pi_), quants))

    return dd12_ls


def pw_within(pos, gt, win_size, length_bp):
    """Pairwise divergence among all pairs in a gt matrix."""
    p_ix = gt.shape[1]
    gtseg, pos_s = get_seg(gt, pos)
    dxy_ls = []
    coords = (range(0, length_bp + win_size, win_size))
    windows = list(zip(coords[0::1], coords[1::1]))
    for s, e in windows:
        ix = pos_s.locate_range(s, e)
        dm = []
        gtpop1 = gtseg[ix, :]
        for i1 in range(p_ix):
            p1_i = gtpop1[:, i1]
            dxy_ = p1_i[:, None] != gtpop1
            pi_w = np.count_nonzero(dxy_, axis=0) / win_size
            dm.extend(np.sort(pi_w)[1:])
        dxy_ls.append(dm)
    pi_wn = np.array(dxy_ls)
    return pi_wn


def ddRank1_2(p1, pos, gt, win_size, length_bp, quants):
    """Rank percentile of the dmin value to the within pairwise divergence.

    ddRank1 = np.percentile(dmin, dxy_within_1)
    ddRank2 = np.percentile(dmin, dxy_within_2)
    """
    ddR12_ls = []
    dmin_ = dmin(p1, pos, gt, win_size, length_bp)
    gtseg, pos_s = get_seg(gt, pos)
    p1_ = range(p1)
    p2_ = range(p1, gtseg.shape[1])
    for p in [p1_, p2_]:
        gtpop = gt.take(p, axis=1)
        pw_win = pw_within(pos, gtpop, win_size, length_bp)
        ddR12 = [stats.percentileofscore(pw, dmin_[i]) for i, pw in enumerate(pw_win)]
        ddR12_ls.extend(np.quantile(ddR12, quants))
    return ddR12_ls


def ld_window(pos, hap, win_size, length_bp):
    """LD statistics within a given window."""
    D2 = 0
    pos, gt = h2gt(pos, hap)
    ld_wins = []
    coords = (range(0, length_bp + win_size, win_size))
    windows = list(zip(coords[0::1], coords[1::1]))
    for s, e in windows:
        gt_win = gt[:, (pos > s) & (pos <= e)]
        ld_win = mold.Parsing.compute_average_stats(gt_win.T)[D2]
        ld_wins.append(ld_win)

    return ld_wins


def zx(p1, pos, hap, win_size, length_bp):
    """FILET statistic.

    Zx = (Zn_s1 + Zn_s2)/(2*Zn_sg)
    Zn_s1 = avg LD all variants w/in window pop1
    Zn_s2 = avg LD all variants w/in window pop2
    Zn_sg = avg LD all variants w/in window pop1+pop2
    """
    # calc zn_sg
    p1_ = range(p1)
    p2_ = range(p1, hap.shape[0])
    pos, h1, h2 = pop2seg(p1_, p2_, pos, hap)
    zn_sg = np.array(ld_window(pos, np.concatenate([h1, h2], axis=0), win_size, length_bp))
    # calc Zns1 & Zns2
    zn1 = np.array(ld_window(pos, h1, win_size, length_bp))
    zn2 = np.array(ld_window(pos, h2, win_size, length_bp))
    zx_ = (zn1 + zn2) / (2 * zn_sg)

    return zx_


def ibs_max(pos, hap, dmax):
    """Compute the max ibs length for a subset of 2 haplotypes.

    Original Authors: Simon Boitard, Flora Jay
    """
    q = np.array([])
    d = np.zeros(shape=1, dtype='int32')
    pos_temp = pos
    data_temp = hap
    count_temp = np.sum(data_temp, axis=0)
    pos_temp = pos_temp[(count_temp > 0)*(count_temp < (2))]
    if len(pos_temp) > 1:
        d = np.concatenate((d, np.diff(pos_temp)))
    else:
        d = np.concatenate((d, dmax*np.ones(shape=1, dtype='int32')))
    d = np.minimum(d, dmax*np.ones(shape=len(d), dtype='int32'))
    q = stats.mstats.mquantiles(d[1:], prob=1.0, alphap=1, betap=1)
    return q[0]


def ibs_maxxy(p1, pos, hap, length_bp):
    """Compute the max value of IBS between 2 populations."""
    ibsxy = []
    p1_ = range(p1)
    p2_ = range(p1, hap.shape[0])
    for i12 in product(p1_, p2_):
        hap_12 = hap[i12, :]
        ibs = ibs_max(pos, hap_12, length_bp)
        ibsxy.append(ibs)
    return max(ibsxy)
