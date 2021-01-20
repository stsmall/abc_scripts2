# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 23:18:58 2020
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

"""

import numpy as np
from project.stat_modules.msformat import discrete_positions


def add_seq_error(pos, haps, length_bp, perfixder):
    # errors at polymorphic sites
    n_haps = haps.shape[0]
    seg_err = np.random.binomial(1, np.random.uniform(0, 0.00002), size=[n_haps, len(pos)])
    num_seg_errors = sum(seg_err)
    # print(np.sum(num_seg_errors))
    haps = haps - seg_err
    # 0 - 0, 0 - 1 error to -1 (to derived), 1 - 1  error to 0 (to ancestral), 1 - 0
    haps[haps == -1] = 1
    # errors at monomorphic sites
    monomorphic = int(length_bp) - len(pos)
    fix_derived = np.rint(monomorphic * perfixder)
    mon_err = np.random.binomial(1, np.random.uniform(0, 0.00002), size=[n_haps, monomorphic])
    # add fix derived
    der_err = np.random.binomial(fix_derived, np.random.uniform(0, 0.00002))
    rfix_pos = np.random.choice(range(monomorphic), der_err, replace=False)
    mon_err[:, rfix_pos] += 1
    mon_err[mon_err == 2] = 0
    counts = np.sum(mon_err, axis=0)
    err_mask = (counts > 0) & (counts < haps.shape[0])
    mon_err = mon_err[:, err_mask]
    num_mon_errors = np.sum(err_mask)
    # get pos and insert ix
    mon_pos = list(set(range(length_bp)) - set(pos))
    mon_pos_arr = np.random.choice(mon_pos, num_mon_errors, replace=False)
    mon_pos_arr = np.array(list(mon_pos_arr) + list(pos))
    mon_pos_ix = np.argsort(np.argsort(mon_pos_arr))[0:num_mon_errors]
    pos = np.array(sorted(mon_pos_arr))
    # insert mono errors into haps
    for i, ix in enumerate(mon_pos_ix):
        try:
            haps = np.insert(haps, ix, mon_err[:, i], axis=1)
        except IndexError:
            e = haps.shape[1]
            haps = np.insert(haps, e, mon_err[:, i], axis=1)

    return pos, haps


def add_seqerror(pos, haps, length_bp, pfe, seq_error):
    if seq_error:
        pos, haps = add_seq_error(pos, haps, length_bp, pfe)

    counts = haps.sum(axis=0).astype(int)

    return pos, haps, counts


def read_trees(ts, length_bp, pfe, seq_error):
    pos = np.array([variant.site.position for variant in ts.variants()])
    pos = pos.astype(int)
    haps = ts.genotype_matrix().T
    breakpoints = np.array(list(ts.breakpoints()))
    # gt_list.append(ts.genotype_matrix())
    if seq_error:
        pos, haps = add_seq_error(pos, haps, length_bp, pfe)
    counts = haps.sum(axis=0).astype(int)

    return pos, haps, counts, breakpoints


def read_ms(msfiles, msexe, nhaps, length_bp):
    ms_dt = {}
    pos_ls = []
    hap_ls = []
    i = 1
    for msfile in msfiles:
        with open(msfile) as ms:
            next(ms)  # skip the first header
            for line in ms:
                if line.startswith(msexe):
                    ms_dt[i] = (pos_ls, hap_ls)
                    pos_ls = []
                    hap_ls = []
                    i += 1
                if line.startswith("positions"):
                    positions = line.strip().split()
                    pos_arr = np.array(positions[1:], dtype=np.float64)
                    new_pos = discrete_positions(pos_arr, length_bp)
                    # haps line
                    hap_arr = np.zeros((nhaps, pos_arr.shape[0]), dtype=np.uint8)
                    for cix in range(nhaps):
                        line = next(ms)
                        line = list(line.strip())
                        hap_arr[cix, :] = np.array(line, dtype=np.uint8)
                    pos_ls.append(new_pos)
                    hap_ls.append(hap_arr)

    return ms_dt


def read_ms_stream(output, nhaps, length_bp, pfe, seq_error):
    pos_ls = []
    hap_ls = []
    ms_it = iter(output.splitlines())
    for line in ms_it:
        if line.startswith(b"positions"):
            line = line.decode()
            positions = line.strip().split()
            pos_arr = np.array(positions[1:], dtype=np.float64)
            new_pos = discrete_positions(pos_arr, length_bp)
            # haps line
            hap_arr = np.zeros((nhaps, pos_arr.shape[0]), dtype=np.uint8)
            for cix in range(nhaps):
                line = next(ms_it).decode()
                line = list(line.strip())
                hap_arr[cix, :] = np.array(line, dtype=np.uint8)
            pos_ls.append(new_pos)
            hap_ls.append(hap_arr)

    if seq_error:
        pos_err = []
        hap_err = []
        for pos, hap in zip(pos_ls, hap_ls):
            pos_, hap_ = add_seq_error(pos, hap, length_bp, pfe)
            pos_err.append(pos_)
            hap_err.append(hap_)
        count_err = [hap.sum(axis=0).astype(int) for hap in hap_err]
        return pos_err, hap_err, count_err

    count_ls = [hap.sum(axis=0).astype(int) for hap in hap_ls]
    return pos_ls, hap_ls, count_ls


def get_seg(gt, pos, maf=0):
    """Retain only sites and positions that are segregating in the sample.

    Parameters
    ----------
    gt : allel.HaplotypeArray
        DESCRIPTION.
    pos : allel.SortedIndex
        DESCRIPTION.

    Returns
    -------
    gtseg : TYPE
        DESCRIPTION.
    pos_s : TYPE
        DESCRIPTION.

    """
    acpop = gt.count_alleles()
    freq = acpop.to_frequencies()
    freq_mask = (freq[:, 1] > maf) & (freq[:, 1] < 1)
    gtseg = gt.compress(freq_mask)
    pos_s = pos[freq_mask]

    return gtseg, pos_s


def get_ac_seg(p1, pos, gt, maf=0):
    """Select that are segregating in both populations.

    Parameters
    ----------
    p1 : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    gt : TYPE
        DESCRIPTION.

    Returns
    -------
    ac1 : TYPE
        DESCRIPTION.
    ac2 : TYPE
        DESCRIPTION.
    pos_s : TYPE
        DESCRIPTION.

    """
    acpop = gt.count_alleles()
    freq = acpop.to_frequencies()
    freq_mask = (freq[:, 1] > maf) & (freq[:, 1] < 1)
    gt = gt.compress(freq_mask)
    pos = pos[freq_mask]
    # select subpops and count alleles
    gtseg, pos_s = get_seg(gt, pos)
    p1_ = range(p1)
    p2_ = range(p1, gtseg.shape[1])
    gtpop1 = gtseg.take(p1_, axis=1)
    gtpop2 = gtseg.take(p2_, axis=1)
    ac1 = gtpop1.count_alleles()
    ac2 = gtpop2.count_alleles()

    return ac1, ac2, pos_s


def h2gtr(hap):
    """Transform a list of haplotypes into a list of genotypes.

    pairs of haplotypes are RANDOMLY sampled for each chromosome


    Parameters
    ----------
    hap : TYPE
        DESCRIPTION.

    Returns
    -------
    geno_ls : TYPE
        DESCRIPTION.

    """
    geno_ls = []
    n = hap.shape[0]
    p = hap.shape[1]
    permut = np.random.permutation(n)
    geno = -np.ones(shape=(n/2, p), dtype='int32')
    for i in range(n/2):
        geno[i, :] = hap[permut[2*i], :]+hap[permut[2*i+1], :]
    geno_ls.append(geno)

    return geno_ls


def h2gt(pos, hap, maf=0):
    """Transform a list of haplotypes into a list of genotypes."""
    nhaps = hap.shape[0]
    mac = nhaps * maf
    mac_mask = (np.sum(hap, axis=0) > mac) & (np.sum(hap, axis=0) < nhaps)
    hap = hap[:, mac_mask]
    pos = pos[mac_mask]
    gt = hap[0::2, :]+hap[1::2, :]
    return pos, gt


def pop2seg(p1, p2, pos, hap, maf=0):
    """Keep sites that are segregating in 2 populations."""
    nhaps_1 = len(p1)
    mac1 = nhaps_1 * maf
    nhaps_2 = len(p2)
    mac2 = nhaps_2 * maf
    gtp1 = hap[p1, :]
    gtp2 = hap[p2, :]
    # segregating in both pops
    gtp1_mask = (np.sum(gtp1, axis=0) > mac1) & (np.sum(gtp1, axis=0) < nhaps_1)
    gtp2_mask = (np.sum(gtp2, axis=0) > mac2) & (np.sum(gtp2, axis=0) < nhaps_2)
    loc_asc = gtp1_mask * gtp2_mask
    gtp1_seg = gtp1[:, loc_asc]
    gtp2_seg = gtp2[:, loc_asc]
    pos = pos[loc_asc]
    return pos, gtp1_seg, gtp2_seg
