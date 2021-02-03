#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:27:35 2021
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

"""
import numpy as np
import allel
from .sequtils import get_seg


def spatial_histo_fast(pos, count, M, dmax=np.inf):
    """Compute the site frequency spectrum.

    Fast version of spatial_histo
    Note: This is the correct implementation of dist
    Author Flora Jay

    Parameters
    ----------
    pos : TYPE
        DESCRIPTION.
    count : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    dmax : TYPE, optional
        DESCRIPTION. The default is np.inf.

    Returns
    -------
    sfs : TYPE
        the site frequency spectrum from 1 to M (percentages, sum to 1)
    dist : TYPE
        the variance of the distance between sites with count i, for all i.
            This variance needs to be multiplied by the overall proportion of SNPs.
            positions of vector pos are assumed to be SORTED within each chromosome

    """
    histo = np.zeros(shape=M, dtype='float')
    d = [[] for i in range(1, M+1)]
    posfq = [[] for i in range(1, M+1)]
    for snp in range(pos.shape[0]):
        i = count[snp].astype(int)
        try:
            histo[i-1] += 1
            posfq[i-1].append(pos[snp])
        except IndexError:
            continue
    [d[i-1].append(x) for i in range(1, M+1) for x in np.diff(posfq[i-1]) if x <= dmax]
    dist = np.asarray([np.std(d_at_freq) if len(d_at_freq) > 1 else 0.0 for d_at_freq in d])
    sfs = histo/np.sum(histo)

    return dist


def asfs_stats(gt, pos, fold):
    """Calculate the allele frequence spectrum.

    With many individuals the SFS becomes unwieldy, here I collapse the
    intermediate frequencies into 1 bin. This differs from the one above by
    randomly sampling to reduce linkage.

    Future implementations will utilize the breakpoints from msprime tree object
    to find unlinked positions.

    Parameters
    ----------
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    fold : bool
        if True, return folded SFS

    Returns
    -------
    sfs : TYPE
        DESCRIPTION.

    """
    gtseg, pos_s = get_seg(gt, pos)
    # sfs
    if fold:
        sfsp = (allel.sfs_folded(gtseg.count_alleles(), gtseg.shape[1]))[1:]
    else:
        sfsp = (allel.sfs(gtseg.count_alleles()[:, 1], gtseg.shape[1]))[1:-1]
    tots = np.sum(sfsp)
    sfs = sfsp / tots

    return sfs


def summarizejsfs(fs):
    """Create summary of the jsfs.

    Parameters
    ----------
    fs : TYPE
        DESCRIPTION.

    Returns
    -------
    props : TYPE
        DESCRIPTION.

    """
    # # summarize
    jsfsarray = np.zeros(23, dtype=int)
    jsfsarray[0] = np.sum(fs[0, 1:3])
    jsfsarray[1] = np.sum(fs[1:3, 0])
    jsfsarray[2] = np.sum(fs[0, 3:-3])
    jsfsarray[3] = np.sum(fs[3:-3, 0])
    jsfsarray[4] = np.sum(fs[0, -3:-1])
    jsfsarray[5] = np.sum(fs[-3:-1, 0])
    jsfsarray[6] = np.sum(fs[1:3, 1:3])
    jsfsarray[7] = np.sum(fs[1:3, 3:-3])
    jsfsarray[8] = np.sum(fs[3:-3, 1:3])
    jsfsarray[9] = np.sum(fs[-3:-1, 3:-3])
    jsfsarray[10] = np.sum(fs[3:-3, -3:-1])
    jsfsarray[11] = np.sum(fs[1:3, -3:-1])
    jsfsarray[12] = np.sum(fs[-3:-1, 1:3])
    jsfsarray[13] = np.sum(fs[3:-3, 3:-3])
    jsfsarray[14] = np.sum(fs[-3:-1, -3:-1])
    jsfsarray[15] = np.sum(fs[0, -1])
    jsfsarray[16] = np.sum(fs[-1, 0])
    jsfsarray[17] = np.sum(fs[-1, 1:3])
    jsfsarray[18] = np.sum(fs[1:3, -1])
    jsfsarray[19] = np.sum(fs[-1, 3:-3])
    jsfsarray[20] = np.sum(fs[3:-3, -1])
    jsfsarray[21] = np.sum(fs[-1, -3:-1])
    jsfsarray[22] = np.sum(fs[-3:-1, -1])
    jsfstotal = np.sum(jsfsarray)
    props = jsfsarray/jsfstotal
    return props


def jsfs_stats(p1, gt, pos, fold):
    """Calculate the joint site frequency spectrum between two populations.

    Parameters
    ----------
    p1 : TYPE
        DESCRIPTION.
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    fold : TYPE
        DESCRIPTION.
    rand : TYPE
        DESCRIPTION.
    randn : TYPE
        DESCRIPTION.

    Returns
    -------
    props : TYPE
        DESCRIPTION.

    """
    gtr, pos_s = get_seg(gt, pos)
    gtpop1 = gtr.take(range(p1), axis=1)
    gtpop2 = gtr.take(range(p1, gtr.shape[1]), axis=1)
    ac1 = gtpop1.count_alleles()
    ac2 = gtpop2.count_alleles()
    # jsfs
    if fold:
        # pad for allel as well
        #popsizeA, popsizeB = p1/2, (gtr.shape[1]-p1)/2
        jsfs = allel.joint_sfs_folded(ac1, ac2, gtpop1.shape[1], gtpop2.shape[1])
        #fss = np.resize(jsfs, (int(popsizeA)+1, int(popsizeB)+1))
    else:
        # pad for allel as well
        #popsizeA, popsizeB = p1, gtr.shape[1]-p1
        jsfs = allel.joint_sfs(ac1[:, 1], ac2[:, 1], gtpop1.shape[1], gtpop2.shape[1])
        #fss = np.resize(jsfs, (int(popsizeA)+1, int(popsizeB)+1))
    props = summarizejsfs(jsfs)

    return props