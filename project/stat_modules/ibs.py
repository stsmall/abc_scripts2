#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:40:00 2021
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

"""

from scipy import stats
import numpy as np


def ibs_quantiles_from_data(m, pos, hap, prob, dmax, data_type=1, moments=False):
    """Compute the quantiles of the ibs length distribution.

    WARNING: Set data_type to 1 if haplotypes, to 2 if genotypes
    if m==1 and data_type==2 this corresponds to ROH
    Authors: Simon Boitard, Flora Jay

    Parameters
    ----------
    m : TYPE
        nb of haplotypes or genotypes to randomly subsample
    pos : TYPE
        positions of snps
    hap : TYPE
        haplotypes or genotypes
    prob : TYPE
        vector of probabilities for which quantiles are computed
    dmax : TYPE
        maximum length of ibs (eg. length of the segments)
    data_type : TYPE, optional
        if 1 data are haplotypes if 2 genotypes
    moments : bool
        compute moments of ibs-length distribution?

    Returns
    -------
    q : TYPE
        quantiles
    moms: List
        moments 1 to 4th of ibs-length distrib

    """
    q = np.array([])
    # builds a ibs length sample
    d = np.zeros(shape=1, dtype='int32')
    pos_temp = pos
    data_temp = hap
    n = data_temp.shape[0]
    if m < n:
        subset = np.random.choice(n, size=m, replace=False)
        count_temp = np.sum(data_temp[subset, ], axis=0)
        pos_temp = pos_temp[(count_temp > 0)*(count_temp < (data_type*m))]
    if len(pos_temp) > 1:
        d = np.concatenate((d, np.diff(pos_temp)))
    else:
        d = np.concatenate((d, dmax*np.ones(shape=1, dtype='int32')))
    d = np.minimum(d, dmax*np.ones(shape=len(d), dtype='int32'))
    # computes the quantiles and/or the moments of this sample
    if moments:
        moms = -np.ones(shape=4)
        moms[0] = np.mean(d[1:])
        moms[1] = np.std(d[1:])
        #  mom3:skewness, mom4:kurtosis
        for m1 in range(3, 5):
            moms[m1-1] = np.mean(np.power((d[1:]-moms[0])/moms[1], m1))
        return moms
    q = stats.mstats.mquantiles(d[1:], prob=prob, alphap=1, betap=1)
    return q


def distrib_ibs(pos, distance_list, dmax=np.inf):
    """Probability of ibs exceeding a given distance for a list of distances.

    Parameters
    ----------
    pos : TYPE
        DESCRIPTION.
    distance_list : list
        distances to compute probability
    dmax : TYPE, optional
        DESCRIPTION. The default is np.inf.

    Returns
    -------
    cdf : TYPE
        DESCRIPTION.

    """
    # builds a ibs length sample
    d = np.zeros(shape=1, dtype='int32')
    pos_temp = pos
    d_temp = pos_temp[1:]-pos_temp[:(len(pos_temp)-1)]
    d = np.concatenate((d, d_temp))
    if not dmax == np.inf:
        d = np.minimum(d, dmax*np.ones(shape=len(d), dtype='int32'))
    # computes the ecdf of this sample
    p = len(distance_list)
    cdf = -np.ones(shape=p, dtype='float')
    if len(d) > 1:
        d = d[1:]
        for i in range(0, p):
            sel = (d >= distance_list[i])
            cdf[i] = sum(sel)
        cdf = cdf/len(d)

    return cdf


def break_chr(pos_list, hap_list, dmax=2000000):
    """Break a list of long chromosomes into an equivalent list chromosomes.

    To be used before ibs_quantiles in the case of real data sets with unequal
    chromosomes lengths.

    Parameters
    ----------
    pos_list : TYPE
        DESCRIPTION.
    hap_list : TYPE
        DESCRIPTION.
    dmax : TYPE, optional
        DESCRIPTION. The default is 2000000.

    Returns
    -------
    pos_list_new : TYPE
        DESCRIPTION.
    hap_list_new : TYPE
        DESCRIPTION.

    """
    pos_list_new = []
    hap_list_new = []
    for chro in range(0, len(pos_list)):
        print(f"breaking chromosome {chro}")
        pos_temp = pos_list[chro]
        hap_temp = hap_list[chro]
        outlier_ind = (pos_temp > dmax)
        while np.sum(outlier_ind) > 0:
            if np.prod(outlier_ind) == 0:
                pos_list_new.append(pos_temp[np.logical_not(outlier_ind)])
                hap_list_new.append(hap_temp[:, np.logical_not(outlier_ind)])
            pos_temp = pos_temp[outlier_ind]-dmax
            hap_temp = hap_temp[:, outlier_ind]
            outlier_ind = (pos_temp > dmax)
        pos_list_new.append(pos_temp)
        hap_list_new.append(hap_temp)

    return pos_list_new, hap_list_new
