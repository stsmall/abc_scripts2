#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:44:25 2021
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

"""

import numpy as np


def spatial_histo_fast_unpol(polarized, pos, counts, M, dmax=np.inf):
    """Fast version of spatial_histo.

    This is the correct implementation of dist
    Author Flora Jay

    Parameters
    ----------
    polarized : np.array[Nsnp_seg]
        polarized or not? for SNP for each segment
    pos : np.array[Nsnp_seg]
        positions of SNP for each segment
    counts : np.array[Nsnp_seg]
        number of derived alleles at each position for each segment
    M : TYPE
        DESCRIPTION.
    dmax : TYPE, optional
        DESCRIPTION. The default is np.inf.

    Returns
    -------
    usfs : TYPE
        the site frequency spectrum from 1 to M (percentages, sum to 1).
    dist : TYPE
        the variance of the distance between sites with count i, for all i.
        This variance needs to be multiplied by the overall proportion of SNPs.
        positions of vector pos are assumed to be SORTED within each chromosome

    """
    histo = np.zeros(shape=M, dtype='float')
    nb_snp = 0
    d = [[] for i in range(1, M+1)]
    posfq = [[] for i in range(1, M+1)]
    for snp in range(pos.shape[0]):
        if not polarized[snp]:
            continue
        i = counts[snp]
        try:
            histo[i-1] += 1
            posfq[i-1].append(pos[snp])
        except IndexError:
            continue

    [d[i-1].append(x) for i in range(1, M+1) for x in np.diff(posfq[i-1]) if x <= dmax]

    # for each frequency, compute the std of the distance list, after removing distance longer than dmax
    # if no distances deviation set to 0 (was -1 before)
    dist = np.asarray([np.std(d_at_freq) if len(d_at_freq) > 1 else 0.0 for d_at_freq in d])
    usfs = histo/np.sum(histo)

    return usfs, dist


def afibs_fast_unpol(polarized, data, posOnChrom, counts, afibs=None):
    """Compute AF-IBS as defined by Theunert et al. 2012.

    An alternative version based on PBWT algorithm is also available, that works
    for large sample size execution time was ~ 2 s for 2Mb segment 20 haplotypes
    faster than a regular search it has not been tested for a larger number of
    individuals.

    Parameters
    ----------
    data : bool np.array
        DESCRIPTION.
    posOnChrom :  int array
        DESCRIPTION.
    counts : list(list)
        list containing for each frequency f the list of lengths of afibs tracts
        around derived alleles at freq. f (eg afibs[2] contains list of len for
        sites with 2 derived alleles) for segments already analysed
    afibs : list of list [Nhap], optional
        list of afibs tract length to be updated, if None an empty one will be
        created The default is None.

    Returns
    -------
    afibs : TYPE
        the list is updated with the current segment's statistics

    """
    Nhap, Nsnp = data.shape
    ibs = np.zeros(Nsnp)
    curr_borders = dict()
    if afibs is None:
        afibs = [[] for der in range(Nhap)]

    for snp in range(Nsnp):
        Nderived = counts[snp].astype(int)
        if not polarized[snp]: continue
        if Nderived <= 1 or Nderived == Nhap:
            continue
        # Calculates a "code" corresponding to the "configuration" of each snp:
        # For each snp, the vector of 0s and 1s (ancestral and derived alleles
        # stored for each haplo) is converted to a number in base 10
        # conf[snp] = sum_i (haplo_i * 2^i), (haplo_i=0 if ancestral, 1 if derived)
        # This avoids recalculating the borders if there is a snp close by with
        # the same configuration and for which we already calculated the borders.

        conf = sum([2**x for x in np.where(data[:, snp])[0].astype(np.float)])
        if conf in curr_borders:
            # if the last calculated right border for the same conf is further right,
            # then the snp is in the same AF-IBS segment than the previous snp
            # having this conf so the borders of the segment do not change.

            if curr_borders[conf][1] > snp:
                l_snp, r_snp = curr_borders[conf]
                if l_snp >= 0 and r_snp != Nsnp:
                    afibs[Nderived].append(posOnChrom[r_snp]-posOnChrom[l_snp])
                    continue
            else:
                # Left border cannot be before the previous right border for the same config
                minPotentialLeft = curr_borders[conf][1]
                mask = data[:, snp]
                vec = data[mask, minPotentialLeft:snp].sum(axis=0) % Nderived
                l_snp = np.where(vec != 0)[0][-1] + minPotentialLeft
                r_snp, foundr = snp, False
                while not foundr and r_snp < Nsnp - 1:
                    r_snp += 1
                    foundr = data[mask, r_snp].sum() % Nderived
                if not foundr:
                    # No right border in analysed segment
                    curr_borders[conf] = [l_snp, Nsnp]
                else:
                    curr_borders[conf] = [l_snp, r_snp]
                    afibs[Nderived].append(posOnChrom[r_snp]-posOnChrom[l_snp])
                continue

        # NOT in CURR_BORDERS
        mask = data[:, snp]
        if Nderived == 2:
            # It is faster when Nderived is small to compute vec
            vec = data[mask, :].sum(axis=0) % Nderived
            try:
                l_snp = np.where(vec[:snp] != 0)[0][-1]
            except IndexError:
                l_snp = -1
            try:
                r_snp = np.where(vec[snp+1:] != 0)[0][0] + snp + 1
            except IndexError:
                r_snp = Nsnp
            curr_borders[conf] = [l_snp, r_snp]
            if l_snp >= 0 and r_snp != Nsnp:
                afibs[Nderived].append(posOnChrom[r_snp]-posOnChrom[l_snp])
                continue

        l_snp, foundl = snp, False
        while not foundl and l_snp > 0:
            l_snp -= 1
            foundl = data[mask, l_snp].sum() % Nderived
        l_snp -= int(not foundl)

        r_snp, foundr = snp, False
        while not foundr and r_snp < Nsnp-1:
            r_snp += 1
            foundr = data[mask, r_snp].sum() % Nderived
        r_snp += int(not foundr)

        curr_borders[conf] = [l_snp, r_snp]
        if l_snp >= 0 and r_snp != Nsnp:
            afibs[Nderived].append(posOnChrom[r_snp]-posOnChrom[l_snp])

    return afibs


def distrib_afibs_unpol(polarized, haps, pos, counts, durbin_bwt=False):
    """Moments for length distributions of AF-IBS as defined by Theunert 2012.

    Parameters
    ----------
    haps : np.array[Nhap,Nsnp_seg]
        haplotype data for each segment
    pos : np.array[Nsnp_seg]
        positions of SNP for rach segment
    counts : np.array[Nsnp_seg]
        number of derived alleles at each position for each segment
    durbin_bwt : bool, optional
        whether to use algorithm based on durbin ibs algo using Burrows-Wheeler
        Transform. The default is False.

    Returns
    -------
    mean_sd_afibs : np.array(mean_2,sd_2, mean_3,sd_3, ...)
        (mean,sd) afibs lengths for each category of derived alleles number2..n

    """
    Nhap = haps.shape[0]
    afibs = [[] for der in range(Nhap)]
    afibs = afibs_fast_unpol(polarized, haps == 1, pos, counts, afibs)
    # haps==1 because afibs_fast takes a boolean array as argument, not int.
    mean_sd_afibs = np.zeros(shape=(len(afibs)-2)*2)
    # we don't compute afibs values for singletons (does not make sense)
    # nor for fixed derived (because we don't simulated fixed derived,
    # and the ones appearing because of errors added afterwards are pruned)
    i = 0
    for der in range(2, len(afibs)):
        if len(afibs[der]) > 0:
            mean_sd_afibs[i] = np.mean(afibs[der])
            mean_sd_afibs[i+1] = np.std(afibs[der])
        i += 2

    return mean_sd_afibs
