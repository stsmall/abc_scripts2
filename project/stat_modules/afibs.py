#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:31:20 2021
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

"""
import numpy as np


def afibs_durbin_left_length(haplos, posOnChrom, counts, rightbounds, afibs):
    """Compute afibs-leftbound and length of afibs segment using positions.

    Based on Durbin (PBWT), Bioinformatics 2014, Algorithm 2
    Author: Flora Jay

    Parameters
    ----------
    haplos : np.array[Nhap,Nsnp]
        haplotype data for one region for Nhap individuals x Nsnp SNPs
    posOnChrom : int array
        positions (bp) of each polymorphism relative to its chromosome
    counts : np.array[Nsnp_seg]
         number of derived alleles at each position
    rightbounds : list of int [Nsnp]
        SNP index of afibs-rightbound at each position returned by afibs_durbin_right
    afibs : list of list [Nhap]
        list of afibs tract length to be updated

    Returns
    -------
    afibs : list of list [Nhap]
        list containing a list for each allele count 0...Nhap

    """
    Nhap, Nsnp = haplos.shape
    # To keep the details of all segment lengths:
    if afibs is None:
        afibs = [[] for der in range(Nhap)]

    # LEFT BOUND

    # Most variable names follow Durbin algo 2
    # But haplos replaces yi
    # d: divergence array
    # a: positional prefix array

    acurr = range(Nhap)
    dcurr = [0]*Nhap
    leftbounds = [0]*Nsnp
    for k in range(Nsnp):

        p, q = k+1, k+1
        a, b, d, e = [[] for i in range(4)]
        commonbound = 0
        # will contain the MAXIMAL index such that haplotypes carrying '1' at
        # position k are all identical from commonbound (included) to k
        firstDerived = True
        for i in range(Nhap):
            if dcurr[i] > p:
                p = dcurr[i]
            if dcurr[i] > q:
                q = dcurr[i]
            if haplos[acurr[i], k] == 0:
                a.append(acurr[i])
                d.append(p)
                p = 0
            else:
                b.append(acurr[i])
                e.append(q)
                if firstDerived:
                    # When we encounter the first haplotype carrying a derived allele at k
                    # There is no bound to look for yet
                    # because the haplo differs from previous haplo at position k (previous haplo carries a '0')
                    firstDerived = False
                elif q > commonbound:
                    commonbound = q
                    # contains the current MAXIMAL index such that haplotypes already
                    # parsed and carrying '1' at position k are all identical from commonbound to k
                q = 0
        acurr = a+b
        dcurr = d+e
        leftbounds[k] = commonbound-1  # try: np.max(e[1:])-1; except: pass
        # If left!=1 and right!=Nsnp then both bounds are inside the genomic region
        # and we can save the length of the segment
        # Otherwise it means they were not updated because the segment overlaps the region boundaries
        # so the exact length is not known and not saved
        if (leftbounds[k] != -1 and rightbounds[k] != Nsnp):
            seglen = posOnChrom[rightbounds[k]]-posOnChrom[leftbounds[k]]
            if counts[k] < 1 or counts[k] == Nhap:
                continue
            else:
                afibs[counts[k]].append(seglen)
            # otherstat[counts[k]] += seglen**2  # if you want to directly compute
            # other stats (eg for the moments, do it here and remember do the init and to add them to the return list)
    return afibs


def afibs_durbin_right(haplos):
    """Search for afibs-rightbound at each SNP.

    Based on Durbin (PBWT), Bioinformatics 2014
    Author: Flora Jay

    Parameters
    ----------
    haplos : np.array[Nhap,Nsnp]
        haplotype data for one region for Nhap individuals x Nsnp SNPs

    Returns
    -------
    rightbounds : list of int [Nsnp]
        SNP index of afibs-rightbound at each position

    """
    Nhap, Nsnp = haplos.shape

    # Looking for RIGHT BOUND by applying Durbin algo starting the right
    # ie Loop on snp index starts at Nsnp and finishes at 0

    # Most variable names follow Durbin algo 2
    # But haplos replaces yi
    # d: divergence array
    # a: positional prefix array

    acurr = range(Nhap)
    dcurr = [Nsnp-1]*Nhap
    rightbounds = [0]*Nsnp
    for k in reversed(range(Nsnp)):  # differs from leftbound search algo (I'll put a DIFF label for these lines)
        p, q = k-1, k-1  # DIFF FROM LEFT k+1,k+1
        a, b, d, e = [[] for i in range(4)]
        commonbound = Nsnp  # DIFF
        firstDerived = True
        for i in range(Nhap):
            if dcurr[i] < p:  # DIFF
                p = dcurr[i]
            if dcurr[i] < q:  # DIFF
                q = dcurr[i]
            if haplos[acurr[i], k] == 0:
                a.append(acurr[i])
                d.append(p)
                p = Nsnp-1  # DIFF
            else:
                b.append(acurr[i])
                e.append(q)
                if firstDerived:
                    # When we encounter the first haplotype carrying a derived allele at k
                    # There is no bound to look for yet
                    # because the haplo differs from all previous haplo at position k (previous haplo carries a '0')
                    firstDerived = False
                elif q < commonbound:
                    commonbound = q
                    # commonbound contains the current MINIMAL index such that haplotypes already parsed and carrying '1' at position k are all identical from k to commonbound (included)
                q = Nsnp-1  # DIFF
        acurr = a+b
        dcurr = d+e
        rightbounds[k] = commonbound+1   # DIFF

    return rightbounds


def afibs_durbin_compact(haplos, posOnChrom, counts, afibs=None):
    """Compute afibs length (Theunert 2012) for each allele count >=2.

    Algo was adapted from Durbin (PBWT), Bioinformatics 2014.
    Author: Flora Jay

    Parameters
    ----------
    haplos : np.array[Nhap,Nsnp]
        haplotype data for one region for Nhap individuals x Nsnp SNPs.
    posOnChrom : int array
        positions (bp) of each polymorphism relative to its chromosome.
    counts : np.array[Nsnp_seg]
        number of derived alleles at each position.
    afibs : list of list [Nhap], optional
        list of afibs tract length to be updated, if None an empty one will be
        created. The default is None.

    Returns
    -------
    afibs : TYPE
        DESCRIPTION.

    """
    rightbounds = afibs_durbin_right(haplos)
    afibs = afibs_durbin_left_length(haplos, posOnChrom, counts, rightbounds, afibs)
    # more stats?
    return afibs


def afibs_fast(data, posOnChrom, counts, afibs=None):
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


def distrib_afibs(haps, pos, counts, durbin=False):
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
    if durbin:
        afibs = afibs_durbin_compact(haps, pos, counts, afibs)
    else:
        afibs = afibs_fast(haps == 1, pos, counts, afibs)
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
