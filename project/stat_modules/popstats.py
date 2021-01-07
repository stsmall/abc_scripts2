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
import allel
from .sequtils import get_seg
import bisect
from collections import Counter


def classical_stats(nhaplo, counts):
    """Compute heterozygosity, diversity (pairwise differences) and Tajima's D.

    All calculation are per site

    Parameters
    ----------
    nhaplo : int
        total number of haplotypes
    counts : np.array[Nsnp_seg]
        number of derived alleles at each position for each segment

    Returns
    -------
    H, PI, D: float, float, float
        mean of het, pi, and tajD

    """
    Nhap = np.float(nhaplo)
    a1 = np.sum([1.0/i for i in range(1, nhaplo)])
    a2 = np.sum([1.0/i**2 for i in range(1, nhaplo)])
    c1 = (Nhap+1)/(3*(Nhap-1)) - 1/a1   # b1-1/a1
    c2 = 2*(Nhap**2+Nhap+3)/(9*Nhap*(Nhap-1)) - (Nhap+2)/(a1*Nhap) + a2/a1**2
    H, PI, D = [], [], []
    Nsnp = np.float(counts.shape[0])
    # Expected heterozygosity (at each site) for snp data Arlequin 8.1.1.2 p.115
    H = 2.0/(Nhap-1) * (counts-counts**2 / Nhap)
    # Mean number of pariwise difference (at each site) for snp data Arlequin 8.1.2.1 p.116
    PI = 2.0 / (Nhap * (Nhap-1)) * (counts * (Nhap-counts))
    theta_pi = sum(PI)
    # Other estimate of theta :
    theta_s = Nsnp/a1
    # var_theta_s = (a1**2 * Nsnp + a2 * Nsnp**2) / (a1**2 * (a1**2 + a2) )
    # var_PI= (3*Nhap*(Nhap+1)*PI + 2*(Nhap**2+Nhap+3)*PI**2) / (11*(Nhap**2-7*Nhap+6))
    # var_theta_pi= (3*Nhap*(Nhap+1)*theta_pi + 2*(Nhap**2+Nhap+3)*theta_pi**2) / (11*(Nhap**2-7*Nhap+6))
    # Tajima D, formula from Tajim's paper (1989)
    D = (theta_pi - theta_s) / np.sqrt(c1/a1 * Nsnp + (c2/(a1**2+a2))*Nsnp*(Nsnp-1))

    return H, PI, D


def haplo_het(pos, gt, win_size, length_bp):
    """Estimate haplotype diversity in moving windows.

    Parameters
    ----------
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    size : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    haphet_mean : TYPE
        DESCRIPTION.
    haphet_std : TYPE
        DESCRIPTION.

    """
    hpseg, pos_s = get_seg(gt, pos)
    hh_wins = []
    coords = (range(0, length_bp + win_size, win_size))
    windows = list(zip(coords[0::1], coords[1::1]))
    for s, e in windows:
        hp1_win = hpseg.compress((pos_s > s) & (pos_s <= e), axis=0)
        hh_ = allel.haplotype_diversity(hp1_win)
        hh_wins.append(hh_)
    haphet_mean = np.nanmean(hh_wins)
    haphet_std = np.nanstd(hh_wins)

    return haphet_mean, haphet_std


def tajimaD(pos, gt, win_size, length_bp):
    """Calculate Tajima's D in steps of seg sites.

    Parameters
    ----------
    ac : array
        allele counts array
    size : int, optional
        window size in number of variants. The default is 4.

    Returns
    -------
    tajd_mean : float
        DESCRIPTION.
    tajd_std : float
        DESCRIPTION.

    """
    gtseg, pos_s = get_seg(gt, pos)
    ac = gtseg.count_alleles()
    tajd_, *_ = allel.windowed_tajima_d(pos_s, ac, size=win_size, start=1, stop=length_bp)
    tajd_mean = np.nanmean(tajd_)
    tajd_std = np.nanstd(tajd_)

    return tajd_mean, tajd_std


def pi_window(pos, gt, win_size, length_bp):
    """Calculate pi in windows.

    Parameters
    ----------
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    win_size : TYPE
        DESCRIPTION.
    length_bp : TYPE
        DESCRIPTION.

    Returns
    -------
    pi_mean : TYPE
        DESCRIPTION.
    pi_std : TYPE
        DESCRIPTION.

    """
    gtseg, pos_s = get_seg(gt, pos)
    ac = gtseg.count_alleles()
    pi, *_ = allel.windowed_diversity(pos_s, ac, size=win_size, start=1, stop=length_bp)
    pi_mean = np.nanmean(pi)
    pi_std = np.nanstd(pi)

    return pi_mean, pi_std


def exp_het(pos, gt):
    """Calculate the expected rate of heterozygosity for each variant under HWE.

    Parameters
    ----------
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.

    Returns
    -------
    het_mean : TYPE
        DESCRIPTION.
    het_std : TYPE
        DESCRIPTION.

    """
    gtseg, pos_s = get_seg(gt, pos)
    af = gtseg.count_alleles().to_frequencies()
    het = allel.heterozygosity_expected(af, ploidy=2)
    het_mean = np.nanmean(het)
    het_std = np.nanstd(het)

    return het_mean, het_std


def het_one_win(dataslice, Nhap):
    """Compute haplotypic heterozygosity of a given window.

    Parameters
    ----------
    dataslice : np.array
        subset of the data corresponding to a given window of the sequence
    Nhap : int
        total number of haplotypes

    Returns
    -------
    het : float
        haplotypic heterozygosity of dataslice
    """
    haplos = [''.join([repr(num) for num in dataslice[i, :]]) for i in range(Nhap)]
    tab = Counter(haplos)
    het = 1.0-sum([x**2 for x in tab.values()])/float(Nhap)**2

    return het


def haplo_win(pos, hap, win_size, length_bp):
    """Compute haplotypic heterozygosity in windows sliding along the genome.

    Parameters
    ----------
    pos : np.array
        positions of SNP for rach segment
    hap : np.array
        haplotype data for each segment
    win_size : int
        lengh of the sliding windows considered as haplotypes (bp)
    length_bp : int
        length of each simulated segment (bp)

    Returns
    -------
    hap_het : float, float
        mean and standard deviation of haplotypic heterozygosity

    """
    Nhap = hap.shape[0]
    L = length_bp
    win_size = int(win_size)
    chunks = [bisect.bisect(pos, x) for x in range(0, L, win_size)]
    hets = [het_one_win(hap[:, chunks[i]:chunks[i+1]], Nhap) for i in range(len(chunks)-1)]
    Nhap = np.float(Nhap)
    haphet_mean = Nhap/(Nhap-1.0) * np.mean(hets)
    haphet_std = Nhap/(Nhap-1.0) * np.std(hets)

    return haphet_mean, haphet_std
