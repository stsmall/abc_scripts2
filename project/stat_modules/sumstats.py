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
from itertools import combinations
from project.stat_modules import afibs
from project.stat_modules import afs
from project.stat_modules import ibs
from project.stat_modules import ld as ldfx
from project.stat_modules import popstats
from project.stat_modules import pwpopstats


class PopSumStats:
    """Calculates stats from pos/gt and msprime object."""

    def __init__(self, pos=None, haparr=None, counts=None, stats=None):
        self.pos = pos
        self.haparr = haparr
        self.counts = counts
        self.stats = stats

    def split_pop(self, pop_ix):
        hap_p = self.haparr[pop_ix, :]
        seg = hap_p.sum(axis=0).astype(int)
        seg_mask = seg > 0
        pos_p = self.pos[seg_mask]
        hap_p = hap_p[:, seg_mask]
        counts_p = hap_p.sum(axis=0).astype(int)

        return hap_p, pos_p, counts_p

    def pi(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        win_size = self.stats["win_size1"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for pop in self.stats["pop_config"]:
            gtpop = gt.take(pop, axis=1)
            pi_, pi_std = popstats.pi_window(pos, gtpop, win_size, length_bp)
            stats_ls.extend([pi_, pi_std])

        return stats_ls

    def tajd(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        win_size = self.stats["win_size1"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for pop in self.stats["pop_config"]:
            gtpop = gt.take(pop, axis=1)
            tajd_, tajd_std = popstats.tajimaD(pos, gtpop, win_size, length_bp)
            stats_ls.extend([tajd_, tajd_std])

        return stats_ls

    def classic(self):
        stats_ls = []
        for p in self.stats["pop_config"]:
            hap_p, pos_p, counts_p = self.split_pop(p)
            het_, pi_, tajd_ = popstats.classical_stats(len(p), counts_p)
            stats_ls.extend([het_, pi_, tajd_])

        return stats_ls

    def hap_het(self):
        win_size = self.stats["win_size1"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for p in self.stats["pop_config"]:
            hap_p, pos_p, counts_p = self.split_pop(p)
            res_hh = popstats.haplo_win(pos_p, hap_p, win_size, length_bp)
            stats_ls.extend(res_hh)  # het_win, het_win_std
        return stats_ls

    def spatial_sfs(self):
        stats_ls = []
        fold = self.stats["spat_fold"]
        for p in self.stats["pop_config"]:
            hap_p, pos_p, counts_p = self.split_pop(p)
            n_haplo = len(p)
            spat_ = afs.spatial_histo_fast(pos_p, counts_p, n_haplo-1)
            if fold:
                spat_flip = np.flip(spat_, axis=0)
                spat_fold = (spat_flip + spat_)
                haps = int(len(p)/2)
                spat_fold = spat_fold[0:haps]
                spat_ = spat_fold
            stats_ls.extend(spat_)

        return stats_ls

    def afibs(self, durbin=True):
        stats_ls = []
        fold = self.stats["afibs_fold"]
        for p in self.stats["pop_config"]:
            hap_p, pos_p, counts_p = self.split_pop(p)
            res_afibs = afibs.distrib_afibs(hap_p, pos_p, counts_p, durbin)
            afibs_m = res_afibs[::2]  # mean afibs
            afibs_std = res_afibs[1::2]  # afibs_std
            if fold:
                for a in [afibs_m, afibs_std]:
                    afibs_flip = np.flip(a, axis=0)
                    afibs_fold = (afibs_flip + a)
                    haps = int(len(p)/2)
                    stats_ls.extend(afibs_fold[0:haps])
            else:
                stats_ls.extend(afibs_m)  # mean afibs
                stats_ls.extend(afibs_std)  # afibs_std
        return stats_ls

    def ibs(self, moments=False):
        dmax = self.stats["length_bp"]
        size = self.stats["ibs_params"][1]
        prob = self.stats["ibs_params"][0]
        stats_ls = []
        for m in size:
            if moments:
                for p in self.stats["pop_config"]:
                    hap_p, pos_p, counts_p = self.split_pop(p)
                    res_ibs = ibs.ibs_quantiles_from_data(m, pos_p, hap_p, prob, dmax, moments=True)
                    stats_ls.extend(res_ibs)  # mean, variance, skewness, kurtosis
            else:
                for p in self.stats["pop_config"]:
                    hap_p, pos_p, counts_p = self.split_pop(p)
                    res_ibs = ibs.ibs_quantiles_from_data(m, pos_p, hap_p, prob, dmax)
                    stats_ls.extend(res_ibs)
        # dict(zip(size_list, [ss.ibs_quantiles_from_data(size, self.pos, 1, self.haparr, prob, dmax, quantiles=False, moments=True) for size in size_list]))
        # pd_ibs_mom = pd.DataFrame(res_ibs_mom, index=['mean', 'variance', 'skewness', 'kurtosis'])
        return stats_ls

    def ld(self):
        stats_ls = []
        intervals = self.stats["ld_params"]
        for p in self.stats["pop_config"]:
            hap_p, pos_p, counts_p = self.split_pop(p)
            #pw_ld = ldfx.distrib_r2(pos_p, hap_p, intervals)  # Boitard 2015 r2
            pw_ld = ldfx.ld_pop_mp(pos_p, hap_p, intervals)  # momentsLD 100 snps
            #pw_ld3 = ldfx.ld_pop_mb(pos_p, hap_p, intervals)  # momentsLD + Boitard 2015 selection
            #pw_ld2 = ldfx.ld_pop_complete(pos_p, hap_p, intervals)  # momentsLD pw-all very slow
            stats_ls.extend(pw_ld)

        return stats_ls

    def sfs(self, fold=False):
        fold = self.stats["sfs_fold"]
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        stats_ls = []
        for pop in self.stats["pop_config"]:
            gtpop = gt.take(pop, axis=1)
            sfs = afs.asfs_stats(gtpop, pos, fold)
            stats_ls.extend(sfs)

        return stats_ls

    def jsfs(self, fold=False):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            gtpops = gt.take(p1+p2, axis=1)
            props = afs.jsfs_stats(len(p1), gtpops, pos, fold)
            stats_ls.extend(props)

        return stats_ls

    def delta_tajD(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        win_size = self.stats["win_size1"]
        length_bp = self.stats["length_bp"]
        quants = self.stats["pw_quants"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            gtpops = gt.take(p1+p2, axis=1)
            flt = pwpopstats.d_tajD(len(p1), pos, gtpops, win_size, length_bp, quants)
            stats_ls.extend(flt)
        return stats_ls

    def ld_2pop(self, D=0):
        # ld_pop2_avg(p1, gt, pos, win_size, length_bp, ld, maf=0.05)
        # D2 = 0; Dz = 1; pi2 = 2
        quants = self.stats["pw_quants"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            hap_p, pos_p, counts_p = self.split_pop(p1+p2)
            pw_ld = ldfx.ld_pop2(len(p1), pos_p, hap_p, quants)
            stats_ls.extend(pw_ld)

        return stats_ls

    def ld_2pop_win(self, D=0):
        # D2 = 0; Dz = 1; pi2 = 2
        win_size = self.stats["win_size2"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            hap_p, pos_p, counts_p = self.split_pop(p1+p2)
            pw_ld = ldfx.ld_pop2_win(len(p1), pos_p, hap_p, win_size, length_bp)
            stats_ls.extend(pw_ld)

        return stats_ls

    def FST(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        quants = self.stats["pw_quants"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            gtpops = gt.take(p1+p2, axis=1)
            flt = pwpopstats.fst(len(p1), pos, gtpops, quants)
            try:
                stats_ls.extend(flt)
            except TypeError:
                flt = [np.nan]*len(quants)
                stats_ls.extend(flt)
        return stats_ls

    def dXY(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        quants = self.stats["pw_quants"]
        win_size = self.stats["win_size2"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            gtpops = gt.take(p1+p2, axis=1)
            flt = pwpopstats.dxy(len(p1), pos, gtpops, win_size, length_bp)
            if quants[0] < 0:
                dxy_ = [np.nanmean(flt)]
            else:
                dxy_ = np.nanquantile(flt, quants)
            stats_ls.extend(dxy_)
        return stats_ls

    def dmin(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        quants = self.stats["pw_quants"]
        win_size = self.stats["win_size2"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            gtpops = gt.take(p1+p2, axis=1)
            flt = pwpopstats.dmin(len(p1), pos, gtpops, win_size, length_bp, quants)
            if quants[0] < 0:
                dminq = [np.nanmean(flt)]
            else:
                dminq = np.nanquantile(flt, quants)
            stats_ls.extend(dminq)
        return stats_ls

    def gmin(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        quants = self.stats["pw_quants"]
        win_size = self.stats["win_size2"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            gtpops = gt.take(p1+p2, axis=1)
            flt = pwpopstats.gmin(len(p1), pos, gtpops, win_size, length_bp, quants)
            stats_ls.extend(flt)
        return stats_ls

    def dd12(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        quants = self.stats["pw_quants"]
        win_size = self.stats["win_size2"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            gtpops = gt.take(p1+p2, axis=1)
            flt = pwpopstats.dd1_2(len(p1), pos, gtpops, win_size, length_bp, quants)
            stats_ls.extend(flt) # 2 values returned as list [dd1, dd2]
        return stats_ls

    def ddRank12(self):
        gt = allel.HaplotypeArray(self.haparr.T)
        pos = allel.SortedIndex(self.pos)
        quants = self.stats["pw_quants"]
        win_size = self.stats["win_size2"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            gtpops = gt.take(p1+p2, axis=1)
            flt = pwpopstats.ddRank1_2(len(p1), pos, gtpops, win_size, length_bp, quants)
            stats_ls.extend(flt)  # 2 values returned as list [dd1, dd2]
        return stats_ls

    def Zx(self):
        win_size = self.stats["win_size2"]
        quants = self.stats["pw_quants"]
        length_bp = self.stats["length_bp"]
        stats_ls = []
        stats_ls = []
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            hap_p, pos_p, counts_p = self.split_pop(p1+p2)
            flt = pwpopstats.zx(len(p1), pos_p, hap_p, win_size, length_bp, quants)
            stats_ls.extend(flt)
        return stats_ls

    def IBS_maxXY(self):
        stats_ls = []
        length_bp = self.stats["length_bp"]
        for p1, p2 in combinations(self.stats["pop_config"], 2):
            hap_p, pos_p, counts_p = self.split_pop(p1+p2)
            flt = pwpopstats.ibs_maxxy(len(p1), pos_p, hap_p, length_bp)
            stats_ls.append(flt)
        return stats_ls
