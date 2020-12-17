#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:05:14 2018
Class for calculating summary statistics from simulations input as ms or directly
as msprime
@author: stsmall
"""
from subprocess import run, PIPE
import numpy as np
import os
import allel
from itertools import combinations
from collections import defaultdict
import bisect

try:
    import dadi
    dadi_install = True
except ModuleNotFoundError:
    print("dadi not installed, scripts will not be available")
    dadi_install = False


class SumStats:
    """Calculates stats from pos/gt and msprime object."""

    def __init__(self, haparr=None, pos=None, pops=None):
        self.haparr = haparr
        self.pos = pos
        self.pops = pops

    def asfsStats(self, fold=False, rand=True, randn=10000):
        """Calculate the aggregate SFS, singletons and doubletons.

        Parameters
        ----------
        fold : TYPE, optional
            DESCRIPTION. The default is False.
        rand : TYPE, optional
            DESCRIPTION. The default is True.
        randn : TYPE, optional
            DESCRIPTION. The default is 100000.

        Returns
        -------
        asfs_dict : TYPE
            DESCRIPTION.

        """
        asfs_dict = {}
        gt_list = []
        for hap in self.haparr:
            if type(hap) is list:
                gt_list.append(allel.HaplotypeArray(np.vstack(hap), dtype='i1'))
            else:
                gt_list.append(allel.HaplotypeArray(hap))
        for rep, gt in enumerate(gt_list):
            aSFS1 = []
            aSFS2 = []
            aSFS = []
            for pop in self.pops:
                gtpop = gt.take(pop, axis=1)
                acpop = gtpop.count_alleles()
                seg = acpop.is_segregating()
                try:
                    gtseg = gtpop.compress(seg)
                    # random snps
                    if rand:
                        n = randn  # number of SNPs to choose randomly
                        try:
                            vidx = np.random.choice(gtseg.shape[0], n, replace=False)
                            vidx.sort()
                            gtp = gtseg.take(vidx, axis=0)
                        except ValueError:
                            gtp = gtseg
                    else:
                        gtp = gtseg
                    # sfs
                    if fold:
                        sfsp = (allel.sfs_folded(gtp.count_alleles()))
                    else:
                        sfsp = (allel.sfs(gtp.count_alleles()[:, 1]))
                    tots = np.sum(sfsp)
                except ValueError:
                    # no segregating snps
                    tots = 1
                    sfsp = [0]*len(pop)
                try:
                    aSFS1.append(sfsp[1]/tots)
                except IndexError:
                    aSFS1.append(0)
                try:
                    aSFS2.append(sfsp[2]/tots)
                except IndexError:
                    aSFS2.append(0)
                aSFS.append(sfsp)
            agg = aSFS1 + aSFS2
            asfs_dict[rep] = " ".join(map(str, agg))
        return asfs_dict

    def summarizejsfs(self, fs):
        """Create summary jsfs.

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

    def jsfsStats(self, pairs, fold=False, rand=True, randn=10000):
        """Calculate joint site frequency spectrum (jsfs) with scikit-allel.

        Parameters
        ----------
        pairs : TYPE
            DESCRIPTION.
        fold : TYPE, optional
            DESCRIPTION. The default is False.
        seg : TYPE, optional
            DESCRIPTION. The default is True.
        rand : TYPE, optional
            DESCRIPTION. The default is True.
        randn : TYPE, optional
            DESCRIPTION. The default is 100000.

        Returns
        -------
        jsfs_dict : TYPE
            DESCRIPTION.

        """
        jsfs_dict = {}
        gt_list = []
        for hap in self.haparr:
            if type(hap) is list:
                gt_list.append(allel.HaplotypeArray(np.vstack(hap), dtype='i1'))
            else:
                gt_list.append(allel.HaplotypeArray(hap))
        for rep, gt in enumerate(gt_list):
            jsfs_list = []
            for pair in pairs:
                i, j = pair.split("-")
                p1 = self.pops[int(i)]
                p2 = self.pops[int(j)]
                gtpops = gt.take(p1+p2, axis=1)
                acpops = gtpops.count_alleles()
                segpops = acpops.is_segregating()
                gtseg = gtpops.compress(segpops)
                # random snps
                if rand:
                    n = randn  # number of SNPs to choose randomly
                    try:
                        vidx = np.random.choice(gtseg.shape[0], n, replace=False)
                        vidx.sort()
                        gtr = gtpops.take(vidx, axis=0)
                    except ValueError:
                        gtr = gtseg
                else:
                    gtr = gtseg
                gtpop1 = gtr.take(range(len(p1)), axis=1)
                gtpop2 = gtr.take(range(len(p1), gtr.shape[1]), axis=1)
                ac1 = gtpop1.count_alleles()
                ac2 = gtpop2.count_alleles()
                # jsfs
                if fold:
                    # pad for allel as well
                    popsizeA, popsizeB = len(p1)/2, len(p2)/2
                    jsfs = allel.joint_sfs_folded(ac1, ac2)
                    fs = np.resize(jsfs, (popsizeA+1, popsizeB+1))
                else:
                    # pad for allel as well
                    popsizeA, popsizeB = len(p1), len(p2)
                    jsfs = allel.joint_sfs(ac1[:, 1], ac2[:, 1])
                    fs = np.resize(jsfs, (popsizeA+1, popsizeB+1))
                props = self.summarizejsfs(fs)
                jsfs_list.append(props)
            jsfs_dict[rep] = " ".join(map(str, np.concatenate(jsfs_list).ravel()))
        return jsfs_dict

    def calc_afibs(self, gt, pos, basepairs, fold):
        """Calculate afibs.

        Parameters
        ----------
        gt : TYPE
            DESCRIPTION.
        pos : TYPE
            DESCRIPTION.
        basepairs : TYPE
            DESCRIPTION.
        fold : TYPE
            DESCRIPTION.

        Returns
        -------
        afibs_list : TYPE
            DESCRIPTION.

        """
        afibs = {}
        for pix, pop in enumerate(self.pops):
            ibsarray = np.zeros((len(pop), len(pop)-1))
            gtpop = gt.take(pop, axis=1)
            acpop = gtpop.count_alleles()
            seg = acpop.is_segregating()
            gtseg = gtpop.compress(seg)
            # ac = gtseg.count_alleles()
            poslist = pos[seg]
            freqlist = np.sum(gtseg, axis=1)
            for ind in range(len(pop)):
                indpos = poslist[gtseg[:, ind] > 0]
                indpos = np.insert(indpos, 0, 0)
                indpos = np.insert(indpos, len(indpos), basepairs)
                for freq in range(1, len(pop)):
                    ibs = 0
                    mut_ix = np.where(freqlist == freq)[0]
                    if len(mut_ix) > 0:
                        for m in mut_ix:
                            start = bisect.bisect_left(indpos, poslist[m])
                            end = bisect.bisect_right(indpos, poslist[m])
                            ibs += indpos[end] - indpos[start - 1]
                        ibsarray[ind, freq-1] = (ibs / len(mut_ix))
            ibsarray[ibsarray == 0] = np.nan
            afibs[pix] = np.nanmean(ibsarray, axis=0)
        # fold and export to list
        afibs_list = []
        if fold:
            for pix, pop in enumerate(self.pops):
                ibs_flip = np.flip(afibs[pix], axis=0)
                ibs_fold = (ibs_flip + afibs[pix]) / 2
                haps = int(len(pop)/2)
                ibs = ibs_fold[0:haps]
                afibs_list.append(ibs)
        else:
            for i in range(len(self.pops)):
                afibs_list.append(afibs[i])
        return afibs_list

    def afibs(self, basepairs, fold=False):
        """Allele Frequency spectrum of ibs lengths.

        For each individual within a population calculate up/down distance to
        nearest SNP for each frequency class of alleles. Average among individuals
        within a population and return a vector of length pop-2 (no fixed classes)
        of sizes for each freq class.

        Parameters
        ----------
        basepairs : TYPE
            DESCRIPTION.
        fold : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        afibs_dict : TYPE
            DESCRIPTION.

        """
        afibs_dict = {}
        for rep, hap in enumerate(self.haparr):
            if type(hap) is list:
                afibslist = []
                for sub_rep in list(zip(self.pos[rep], hap)):
                    pos, gtarr = sub_rep
                    gt = allel.HaplotypeArray(gtarr)
                    afibslist.append(self.calc_afibs(gt, pos, basepairs, fold))
                afibs_zip = list(zip(*afibslist))
                afibsmean = [np.mean(x, axis=0) for x in afibs_zip]
            else:
                gt = allel.HaplotypeArray(hap)
                pos = self.pos[rep][0]
                afibsmean = self.calc_afibs(gt, pos, basepairs, fold)
            afibs_dict[rep] = " ".join(map(str, [i for t in afibsmean for i in t]))
        return afibs_dict

    def filetStats(self, basepairs, filet_path, block):
        """Calculate stats using FILET.

        Parameters
        ----------
        block : TYPE
            DESCRIPTION.
        filet_path : TYPE
            DESCRIPTION.
        window : TYPE
            DESCRIPTION.

        Returns
        -------
        filet_dict : TYPE
            DESCRIPTION.

        """
        if basepairs > 100000:
            block = 100000
        if block == 0:
            block = basepairs

        keep_stats = np.array([True, True, True, True, False, True, False, False,
                               True, True, True, True, True, False, True, False,
                               False, True, True, True, True, True, True, True,
                               True, True, True, True, False, False, False])
        norm = np.array([block, block**2, block, block, 1, 1, block, block**2,
                         block, block, 1, 1, 1, 1, block, block, block, 1, 1, 1, 1, 1])
        filet_dict = defaultdict(list)
        for pop1, pop2 in combinations(self.pops, 2):
            n1 = len(pop1)
            n2 = len(pop2)
            fakems_haps = []
            rep_dict = defaultdict(list)
            loci_r = 0
            for rep, hap in enumerate(self.haparr):
                if type(hap) is list:
                    for sub_rep in list(zip(self.pos[rep], hap)):
                        posr, gtarr = sub_rep
                        gt = allel.HaplotypeArray(gtarr)
                        gtpops = gt.take(pop1+pop2, axis=1)
                        acpops = gtpops.count_alleles()
                        segpops = acpops.is_segregating()
                        gtseg = gtpops.compress(segpops)
                        posit = posr[segpops] / block
                        if basepairs > block:
                            start = 0
                            step = block
                            end = start + step
                            while end < basepairs:
                                rep_dict[rep].append(loci_r)
                                loci_r += 1
                                s_ix = bisect.bisect_left(posit, start)
                                e_ix = bisect.bisect_right(posit, end) - 1
                                posit_block = posit[s_ix:e_ix] / basepairs
                                gtseg_block = gtseg[s_ix:e_ix]
                                seg = gtseg_block.shape[0]
                                fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit_block))}\n")
                                for geno in gtseg_block.transpose():
                                    fakems_haps.append(f"{''.join(map(str, geno))}\n")
                                start += step
                                end += step
                        else:
                            rep_dict[rep].append(loci_r)
                            loci_r += 1
                            posit = posit / block
                            seg = np.count_nonzero(segpops)
                            fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit))}\n")
                            for geno in gtseg.transpose():
                                fakems_haps.append(f"{''.join(map(str, geno))}\n")
                else:
                    if basepairs > block:
                        start = 0
                        step = block
                        end = start + step
                        while end < basepairs:
                            rep_dict[rep].append(loci_r)
                            loci_r += 1
                            s_ix = bisect.bisect_left(posit, start)
                            e_ix = bisect.bisect_right(posit, end) - 1
                            posit_block = posit[s_ix:e_ix] / basepairs
                            gtseg_block = gtseg[s_ix:e_ix]
                            seg = gtseg_block.shape[0]
                            fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit_block))}\n")
                            for geno in gtseg_block.transpose():
                                fakems_haps.append(f"{''.join(map(str, geno))}\n")
                            start += step
                            end += step
                    else:
                        rep_dict[rep].append(loci_r)
                        loci_r += 1
                        gt = allel.HaplotypeArray(hap)
                        posr = self.pos[rep][0]
                        gtpops = gt.take(pop1+pop2, axis=1)
                        acpops = gtpops.count_alleles()
                        segpops = acpops.is_segregating()
                        gtseg = gtpops.compress(segpops)
                        posit = posr[segpops] / block
                        seg = np.count_nonzero(segpops)
                        fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit))}\n")
                        for geno in gtseg.transpose():
                            fakems_haps.append(f"{''.join(map(str, geno))}\n")
            fakems_head = f"ms {n1+n2} {loci_r} -t tbs -r tbs {block} -I 2 {n1} {n2}\n1234\n"
            fakems = "".join(fakems_haps)
            msinput = fakems_head + fakems
            filet_prog = os.path.join(filet_path, "twoPopnStats_forML")
            cmd = [filet_prog, str(n1), str(n2)]
            proc = run(cmd, stdout=PIPE, input=msinput, encoding='ascii', check=True)
            # collect stats
            lstats = proc.stdout.rstrip().split('\n')[1:]
            filet_list = [list(map(float, lf.split())) for lf in lstats]
            for rep, stat_ix in rep_dict.items():
                # must be a more elegant way to cluster these
                if len(stat_ix) > 1:
                    stat_vec = [filet_list[i] for i in stat_ix]
                    stat_arr = np.vstack(stat_vec)
                    stat_arr[np.isinf(stat_arr)] = 'nan'
                    filetmean = np.nanmean(stat_arr, axis=0)
                    filet_norm = filetmean[keep_stats] / norm
                    filet_dict[rep].append(" ".join(map(str, filet_norm)))
                else:
                    stat_vec = filet_list[stat_ix[0]]
                    stat_arr = np.array(stat_vec)[keep_stats]
                    stat_arr[np.isinf(stat_arr)] = 'nan'
                    filet_norm = stat_arr / norm
                    filet_dict[rep].append(" ".join(map(str, filet_norm)))
        # filet = " ".join(map(str, np.concatenate(filet_list).ravel()))
        return filet_dict

    def filetStatsMP(self, args):
        """Calculate stats using FILET w/ multiprocessors.

        Parameters
        ----------
        args : TYPE
            DESCRIPTION.

        Returns
        -------
        filet_ldict : TYPE
            DESCRIPTION.

        """
        # args in
        pop1, pop2, basepairs, filet_path, block = args

        if basepairs > 100000:
            block = 100000
        if block == 0:
            block = basepairs

        # stats and norm
        keep_stats = np.array([True, True, True, True, False, True, False, False,
                               True, True, True, True, True, False, True, False,
                               False, True, True, True, True, True, True, True,
                               True, True, True, True, False, False, False])
        norm = np.array([block, block**2, block, block, 1, 1, block, block**2,
                         block, block, 1, 1, 1, 1, block, block, block, 1, 1, 1, 1, 1])
        # start calc
        filet_ldict = []
        n1 = len(pop1)
        n2 = len(pop2)
        fakems_haps = []
        rep_dict = defaultdict(list)
        loci_r = 0
        for rep, hap in enumerate(self.haparr):
            if type(hap) is list:
                for sub_rep in list(zip(self.pos[rep], hap)):
                    posr, gtarr = sub_rep
                    gt = allel.HaplotypeArray(gtarr)
                    gtpops = gt.take(pop1+pop2, axis=1)
                    acpops = gtpops.count_alleles()
                    segpops = acpops.is_segregating()
                    gtseg = gtpops.compress(segpops)
                    posit = posr[segpops] / block
                    if basepairs > block:
                        start = 0
                        step = block
                        end = start + step
                        while end < basepairs:
                            rep_dict[rep].append(loci_r)
                            loci_r += 1
                            s_ix = bisect.bisect_left(posit, start)
                            e_ix = bisect.bisect_right(posit, end) - 1
                            posit_block = posit[s_ix:e_ix] / basepairs
                            gtseg_block = gtseg[s_ix:e_ix]
                            seg = gtseg_block.shape[0]
                            fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit_block))}\n")
                            for geno in gtseg_block.transpose():
                                fakems_haps.append(f"{''.join(map(str, geno))}\n")
                            start += step
                            end += step
                    else:
                        rep_dict[rep].append(loci_r)
                        loci_r += 1
                        posit = posit / block
                        seg = np.count_nonzero(segpops)
                        fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit))}\n")
                        for geno in gtseg.transpose():
                            fakems_haps.append(f"{''.join(map(str, geno))}\n")
            else:
                if basepairs > block:
                    start = 0
                    step = block
                    end = start + step
                    while end < basepairs:
                        rep_dict[rep].append(loci_r)
                        loci_r += 1
                        s_ix = bisect.bisect_left(posit, start)
                        e_ix = bisect.bisect_right(posit, end) - 1
                        posit_block = posit[s_ix:e_ix] / basepairs
                        gtseg_block = gtseg[s_ix:e_ix]
                        seg = gtseg_block.shape[0]
                        fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit_block))}\n")
                        for geno in gtseg_block.transpose():
                            fakems_haps.append(f"{''.join(map(str, geno))}\n")
                        start += step
                        end += step
                else:
                    rep_dict[rep].append(loci_r)
                    loci_r += 1
                    gt = allel.HaplotypeArray(hap)
                    posr = self.pos[rep][0]
                    gtpops = gt.take(pop1+pop2, axis=1)
                    acpops = gtpops.count_alleles()
                    segpops = acpops.is_segregating()
                    gtseg = gtpops.compress(segpops)
                    posit = posr[segpops] / block
                    seg = np.count_nonzero(segpops)
                    fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit))}\n")
                    for geno in gtseg.transpose():
                        fakems_haps.append(f"{''.join(map(str, geno))}\n")
        fakems_head = f"ms {n1+n2} {loci_r} -t tbs -r tbs {block} -I 2 {n1} {n2}\n1234\n"
        fakems = "".join(fakems_haps)
        msinput = fakems_head + fakems
        filet_prog = os.path.join(filet_path, "twoPopnStats_forML")
        cmd = [filet_prog, str(n1), str(n2)]
        proc = run(cmd, stdout=PIPE, input=msinput, encoding='ascii', check=True)
        # collect stats
        lstats = proc.stdout.rstrip().split('\n')[1:]
        filet_list = [list(map(float, lf.split())) for lf in lstats]
        for rep, stat_ix in rep_dict.items():
            # must be a more elegant way to cluster these
            if len(stat_ix) > 1:
                stat_vec = [filet_list[i] for i in stat_ix]
                stat_arr = np.vstack(stat_vec)
                stat_arr[np.isinf(stat_arr)] = 'nan'
                filetmean = np.nanmean(stat_arr, axis=0)
                filet_norm = filetmean[keep_stats] / norm
                filet_ldict.append(" ".join(map(str, filet_norm)))
            else:
                stat_vec = filet_list[stat_ix[0]]
                stat_arr = np.array(stat_vec)[keep_stats]
                stat_arr[np.isinf(stat_arr)] = 'nan'
                filet_norm = stat_arr / norm
                filet_ldict.append(" ".join(map(str, filet_norm)))

        return filet_ldict

    def sfs_dadi(self, npop, seg=True, fold=False, jsfs=True):
        """Calculate the site frequency spectrum with dadi.

        Parameters
        ----------
        npop : TYPE
            DESCRIPTION.
        seg : TYPE, optional
            DESCRIPTION. The default is True.
        fold : TYPE, optional
            DESCRIPTION. The default is False.
        jsfs : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        sfs_list :
            asdf
        jsfs_list :
            asdf

        """
        if not dadi_install:
            print("dadi not installed, exiting")
            return None
        gt = self.haparr
        sfs_list = []
        jsfs_list = []
        for i, p in enumerate(self.pops):
            gtpop = gt.take(p, axis=1)
            acpop = gtpop.count_alleles()
            seg = acpop.is_segregating()
            gtseg = gtpop.compress(seg)
            ac = gtseg.count_alleles()
            sfs = allel.sfs(ac[:, 1])
            sfs.resize(len(p), refcheck=False)
            fs = dadi.Spectrum(sfs, mask_corners=False, pop_ids=["pop{}".format(i)])
            if fold:
                fs = fs.fold()
            sfs_list.append(list(fs))
            jsfs.append(fs)
        if jsfs:
            jsfs_list = self.jsfs_dadi(self, jsfs)
            return sfs_list, jsfs_list
        else:
            return sfs_list

    def jsfs_dadi(self, jsfs, fold=True):
        """Calculate the joint site freq spectrum with dadi.

        Parameters
        ----------
        jsfs : TYPE
            DESCRIPTION.
        fold : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        jsfs_list

        """
        jsfs_list = []
        for i, j in combinations(jsfs, 2):
            fs = dadi.Spectrum([i, j])
            if fold:
                fs = fs.fold()
            jsfsarray = np.zeros(23)
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
            jsfs_list.append(jsfsarray)
        return jsfs_list

# =============================================================================
# Below functions loop through the haps instead of the pops
# Direct to file writing is more efficient when there are lots of reps
# each stats will be printed to a different file in order. If you want them in
# a single file use paste fx in unix
# =============================================================================


def asfsStats(args, fold=False, rand=True, randn=100000):
    """

    Parameters
    ----------
    pos : TYPE
        DESCRIPTION.
    hap : TYPE
        DESCRIPTION.
    pops : TYPE
        DESCRIPTION.
    fold : TYPE, optional
        DESCRIPTION. The default is False.
    rand : TYPE, optional
        DESCRIPTION. The default is True.
    randn : TYPE, optional
        DESCRIPTION. The default is 100000.

    Returns
    -------
    asfs : TYPE
        DESCRIPTION.

    """
    pos, hap, pops = args
    if type(hap) is list:
        gt = (allel.HaplotypeArray(np.vstack(hap), dtype='i1'))
    else:
        gt = (allel.HaplotypeArray(hap))
    aSFS12 = []
    aSFS = []
    for pop in pops:
        gtpop = gt.take(pop, axis=1)
        acpop = gtpop.count_alleles()
        seg = acpop.is_segregating()
        try:
            gtseg = gtpop.compress(seg)
            # random snps
            if rand:
                n = randn  # number of SNPs to choose randomly
                try:
                    vidx = np.random.choice(gtseg.shape[0], n, replace=False)
                    vidx.sort()
                    gtp = gtseg.take(vidx, axis=0)
                except ValueError:
                    gtp = gtseg
            else:
                gtp = gtseg
            # sfs
            if fold:
                sfsp = (allel.sfs_folded(gtp.count_alleles()))
            else:
                sfsp = (allel.sfs(gtp.count_alleles()[:, 1]))
            tots = np.sum(sfsp)
        except ValueError:
            # no segregating snps
            tots = 1
            sfsp = [0]*len(pop)
        try:
            aSFS12.append(sfsp[1]/tots)
        except IndexError:
            aSFS12.append(0)
        try:
            aSFS12.append(sfsp[2]/tots)
        except IndexError:
            aSFS12.append(0)
        aSFS.append(sfsp)
    asfs = " ".join(map(str, aSFS12))
    return f"{asfs}\n"


def summarizejsfs(fs):
    """Create summary jsfs.

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


def jsfsStats(args, fold=False, rand=True, randn=100000):
    """

    Parameters
    ----------
    pos : TYPE
        DESCRIPTION.
    hap : TYPE
        DESCRIPTION.
    pops : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    fold : TYPE, optional
        DESCRIPTION. The default is False.
    rand : TYPE, optional
        DESCRIPTION. The default is True.
    randn : TYPE, optional
        DESCRIPTION. The default is 100000.

    Returns
    -------
    jsfs : TYPE
        DESCRIPTION.

    """
    pos, hap, pops, pairs = args
    if type(hap) is list:
        gt = allel.HaplotypeArray(np.vstack(hap), dtype='i1')
    else:
        gt = allel.HaplotypeArray(hap)
    jsfs_list = []
    for pair in pairs:
        i, j = pair.split("-")
        p1 = pops[int(i)]
        p2 = pops[int(j)]
        gtpops = gt.take(p1+p2, axis=1)
        acpops = gtpops.count_alleles()
        segpops = acpops.is_segregating()
        gtseg = gtpops.compress(segpops)
        # random snps
        if rand:
            n = randn  # number of SNPs to choose randomly
            try:
                vidx = np.random.choice(gtseg.shape[0], n, replace=False)
                vidx.sort()
                gtr = gtpops.take(vidx, axis=0)
            except ValueError:
                gtr = gtseg
        else:
            gtr = gtseg
        gtpop1 = gtr.take(range(len(p1)), axis=1)
        gtpop2 = gtr.take(range(len(p1), gtr.shape[1]), axis=1)
        ac1 = gtpop1.count_alleles()
        ac2 = gtpop2.count_alleles()
        # jsfs
        if fold:
            # pad for allel as well
            popsizeA, popsizeB = len(p1)/2, len(p2)/2
            jsfs = allel.joint_sfs_folded(ac1, ac2)
            fs = np.resize(jsfs, (popsizeA+1, popsizeB+1))
        else:
            # pad for allel as well
            popsizeA, popsizeB = len(p1), len(p2)
            jsfs = allel.joint_sfs(ac1[:, 1], ac2[:, 1])
            fs = np.resize(jsfs, (popsizeA+1, popsizeB+1))
        props = summarizejsfs(fs)
        jsfs_list.append(props)
    jsfs = " ".join(map(str, np.concatenate(jsfs_list).ravel()))
    return f"{jsfs}\n"


def calc_afibs(gt, pos, pops, basepairs, fold):
    """Calculate afibs.

    Parameters
    ----------
    gt : TYPE
        DESCRIPTION.
    pos : TYPE
        DESCRIPTION.
    basepairs : TYPE
        DESCRIPTION.
    fold : TYPE
        DESCRIPTION.

    Returns
    -------
    afibs_list : TYPE
        DESCRIPTION.

    """
    subsample = 'all'
    afibs = {}
    for i, pop in enumerate(pops):
        ibsarray = np.zeros((len(pop), len(pop)-1))
        gtpop = gt.take(pop, axis=1)
        acpop = gtpop.count_alleles()
        seg = acpop.is_segregating()
        gtseg = gtpop.compress(seg)
        # ac = gtseg.count_alleles()
        poslist = pos[seg]
        freqlist = np.sum(gtseg, axis=1)
        if subsample == 'all':
            inds_list = range(len(pop))
        else:
            inds_list = np.random.choice(len(pop), subsample, replace=False)
        for ind in inds_list:
            indpos = poslist[gtseg[:, ind] > 0]
            indpos = np.insert(indpos, 0, 0)
            indpos = np.insert(indpos, len(indpos), basepairs)
            for freq in range(1, len(pop)):
                ibs = 0
                mut_ix = np.where(freqlist == freq)[0]
                if len(mut_ix) > 0:
                    for m in mut_ix:
                        start = bisect.bisect_left(indpos, poslist[m])
                        end = bisect.bisect_right(indpos, poslist[m])
                        ibs += indpos[end] - indpos[start - 1]
                    ibsarray[ind, freq-1] = (ibs / len(mut_ix))
        ibsarray[ibsarray == 0] = np.nan
        afibs[i] = np.nanmean(ibsarray, axis=0)
    # fold and export to list
    afibs_list = []
    if fold:
        for i, pop in enumerate(pops):
            ibs_flip = np.flip(afibs[i], axis=0)
            ibs_fold = (ibs_flip + afibs[i]) / 2
            haps = int(len(pop)/2)
            ibs = ibs_fold[0:haps]
            afibs_list.append(ibs)
    else:
        for i in range(len(pops)):
            afibs_list.append(afibs[i])
    return afibs_list


def afibsStats(args, fold=False):
    """
    For each individual within a population calculate up/down distance to
    nearest SNP for each frequency class of alleles. Average among individuals
    within a population and return a vector of length pop-2 (no fixed classes)
    of sizes for each freq class.

    Parameters
    ----------
    pos : TYPE
        DESCRIPTION.
    hap : TYPE
        DESCRIPTION.
    pops : TYPE
        DESCRIPTION.
    basepairs : TYPE
        DESCRIPTION.
    fold : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    afibs : TYPE
        DESCRIPTION.

    """
    pos, hap, pops, basepairs = args
    if type(hap) is list:
        afibslist = []
        for sub_rep in list(zip(pos, hap)):
            pos, gtarr = sub_rep
            gt = allel.HaplotypeArray(gtarr)
            afibslist.append(calc_afibs(gt, pos, pops, basepairs, fold))
        afibs_zip = list(zip(*afibslist))
        afibsmean = [np.mean(x, axis=0) for x in afibs_zip]
    else:
        gt = allel.HaplotypeArray(hap)
        pos = pos[0]
        afibsmean = calc_afibs(gt, pos, pops, basepairs, fold)
    afibs = " ".join(map(str, [i for t in afibsmean for i in t]))
    return f"{afibs}\n"


def filetStats(args):
    """

    Parameters
    ----------
    pos : TYPE
        DESCRIPTION.
    hap : TYPE
        DESCRIPTION.
    pops : TYPE
        DESCRIPTION.
    basepairs : TYPE
        DESCRIPTION.
    filet_path : TYPE
        DESCRIPTION.
    block : TYPE
        DESCRIPTION.

    Returns
    -------
    filet_list : TYPE
        DESCRIPTION.

    """
    pos, hap, pops, basepairs, filet_path, block = args

    if basepairs > 100000:
        block = 100000
    if block == 0:
        block = basepairs

    keep_stats = np.array([True, True, True, True, False, True, False, False,
                           True, True, True, True, True, False, True, False,
                           False, True, True, True, True, True, True, True,
                           True, True, True, True, False, False, False])
    norm = np.array([block, block**2, block, block, 1, 1, block, block**2,
                     block, block, 1, 1, 1, 1, block, block, block, 1, 1, 1, 1, 1])
    filet_list = []
    for pop1, pop2 in combinations(pops, 2):
        fakems_haps = []
        n1 = len(pop1)
        n2 = len(pop2)
        if type(hap) is list:
            loci_r = 0
            for sub_rep in list(zip(pos, hap)):
                posr, gtarr = sub_rep
                gt = allel.HaplotypeArray(gtarr)
                gtpops = gt.take(pop1+pop2, axis=1)
                acpops = gtpops.count_alleles()
                segpops = acpops.is_segregating()
                gtseg = gtpops.compress(segpops)
                posit = posr[segpops]
                #
                if basepairs > block:
                    start = 0
                    step = block
                    end = start + step
                    while end < basepairs:
                        loci_r += 1
                        s_ix = bisect.bisect_left(posit, start)
                        e_ix = bisect.bisect_right(posit, end) - 1
                        posit_block = posit[s_ix:e_ix] / basepairs
                        gtseg_block = gtseg[s_ix:e_ix]
                        seg = gtseg_block.shape[0]
                        fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit_block))}\n")
                        for geno in gtseg_block.transpose():
                            fakems_haps.append(f"{''.join(map(str, geno))}\n")
                        start += step
                        end += step
                #
                else:
                    loci_r = len(sub_rep)
                    posit = posit / block
                    seg = np.count_nonzero(segpops)
                    fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit))}\n")
                    for geno in gtseg.transpose():
                        fakems_haps.append(f"{''.join(map(str, geno))}\n")
        else:
            gt = allel.HaplotypeArray(hap)
            posr = pos[0]
            gtpops = gt.take(pop1+pop2, axis=1)
            acpops = gtpops.count_alleles()
            segpops = acpops.is_segregating()
            gtseg = gtpops.compress(segpops)
            posit = posr[segpops]
            #
            if basepairs > block:
                loci_r = 0
                start = 0
                step = block
                end = start + step
                while end < basepairs:
                    loci_r += 1
                    s_ix = bisect.bisect_left(posit, start)
                    e_ix = bisect.bisect_right(posit, end) - 1
                    posit_block = posit[s_ix:e_ix] / basepairs
                    gtseg_block = gtseg[s_ix:e_ix]
                    seg = gtseg_block.shape[0]
                    fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit_block))}\n")
                    for geno in gtseg_block.transpose():
                        fakems_haps.append(f"{''.join(map(str, geno))}\n")
                    start += step
                    end += step
            #
            else:
                loci_r = 1
                posit = posit / block
                seg = np.count_nonzero(segpops)
                fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit))}\n")
                for geno in gtseg.transpose():
                    fakems_haps.append(f"{''.join(map(str, geno))}\n")
        fakems_head = f"ms {n1+n2} {loci_r} -t tbs -r tbs {block} -I 2 {n1} {n2}\n1234\n"
        fakems = "".join(fakems_haps)
        msinput = fakems_head + fakems
        filet_prog = os.path.join(filet_path, "twoPopnStats_forML")
        cmd = [filet_prog, str(n1), str(n2)]
        proc = run(cmd, stdout=PIPE, input=msinput, encoding='ascii', check=True)
        # collect stats
        lstats = proc.stdout.rstrip().split('\n')[1:]
        stat_vec = [list(map(float, l.split())) for l in lstats]
        if len(stat_vec) > 1:
            stat_arr = np.vstack(stat_vec)
            stat_arr[np.isinf(stat_arr)] = 'nan'
            filetmean = np.nanmean(stat_arr, axis=0)
            filet_norm = filetmean[keep_stats] / norm
        else:
            stat_arr = np.array(stat_vec[0])[keep_stats]
            stat_arr[np.isinf(stat_arr)] = 'nan'
            filet_norm = stat_arr / norm
        filet_list.append(" ".join(map(str, filet_norm)))
    return f"{' '.join(filet_list)}\n"
