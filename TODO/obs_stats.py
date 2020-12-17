#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: stsmall
"""

import allel
from subprocess import run, PIPE
import numpy as np
import os
import gzip
import bisect
from itertools import combinations


def readSampleToPopFile(sampleToPopFileName):
    table = {}
    with open(sampleToPopFileName) as sampleToPopFile:
        for line in sampleToPopFile:
            sample, pop = line.strip().split()
            table[sample] = pop
    return table


def readMaskDataForScan(maskFileName, chrArm):
    isAccessible = []
    readingMasks = False
    if maskFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    with fopen(maskFileName, 'rt') as maskFile:
        for line in maskFile:
            if line.startswith(">"):
                currChr = line[1:].strip()
                if currChr == chrArm:
                    readingMasks = True
                elif readingMasks:
                    break
            else:
                if readingMasks:
                    for char in line.strip():
                        if char == 'N':
                            isAccessible.append(False)
                        elif char.islower():
                            isAccessible.append(False)
                        else:
                            isAccessible.append(True)
    return isAccessible


def makeAncArray(calls, pos, chr_arm, anc_fasta):
    ref_allele = calls['variants/REF'].subset(sel0=pos)
    anc_list = []
    fa = []
    readingMasks = False
    if anc_fasta.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    with fopen(anc_fasta, 'rt') as anc:
        for line in anc:
            if line.startswith(">"):
                currChr = line[1:].strip()
                if currChr == chr_arm:
                    readingMasks = True
                elif readingMasks:
                    break
            else:
                if readingMasks:
                    fa.append(line.strip())
    seq = "".join(fa)
    for i, p in enumerate(pos):
        char = seq[p]
        if char == "N":
            print(f"Masked ancestral allele at base: {p}")
        elif char == ref_allele[i]:
            anc_list.append(0)
        else:
            anc_list.append(1)
    return np.array(anc_list)


def asfsObsStats(args, fold=False, rand=True, randn=10000):
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
    pos, gt, pops = args
    aSFS12 = []
    aSFS = []
    for pop in pops:
        # gtpop = gt.take(pop, axis=1)
        gtpop = gt.subset(sel1=pop)
        miss_count = gtpop.count_missing(axis=1)
        miss_arr = miss_count == 0
        gtpop = gtpop.compress(miss_arr, axis=0)
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
            sfsp = [0]*len(pop)*2
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


def jsfsObsStats(args, fold=False, rand=True, randn=10000):
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
    pos, gt, pops, = args
    jsfs_list = []
    for p1, p2 in combinations(pops, 2):
        gtpops = gt.take(p1+p2, axis=1)
        miss_count = gtpops.count_missing(axis=1)
        miss_arr = miss_count == 0
        gtpops = gtpops.compress(miss_arr, axis=0)
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
        # jsfs = allel.joint_sfs(ac1[:, 1], ac2[:, 1])
        # jsfs
        if fold:
            # pad for allel as well
            popsizeA, popsizeB = len(p1), len(p2)
            jsfs = allel.joint_sfs_folded(ac1, ac2)
            fs = np.resize(jsfs, (popsizeA+1, popsizeB+1))
        else:
            # this is gt array so count is in diploid not haps
            popsizeA, popsizeB = len(p1)*2, len(p2)*2
            jsfs = allel.joint_sfs(ac1[:, 1], ac2[:, 1])
            fs = np.resize(jsfs, (popsizeA+1, popsizeB+1))
        props = summarizejsfs(fs)
        jsfs_list.append(props)
    jsfs = " ".join(map(str, np.concatenate(jsfs_list).ravel()))
    return f"{jsfs}\n"


def calc_afibs(gt, pos, pops, window, chrlen, fold):
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
    block, step = window
    if step == 0:
        step = block
    subsample = 'all'
    afibs = {}
    for i, pop in enumerate(pops):
        haps = len(pop)*2
        ibsarray = np.zeros((haps, haps-1))
        gtpop = gt.subset(sel1=pop)
        miss_count = gtpop.count_missing(axis=1)
        miss_arr = miss_count == 0
        gtpop = gtpop.compress(miss_arr, axis=0)
        pos = pos[miss_arr]
        acpop = gtpop.count_alleles()
        seg = acpop.is_segregating()
        gtseg = gtpop.compress(seg)
        gthap = gtseg.to_haplotypes()
        poslist = pos[seg]
        freqlist = np.sum(gthap, axis=1)
        if subsample == 'all':
            inds_list = range(haps)
        else:
            inds_list = np.random.choice(haps, subsample, replace=False)
        for ind in inds_list:
            indpos = poslist[gthap[:, ind] > 0]
            indpos = np.insert(indpos, 0, 0)
            indpos = np.insert(indpos, len(indpos), chrlen)
            reps = 0
            start = 0
            step = step
            end = start + step
            while end < chrlen:
                try:
                    loc = poslist.locate_range(start, end)
                    for freq in range(1, haps):
                        ibs = 0
                        mut_ix = np.where(freqlist[loc] == freq)[0]
                        if len(mut_ix) > 0:
                            for m in mut_ix:
                                ix = bisect.bisect_left(indpos, poslist[m])
                                ibs += indpos[ix] - indpos[ix - 1]
                            ibsarray[ind, freq-1] += (ibs / len(mut_ix))
                            reps += 1
                        # except ZeroDivisionError:
                        #     # nothing in that freq class
                        #     ibsarray[ind, freq-1] += 0
                except KeyError:
                    pass
                start += step
                end += step
            ibsarray[ind, :] *= (1/reps)
        ibsarray[ibsarray == 0] = np.nan
        afibs[i] = np.nanmean(ibsarray, axis=0)
    # fold and export to list
    afibs_list = []
    if fold:
        for i, pop in enumerate(pops):
            ibs_flip = np.flip(afibs[i], axis=0)
            ibs_fold = (ibs_flip + afibs[i]) / 2
            haps = int(len(pop))
            ibs = ibs_fold[0:haps]
            afibs_list.append(ibs)
    else:
        for i in range(len(pops)):
            afibs_list.append(afibs[i])
    return afibs_list


def afibsObsStats(args, fold=False):
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
    pos, gt, pops, window, chrlen = args
    afibsmean = calc_afibs(gt, pos, pops, window, chrlen, fold)
    afibs = " ".join(map(str, [i for t in afibsmean for i in t]))
    return f"{afibs}\n"

# TODO: filet, add SortedIndex, locate_range(), dxymin not > 1
def filetObsStats(args, unmasked, anc_arr, unmskfrac, window, filet_path):
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
    pos, gt, pops, chrlen = args
    block, step = window
    if block > 100000:
        block = 100000
    if step == 0:
        step = block
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
        gtpops = gt.take(pop1+pop2, axis=1)
        acpops = gtpops.count_alleles()
        segpops = acpops.is_segregating()
        gtseg = gtpops.compress(segpops)
        posit = pos[segpops]
        anc = anc_arr[segpops]
        #
        loci_r = 0
        start = 0
        step = block
        end = start + step
        while end < chrlen:
            loci_r += 1
            s_ix = bisect.bisect_left(posit, start)
            e_ix = bisect.bisect_right(posit, end) - 1
            if (np.count_nonzero(unmasked[s_ix:e_ix]) / block) > unmskfrac:
                posit_block = posit[s_ix:e_ix]
                gtseg_block = gtseg.subset(sel0=posit_block)
                # seg = gtseg_block.shape[0]
                seg = gtseg_block.n_variants
                ms_pos = posit_block / (block + end)
                fakems_haps.append(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, ms_pos))}\n")
                if np.count_nonzeros(anc[s_ix:e_ix]) > 0:
                    flip_pol = np.where(anc[s_ix:e_ix] > 0)[0]
                    for geno in gtseg_block.transpose():
                        for al in flip_pol:
                            if geno[al] == 1:
                                geno[al] = 0
                            else:
                                geno[al] = 1
                        fakems_haps.append(f"{''.join(map(str, geno))}\n")
                else:
                    for geno in gtseg_block.transpose():
                        fakems_haps.append(f"{''.join(map(str, geno))}\n")
            start += step
            end += step
        fakems_head = f"ms {n1+n2} {loci_r} -t tbs -r tbs {block} -I 2 {n1} {n2}\n1234\n"
        fakems = "".join(fakems_haps)
        msinput = fakems_head + fakems
        filet_prog = os.path.join(filet_path, "twoPopnStats_forML")
        cmd = [filet_prog, str(n1), str(n2)]
        proc = run(cmd, stdout=PIPE, input=msinput, encoding='ascii', check=True)
        # collect stats
        lstats = proc.stdout.rstrip().split('\n')[1:]
        stat_vec = [list(map(float, l.split())) for l in lstats]
        stat_arr = np.array(stat_vec[0])[keep_stats]
        stat_arr[np.isinf(stat_arr)] = 'nan'
        filet_norm = stat_arr / norm
        filet_list.append(" ".join(map(str, filet_norm)))
    return f"{' '.join(filet_list)}\n"
