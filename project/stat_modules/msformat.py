#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 20:18:46 2020
@author: Scott T. Small

Reading and parsing ms-style formatted files for input into abc_stats.py

"""
import numpy as np
import bisect
import os

def getSnpsOverflowingChr(newPositions, totalPhysLen):
    """

    Parameters
    ----------
    newPositions : TYPE
        DESCRIPTION.
    totalPhysLen : TYPE
        DESCRIPTION.

    Returns
    -------
    overflowers : TYPE
        DESCRIPTION.

    """
    overflowers = []
    for i in reversed(range(len(newPositions))):
        if newPositions[i] > totalPhysLen:
            overflowers.append(newPositions[i])
    return overflowers


def fillInSnpSlotsWithOverflowers(newPositions, totalPhysLen, overflowers):
    """

    Parameters
    ----------
    newPositions : TYPE
        DESCRIPTION.
    totalPhysLen : TYPE
        DESCRIPTION.
    overflowers : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    posH = {}
    for pos in newPositions:
        posH[pos] = 1
    for i in range(len(overflowers)):
        del newPositions[-1]
    for pos in reversed(range(1, totalPhysLen+1)):
        if pos not in posH:
            bisect.insort_left(newPositions, pos)
            overflowers.pop()
            if len(overflowers) == 0:
                break


def discrete_positions(positions, totalPhysLen):
    """

    Parameters
    ----------
    positions : TYPE
        DESCRIPTION.
    totalPhysLen : TYPE
        DESCRIPTION.

    Returns
    -------
    newPositions : TYPE
        DESCRIPTION.

    """
    snpNum = 1
    prevPos = -1
    prevIntPos = -1
    newPositions = []
    for position in positions:
        assert position >= 0 and position <= 1., "Mutations positions must all be in [0, 1)"
        assert position >= prevPos
        origPos = position
        if position == prevPos:
            position += 0.000001
        prevPos = origPos

        intPos = int(totalPhysLen*position)
        if intPos == 0:
            intPos = 1
        if intPos <= prevIntPos:
            intPos = prevIntPos + 1
        prevIntPos = intPos
        newPositions.append(intPos)
    overflowers = getSnpsOverflowingChr(newPositions, totalPhysLen)
    if overflowers:
        fillInSnpSlotsWithOverflowers(newPositions, totalPhysLen, overflowers)
    assert len(newPositions) == len(positions)
    assert all(newPositions[i] <= newPositions[i+1]
               for i in range(len(newPositions)-1))
    assert newPositions[-1] <= totalPhysLen
    return newPositions


def ms_parse(infile):
    """Parse ms-type file.

    Parameters
    ----------
    infile : TYPE
        DESCRIPTION.

    Returns
    -------
    msdict : Dict
        DESCRIPTION.

    """
    msdict = {}
    hap_list = []
    pos_list = []
    with open(infile) as ms:
        line = next(ms)
        ms_head = line.split("-")
        if "ms" in ms_head[0]:
            prog, nchrs, reps = ms_head[0].split()
            r_ix = [i for i, r in enumerate(ms_head) if r.startswith("r")][0]
            r, rho, basepairs = ms_head[r_ix].split()
            basepairs, nchrs = int(basepairs), int(nchrs)
            p_ix = [i for i, p in enumerate(ms_head) if p.startswith("I")][0]
            p, subpops, *subpopsizes = ms_head[p_ix].split()
            subpopsizes = list(map(int, subpopsizes))
        elif "discoal" in ms_head[0]:
            prog, nchrs, reps, basepairs = ms_head[0].split()
            basepairs, nchrs = int(basepairs), int(nchrs)
            p_ix = [i for i, p in enumerate(ms_head) if p.startswith("p")][0]
            p, subpops, *subpopsizes = ms_head[p_ix].split()
            subpopsizes = list(map(int, subpopsizes))
        ind = 0
        popconfig = []
        for pop in subpopsizes:
            popconfig.append(list(range(ind, pop+ind)))
            ind += pop
        rep_haplist = []
        rep_poslist = []
        for line in ms:
            if line.startswith(prog):
                hap_list.append(rep_haplist)
                pos_list.append(rep_poslist)
                rep_haplist = []
                rep_poslist = []
            elif line.startswith("seg"):
                seg, seg_sites = line.strip().split()
                seg_sites = int(seg_sites)
                # no seg sites
                if seg_sites == 0:
                    new_pos = []
                    hap_arr = []
                    for i in range(nchrs):
                        hap_arr.append([])
                else:
                    line = next(ms)
                    if line.startswith("positions"):
                        positions = line.strip().split()
                        pos_arr = np.array(positions[1:], dtype=np.float64)
                        new_pos = discrete_positions(pos_arr, basepairs)
                        # haps line
                        hap_arr = np.zeros((nchrs, pos_arr.shape[0]), dtype=np.uint8)
                        for cix in range(nchrs):
                            line = next(ms)
                            line = list(line.strip())
                            hap_arr[cix, :] = np.array(line, dtype=np.uint8)
                rep_poslist.append(np.array(new_pos))
                rep_haplist.append(hap_arr)
    hap_list.append(rep_haplist)
    pos_list.append(rep_poslist)
    msdict = {"pops": popconfig,
              "basepairs": basepairs,
              "pos": pos_list,
              "haps": hap_list}
    return msdict


def split2pairs(msdict, outfile, p1, p2):
    """Split msfile into population pairs.

    Parameters
    ----------
    infile : TYPE
        DESCRIPTION.
    pair : TYPE
        DESCRIPTION.

    Returns
    -------
    loci : TYPE
        DESCRIPTION.
    basepairs : TYPE
        DESCRIPTION.

    """
    mshead = msdict["head"].copy()
    infile = msdict["infile"]
    loci = msdict["loci"]
    basepairs = msdict["basepairs"]
    seed = msdict["seed"]
    popconfig = msdict["pops"]
    seg = msdict["seg"]
    pos = msdict["pos"]
    haps = msdict["haps"]
    outfile_name = f"{p1}-{p2}.{outfile}"
    if os.path.exists(outfile_name):
        infile_name = os.path.split(infile)[-1]
        outfile_name = f"{p1}-{p2}.{infile_name}.{outfile}"
    with open(outfile_name, 'w') as msout:
        if "ms" in mshead[0]:
            prog, nchrs, reps = mshead[0].split()
            nchrs = int(nchrs)
            # r_ix = [i for i, r in enumerate(mshead) if r.startswith("r")][0]
            # r, rho, basepairs = mshead[r_ix].split()
            p_ix = [i for i, p in enumerate(mshead) if p.startswith("I")][0]
            p, subpops, *subpopsizes = mshead[p_ix].split()
            subpopsizes = list(map(int, subpopsizes))
            # rewrite header
            n1 = subpopsizes[p1]
            n2 = subpopsizes[p2]
            new_size = n1 + n2
            new_subpopsizes = [0]*len(subpopsizes)
            new_subpopsizes[p1] = n1
            new_subpopsizes[p2] = n2
            mshead[0] = f"{prog} {new_size} {loci} "
            mshead[p_ix] = f"I {subpops} {' ' .join(map(str, new_subpopsizes))} "
        elif "discoal" in mshead[0]:
            prog, nchrs, reps, *_ = mshead[0].split()
            nchrs = int(nchrs)
            p_ix = [i for i, p in enumerate(mshead) if p.startswith("p")][0]
            p, subpops, *subpopsizes = mshead[p_ix].split()
            subpopsizes = list(map(int, subpopsizes))
            # rewrite header
            pop1 = subpopsizes[p1]
            pop2 = subpopsizes[p2]
            new_size = pop1 + pop2
            new_subpopsizes = [0]*len(subpopsizes)
            new_subpopsizes[p1] = pop1
            new_subpopsizes[p2] = pop2
            mshead[0] = f"{prog} {new_size} {loci} {basepairs} "
            mshead[p_ix] = f"p {subpops} {' ' .join(map(str, new_subpopsizes))} "
        msout.write(f"{'-'.join(mshead)}")
        msout.write(f"{seed}\n")
        # write geno data
        pop1 = popconfig[p1]
        pop2 = popconfig[p2]
        for po, hapmat in zip(pos, haps):
            gt = hapmat[pop1 + pop2]
            seg_pos = np.sum(gt, axis=0)
            seg_mask = (seg_pos > 0) & (seg_pos < (n1 + n2))
            seg = np.count_nonzero(seg_mask)
            posit = po[seg_mask] / basepairs
            gt_seg = gt[:, seg_mask]
            msout.write(f"\n//\nsegsites: {seg}\npositions: {' '.join(map(str, posit))}\n")
            for h in gt_seg:
                msout.write(f"{''.join(map(str, h))}\n")
    return None
