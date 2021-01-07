#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:27:35 2021
@author: Scott T. Small
adopted from script by Konrad Lohse.

Each window is chopped up into blocks of a fixed physical block_length.
Each block is summarised by the (folded/unfolded) SFS.
[(1, 2), 1] # folded SFS of the block
e.g., above there is exactly one subsample which contains 1 singleton and 2 doubleton
mutations.

"""

import itertools
from collections import Counter
import numpy as np


def makeblock_ali(pos, haps, blockL, winL):
    blockdic = {}
    blocklist = []
    for i, p in enumerate(pos):
        blocklist.append([p, tuple(haps[:, i])])
    for i in range(blockL, winL+1, blockL):
        new_bl_list = list(itertools.takewhile(lambda v: v <= i, pos))
        new_bl_listb = list(itertools.dropwhile(lambda v: v <= i-blockL, new_bl_list))
        if new_bl_listb == []:
            new_bl_list_fin = []
        else:
            new_bl_list_fin = list(zip(*new_bl_listb))[1]
        blockdic.update({tuple([i-blockL, i]): new_bl_list_fin})
    return blockdic


def jSFSfold(MutCountsDict, l1):
    # works out the number of freq types in each pop. The SFS is folded to the minor allele in pop1:
    # makes tuple pairs of equivalent mut types:
    tuplist = []
    for i in range(l1+1):
        tuplist.append(tuple(sorted([i, l1-i])))  # folding here
    myNewCounter = {}
    # set() removes duplicates
    for i in set(tuplist):
        if i[0] == i[1]:
            myNewCounter.update({i[0]: MutCountsDict[i[0]]})
        else:
            myNewCounter.update({i[0]: MutCountsDict[i[0]] + MutCountsDict[i[1]]})
    # turns the bSFS into a tuple, removes monomorphic sites:
    sortdkeys = sorted(myNewCounter.keys())
    del sortdkeys[0]
    blockconf = tuple([myNewCounter[i] for i in sortdkeys])

    return blockconf

def jSFSunfold(MutCountsDict, l1):
    myNewCounter = {}
    for i in range(l1+1):
        myNewCounter.update({i: MutCountsDict[i]})
    sortdkeys = sorted(myNewCounter.keys())
    del sortdkeys[0]
    blockconf = tuple([myNewCounter[i] for i in sortdkeys])

    return blockconf


def bSFSconfig(mutl, comb):
    if mutl == []:
        jSFS = []
    else:
        ali = list(zip(*mutl))
        ali1 = [map(int, ali[i]) for i in comb]
    # freq of '1' (which may be a true or arbitrary polarization) in each pop sample:
        count1 = [sum(i) for i in list(zip(*ali1))]
        jSFS = count1
    return Counter(jSFS)


def multibSFSall(block_dt, combos, l1, fold=True):
    ll = []
    for i in block_dt.keys():
        if fold:
            multibSFSlist = Counter([jSFSfold(bSFSconfig(block_dt[i], u), l1) for u in combos])
        else:
            multibSFSlist = Counter([jSFSunfold(bSFSconfig(block_dt[i], u), l1) for u in combos])
        counts = [[u, multibSFSlist[u]] for u in sorted(multibSFSlist.keys())]
        ll.append([counts])
    return ll


def blocksfs(p1, haps, pos, block_len, length_bp):
    block_dt = makeblock_ali(pos, haps, block_len, length_bp)
    combos = list(itertools.combinations(range(p1), p1))
    bsfsout = multibSFSall(block_dt, combos, p1)
    bsfs = np.array([i[0] for i in bsfsout], dtype=object)

    return bsfs
