#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import sys, itertools
import msprime
import numpy as np
from collections import Counter

"""
This script simulates contiguous windows of sequence in msprime (an example demography of a population bottleneck is hardcoded in the function main). 
Each window is chopped up into blocks of a fixed physical length. 
Each block is summarised by the (folded) SFS.

The syntax is:
For example ./pop1_block_sims.py 1blockLength WindowLength num_replicates subsample_size  sample_size

For example ./pop1_block_sims.py 100 1000 10 4 4  
simulates 10 replicates each consisting of a 1kb window which is cut into 100bp blocks for a sample and subsample size of n=4
The output is a list containing elements for each block of the type: {{3, 600, 700}, {{{1, 2}, 1}}}. 
The first sublist specifies the block coordinates, (the example-block above is the 4th simulation replicate, position 600-700). 
The second sublist is the folded SFS of the block (i.e. there is exactly one subsample which contains 1 singleton and 2 doubleton mutations).

"""

def makeblock_ali(simreps, blockL, winL):
    blockdic = {}
    for j, tree_sequence in enumerate(simreps):
        blocklist = []
        for v in tree_sequence.variants():
            blocklist.append([int(round(v.position)),tuple(v.genotypes)])
        for i in range(blockL, winL+1, blockL):
            new_bl_list = list(itertools.takewhile(lambda v: v[0] <= i, blocklist))
            new_bl_listb = list(itertools.dropwhile(lambda v: v[0] <= i-blockL, new_bl_list))
            if new_bl_listb == []:
                new_bl_list_fin = []
            else:
                new_bl_list_fin = zip(*new_bl_listb)[1]
            blockdic.update({tuple([j,i-blockL,i]):new_bl_list_fin})
    return blockdic

def jSFSfolder(MutCountsDict,l1):
# works out the number of freq types in each pop. The SFS is folded to the minor allele in pop1:
# makes tuple pairs of equivalent mut types:
    tuplist=[]
    for i in range(l1+1):
        tuplist.append(tuple(sorted([i,l1-i])))
    myNewCounter={}
# set() removes duplicates
    for i in set(tuplist):
        if i[0] == i[1]:
            myNewCounter.update({i[0]:MutCountsDict[i[0]]})
        else:
            myNewCounter.update({i[0]:MutCountsDict[i[0]]+MutCountsDict[i[1]]})
# turns the bSFS into a tuple, removes monomorphic sites:
    sortdkeys = sorted(myNewCounter.keys())
    del sortdkeys[0]
    blockconf=tuple([myNewCounter[i] for i in sortdkeys])
    return blockconf

def multibSFSall(myBlockDict, combos, l1):
    ll=[]
    for i in myBlockDict.keys():
        multibSFSlist = Counter([jSFSfolder(bSFSconfig(myBlockDict[i], u), l1) for u in combos])
        counts = [[u, multibSFSlist[u]] for u in sorted(multibSFSlist.keys())]
        ll.append([i,counts])
    return ll

#  bSFS for one combo:
def bSFSconfig(mutl, comb):
    if mutl == []:
        jSFS = []
    else:
        ali = zip(*mutl)
        ali1 = [map(int, ali[i]) for i in comb]
    # freq of '1' (which may be a true or arbitrary polarization) in each pop sample:
        count1 = [sum(i) for i in zip(*ali1)]
        jSFS = count1
    return Counter(jSFS)

def main():
    blockL = int(sys.argv[1])
    winL = int(sys.argv[2])
    #num_replicates is the # of windows
    num_replicates = int(sys.argv[3])
    #nsub is the sizes of the subsample
    nsub = int(sys.argv[4])
    #nwhole is the sizes of the subsample
    nwhole = int(sys.argv[5])
    """
   Demographic parameters specifying a step change in Ne to Ne/scal in a single population at time T_bott
    """
    recr = 0
    scal = 0.5
    Ne = 4.05596e5
    T_bott = 2e5
    mu = 3.46e-9
    population_configurations = [msprime.PopulationConfiguration(sample_size=nwhole, initial_size=Ne)]
    demographic_events = [msprime.PopulationParametersChange(time=T_bott, initial_size=Ne/scal, population_id=0)]
# combos lists positions of subsampled inds:
    combos = list(itertools.combinations(range(nwhole),nsub))
# this generates the actual replicates
    replicates = msprime.simulate(
        num_replicates = num_replicates,
        length = winL, 
        recombination_rate = recr,
        population_configurations = population_configurations,
        demographic_events = demographic_events,
        mutation_rate = mu
        )
 
    myBlockDict = makeblock_ali(replicates,blockL,winL)
    out = multibSFSall(myBlockDict, combos, nsub)
    outmath = str(out).replace("[","{").replace("]","}").replace("(","{").replace(")","}")
    print(outmath)
if __name__ == "__main__":
    main()