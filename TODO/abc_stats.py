#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:05:00 2020
@author: Scott T. Small

Main script for generating statistics for ABC and ML training.
depends: parse_sims.py, sims_stats.py, obs_stats.py

Example
-------

abc_stats.py sim --infile ms.out --outfile ms.stats --pops 4 --pairs 0-1 0-2 0-3
    --stats sfs jsfs --mask --gff foo.gff --mode split-run

    --mask for FILET, expects names as 0-1.mask.txt 0-2.mask.txt ...
    --gff to avoid genes

abc_stats.py obs --infile vcf/h5 --fasta foo.mask.fa --gff foo.gff --pops 4
    --pairs 0-1 0-2 0-3 --stats filet --anc_fasta anc.fasta

    --fasta masked fasta for FILET masking
    --anc_fasta for FILET polarizing and unfolded SFS

Notes
-----

Creates summary stats from coalescent simulations: ms, msmove, discoal, msprime
There are two main modes: sims and obs

 Mode 'sims' is for ms-style formats in any file, those produced by abc_sims.py

 Mode 'obs' is for generating the same stats but with a starting vcf. Take a look
    at generating mask files from FILET and diploshic


Example
-------

    $ python abc_stats.py --infile --pairs --stats --out_file --split

"""
import sys
import argparse
from itertools import combinations
import multiprocessing
import glob
import os
import allel
import numpy as np
from math import ceil
from tqdm import tqdm
from timeit import default_timer as timer
from sim_parse import ms_parse, split2pairs
from sim_stats import SumStats, filetStats, asfsStats, jsfsStats, afibsStats
from obs_stats import readMaskDataForScan, readSampleToPopFile, makeAncArray
from obs_stats import asfsObsStats, jsfsObsStats, afibsObsStats, filetObsStats
# set globals
chunk = 100
chunkopt = True


def pair_split(msdict, out_file, pairs):
    """Generate pairs for splitting.

    Parameters
    ----------
    infile : TYPE
        DESCRIPTION.
    outfile : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for p in pairs:
        p1, p2 = p.split("-")
        p1, p2 = int(p1), int(p2)
        split2pairs(msdict, out_file, p1, p2)


def header_fx(pairs, pops, sfs=False, jsfs=False, afibs=False, filet=False, obs=False):
    """Create header for outfile.

    Parameters
    ----------
    pairs : TYPE
        DESCRIPTION.
    filet : TYPE, optional
        DESCRIPTION. The default is False.
    sfs : TYPE, optional
        DESCRIPTION. The default is False.
    jsfs : TYPE, optional
        DESCRIPTION. The default is False.
    afibs : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    header : TYPE
        DESCRIPTION.

    """
    asfs_default = "afS1 afS2"
    jsfs_default = "jfS1 jfS2 jfS3 jfS4 jfS5 jfS6 jfS7 " \
                    "jfS8 jfS9 jfS10 jfS11 jfS12 jfS13 "   \
                    "jfS14 jfS15 jfS16 jfS17 jfS18 jfS19 " \
                    "jfS20 jfS21 jfS22 jfS23"
    filet_default = "pi1 hetVar1 ss1 private1 tajd1 ZnS1 pi2 hetVar2 ss2 "    \
                    "private2 tajd2 ZnS2 Fst snn dxy_mean dxy_min gmin zx "  \
                    "dd1 dd2 ddRank1 ddRank2"
    # note that this FILET does not include haplotype or IBS stats

    # prep headers
    afibs_header = []
    asfs_header = []
    jsfs_header = []
    filet_header = []
    sub_pops = list(dict.fromkeys([item for x in [x.split("-") for x in pairs] for item in x]))

    if afibs:
        for i, p in enumerate(sub_pops):
            haps = len(pops[i])
            if obs:
                haps *= 2
            for j in range(1, haps+1):
                afibs_header.append(f"afL{j}_{p}")
    if sfs:
        for p in sub_pops:
            asfs_header.append(f"afS1_{p} afS2_{p}")
    if jsfs:
        for p in pairs:
            for j in range(1, 24):
                jsfs_header.append(f"jfS{j}_{p}")
    if filet:
        for p in pairs:
            for f in filet_default.split():
                filet_header.append(f"{f}_{p}")

    header = afibs_header + asfs_header + jsfs_header + filet_header

    return " ".join(header)


def write_stats_out(ms, outdir, pairs, pops, asfsdict, jsfsdict, afibsdict, filetdict):
    """Write outstats from sim_pops.

    Parameters
    ----------
    ms : TYPE
        DESCRIPTION.
    outdir : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    asfsdict : TYPE
        DESCRIPTION.
    jsfsdict : TYPE
        DESCRIPTION.
    afibsdict : TYPE
        DESCRIPTION.
    filetdict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dd = [asfsdict, jsfsdict, afibsdict, filetdict]
    s, j, a, f = [True if i else False for i in dd]
    header = header_fx(pairs, pops, s, j, a, f)
    # write
    for d in dd:
        if d:
            reps = list(d.keys())
            break
    out_file = os.path.join(outdir, f"{ms}.stats")
    if os.path.exists(out_file):
        mode = 'a'
    else:
        mode = 'w'

    with open(out_file, mode) as out:
        out.write(f"##{ms}\n")
        out.write(f"{header}\n")
        for rep in reps:
            line = []
            for d in [asfsdict, jsfsdict, afibsdict]:
                if d:
                    line.append(d[rep])
            if filetdict:
                line.append(" ".join(filetdict[rep]))
            out.write(f"{' '.join(line)}\n")


def calc_sfs(ms, outdir, pairs, sum_stats, nprocs):
    """Calculte sfs per rep and write directly to file.

    Parameters
    ----------
    out_file : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    sum_stats : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    argslist = []
    for pos, hap in zip(sum_stats.pos, sum_stats.haparr):
        argslist.append([pos, hap, sum_stats.pops])

    header = header_fx(pairs, "", sfs=True)
    out_file = os.path.join(outdir, f"{ms}.sfs")
    if os.path.exists(out_file):
        mode = 'a'
    else:
        mode = 'w'
    with open(out_file, mode) as out:
        out.write(f"##{ms}\n")
        out.write(f"{header}\n")
        if nprocs == 1:
            for args in argslist:
                sfs_list = asfsStats(args)
                out.write(f"{sfs_list}")
        else:
            # number of chunks
            nk = chunk
            # check that there are not more requested than available
            if nprocs > multiprocessing.cpu_count():
                nprocs = multiprocessing.cpu_count()
            # set pool and map
            pool = multiprocessing.Pool(nprocs)
            chunk_list = [argslist[i:i + nk] for i in range(0, len(argslist), nk)]
            if chunkopt:
                chunksize = ceil(len(chunk_list[0])/nprocs)
            else:
                chunksize = 1
            for args in chunk_list:
                sfs_list = pool.map(asfsStats, args, chunksize=chunksize)
                out.write(f"{''.join(sfs_list)}")
            pool.close()


def calc_jsfs(ms, outdir, pairs, sum_stats, nprocs):
    """Calculte jsfs per rep and write directly to file.

    Parameters
    ----------
    out_file : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    sum_stats : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    argslist = []
    for pos, hap in zip(sum_stats.pos, sum_stats.haparr):
        argslist.append([pos, hap, sum_stats.pops, pairs])

    header = header_fx(pairs, "", jsfs=True)
    out_file = os.path.join(outdir, f"{ms}.jsfs")
    if os.path.exists(out_file):
        mode = 'a'
    else:
        mode = 'w'
    with open(out_file, mode) as out:
        out.write(f"##{ms}\n")
        out.write(f"{header}\n")
        if nprocs == 1:
            for args in argslist:
                jsfs_list = jsfsStats(args)
                out.write(f"{jsfs_list}")
        else:
            # number of chunks
            nk = chunk
            # check that there are not more requested than available
            if nprocs > multiprocessing.cpu_count():
                nprocs = multiprocessing.cpu_count()
            # set pool and map
            pool = multiprocessing.Pool(nprocs)
            chunk_list = [argslist[i:i + nk] for i in range(0, len(argslist), nk)]
            if chunkopt:
                chunksize = ceil(len(chunk_list[0])/nprocs)
            else:
                chunksize = 1
            for args in chunk_list:
                jsfs_list = pool.map(jsfsStats, args, chunksize=chunksize)
                out.write(f"{''.join(jsfs_list)}")
            pool.close()


def calc_afibs(ms, outdir, pairs, sum_stats, basepairs, nprocs):
    """Calculte jsfs per rep and write directly to file.

    Parameters
    ----------
    out_file : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    sum_stats : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    argslist = []
    for pos, hap in zip(sum_stats.pos, sum_stats.haparr):
        argslist.append([pos, hap, sum_stats.pops, basepairs])

    header = header_fx(pairs, sum_stats.pops, afibs=True)
    out_file = os.path.join(outdir, f"{ms}.afibs")
    if os.path.exists(out_file):
        mode = 'a'
    else:
        mode = 'w'
    with open(out_file, mode) as out:
        out.write(f"##{ms}\n")
        out.write(f"{header}\n")
        if nprocs == 1:
            for args in argslist:
                afibs_list = afibsStats(args)
                out.write(f"{afibs_list}")
        else:
            # number of chunks
            nk = chunk
            # check that there are not more requested than available
            if nprocs > multiprocessing.cpu_count():
                nprocs = multiprocessing.cpu_count()
            # set pool and map
            pool = multiprocessing.Pool(nprocs)
            chunk_list = [argslist[i:i + nk] for i in range(0, len(argslist), nk)]
            if chunkopt:
                chunksize = ceil(len(chunk_list[0])/nprocs)
            else:
                chunksize = 1
            for args in chunk_list:
                afibs_list = pool.map(afibsStats, args, chunksize=chunksize)
                out.write(f"{''.join(afibs_list)}")
            pool.close()


def calc_filet(ms, outdir, pairs, sum_stats, block, filet_path, nprocs, window):
    """Write directly to file.

    Parameters
    ----------
    out_file : TYPE
        DESCRIPTION.
    sum_stats : TYPE
        DESCRIPTION.
    block : TYPE
        DESCRIPTION.
    filet_path : TYPE
        DESCRIPTION.
    nprocs : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    argslist = []
    for pos, hap in zip(sum_stats.pos, sum_stats.haparr):
        argslist.append([pos, hap, sum_stats.pops, block, filet_path, window])

    filet_header = header_fx(pairs, "", filet=True)
    out_file = os.path.join(outdir, f"{ms}.filet")
    if os.path.exists(out_file):
        mode = 'a'
    else:
        mode = 'w'

    with open(out_file, mode) as out:
        out.write(f"##{ms}\n")
        out.write(f"{filet_header}\n")
        if nprocs == 1:
            for args in argslist:
                filet_list = filetStats(args)
                out.write(f"{filet_list}")
        else:
            # number of chunks
            nk = chunk
            # check that there are not more requested than available
            if nprocs > multiprocessing.cpu_count():
                nprocs = multiprocessing.cpu_count()
            # set pool and map
            pool = multiprocessing.Pool(nprocs)
            chunk_list = [argslist[i:i + nk] for i in range(0, len(argslist), nk)]
            if chunkopt:
                chunksize = ceil(len(chunk_list[0])/nprocs)
            else:
                chunksize = 1
            for args in tqdm(chunk_list):
                filet_list = pool.map(filetStats, args, chunksize=chunksize)
                out.write(f"{''.join(filet_list)}")
            pool.close()


def calcStats_sim(ms, outdir, stats, pairs, sum_stats, basepairs, filetpath, nprocs, window):
    """Calculate summary stats from ms.

    Loops over each simulation; for sim, for pop. Will actively write file as
    stats are produced. Multiprocessing is available for all calculations.
    Recommended usage for large number of sims and low memory.

    Parameters
    ----------
    ms : TYPE
        DESCRIPTION.
    out_file : TYPE
        DESCRIPTION.
    stats : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    sum_stats : TYPE
        DESCRIPTION.
    basepairs : TYPE
        DESCRIPTION.
    filetpath : TYPE
        DESCRIPTION.
    nprocs : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # calc stats
    if "sfs" in stats:
        calc_sfs(ms, outdir, pairs, sum_stats, nprocs)
    if "jsfs" in stats:
        calc_jsfs(ms, outdir, pairs, sum_stats, nprocs)
    if "afibs" in stats:
        calc_afibs(ms, outdir, pairs, sum_stats, basepairs, nprocs)
    if "filet" in stats:
        calc_filet(ms, outdir, pairs, sum_stats, basepairs, filetpath, nprocs, window)


def calcStats_pop(stats, pairs, sum_stats, basepairs, filetpath, nprocs, window):
    """Calculate summary stats from ms.

    Loops over each population or pair for each simulation; for pop, for sim.
    Will store all in memory until writing it out at the end. Multiprocessing is
    only available for filet stats calculation. Recommended usage for large pops
    and few sims ... maybe as direct input to something like ABCtoolbox mcmc.

    Parameters
    ----------
    ms : TYPE
        DESCRIPTION.
    stats : TYPE
        DESCRIPTION.
    out_file : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    sum_stats : TYPE
        DESCRIPTION.
    basepairs : TYPE
        DESCRIPTION.
    filetpath : TYPE
        DESCRIPTION.
    nprocs : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.

    Returns
    -------
    asfsdict : TYPE
        DESCRIPTION.
    jsfsdict : TYPE
        DESCRIPTION.
    afibsdict : TYPE
        DESCRIPTION.
    filetdict : TYPE
        DESCRIPTION.

    """
    if "sfs" in stats:
        asfsdict = sum_stats.asfsStats()
        # asfs = " ".join(map(str, [i for t in aSFS for i in t]))
    else:
        asfsdict = {}

    if "jsfs" in stats:
        jsfsdict = sum_stats.jsfsStats(pairs)
    else:
        jsfsdict = {}

    if "afibs" in stats:
        afibsdict = sum_stats.afibs(basepairs)
    else:
        afibsdict = {}

    if "filet" in stats:
        if nprocs == 1:
            filetdict = sum_stats.filetStats(basepairs, filetpath, window)
        else:
            # multi processing by pops
            filet_list = []
            argslist = []
            filetdict = {}

            # check that there are not more requested than available
            if nprocs > multiprocessing.cpu_count():
                nprocs = multiprocessing.cpu_count()
            # set pool and map
            pool = multiprocessing.Pool(nprocs)
            for pop1, pop2 in combinations(sum_stats.pops, 2):
                argslist.append([pop1, pop2, basepairs, filetpath, window])
            filet_list.append(pool.map(sum_stats.filetStatsMP, argslist))
            pool.close()
            # resize and zip
            filet_zip = list(zip(*filet_list[0]))
            for r in range(len(filet_zip)):
                filetdict[r] = filet_zip[r]
    else:
        filetdict = {}

    return asfsdict, jsfsdict, afibsdict, filetdict


def calc_simstats(ms, outdir, msdict, pairs, stats, filetpath, nprocs, window):
    """Calculate summary stats from ms.

    ----------
    msdict : TYPE
        DESCRIPTION.
    pairs : TYPE
        DESCRIPTION.
    stats : TYPE
        DESCRIPTION.
    filetpath : TYPE
        DESCRIPTION.
    nprocs : TYPE
        DESCRIPTION.
    window : TYPE, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    asfsdict : TYPE
        DESCRIPTION.
    jsfsdict : TYPE
        DESCRIPTION.
    afibsdict : TYPE
        DESCRIPTION.
    filetdict : TYPE
        DESCRIPTION.

    """
    infile = os.path.split(ms)[-1]
    popconfig = msdict["pops"]
    basepairs = msdict["basepairs"]
    pos_list = msdict["pos"]
    hap_list = msdict["haps"]

    # format for allel by transposing
    hap_arrT = []
    for hap in hap_list:
        if len(hap) > 1:
            hap_tmp = []
            for h in hap:
                hap_tmp.append(h.transpose())
            hap_arrT.append(hap_tmp)
        else:
            hap_arrT.append(hap[0].transpose())

    # stats class
    sub_pops = set([int(item) for x in [x.split("-") for x in pairs] for item in x])
    pops = [popconfig[i] for i in sub_pops]
    sum_stats = SumStats(hap_arrT, pos_list, pops)

    if "pop" in stats:
        asfs, jsfs, afibs, filet = calcStats_pop(stats,
                                                 pairs,
                                                 sum_stats,
                                                 basepairs,
                                                 filetpath,
                                                 nprocs,
                                                 window)
        write_stats_out(infile,
                        outdir,
                        pairs,
                        sum_stats.pops,
                        asfs,
                        jsfs,
                        afibs,
                        filet)
    else:
        calcStats_sim(infile,
                      outdir,
                      stats,
                      pairs,
                      sum_stats,
                      basepairs,
                      filetpath,
                      nprocs,
                      window)


def calc_obsStats(vcfFile, chr_arm, chrlen, outFile, maskFile, anc_fasta,
                  window, unmskfrac, pairs, sampleFile, stats, filet_path):
    """Calculate Obs stats.

    vcfFile
    pops
    chr_arm
    genome_file
    gff3_file
    meta_file
    stats

    Returns
    -------
    None.

    """
    vcf_path = os.path.abspath(vcfFile)
    outdir = os.path.dirname(vcf_path)
    calls = allel.read_vcf(vcfFile)
    chroms = calls["variants/CHROM"]
    positions = np.extract(chroms == chr_arm, calls["variants/POS"])
    #
    samples = calls["samples"]
    sample_pop = readSampleToPopFile(sampleFile)
    sub_pops = list(dict.fromkeys([item for x in [x.split("-") for x in pairs] for item in x]))
    sample_ix = []
    for pop in sub_pops:
        sample_ix.append([i for i in range(len(samples)) if sample_pop.get(samples[i], "popNotFound!") == pop])
    #
    rawgenos = np.take(calls["calldata/GT"], [i for i in range(len(chroms)) if chroms[i] == chr_arm], axis=0)
    #
    if maskFile:
        unmasked = readMaskDataForScan(maskFile, chr_arm)
        assert len(unmasked) == chrlen
    #
    gt = allel.GenotypeArray(rawgenos)
    snps = [i for i in range(len(positions)) if unmasked[positions[i]-1]]
    gt = allel.GenotypeArray(gt.subset(sel0=snps))
    pos = allel.SortedIndex([positions[i] for i in snps])
    # alleleCounts = allel.AlleleCountsArray([[alleleCounts[i][0], max(alleleCounts[i][1:])] for i in snps])
    # start stats
    if "sfs" in stats:
        # asfs
        header = header_fx(pairs, "", sfs=True)
        out_file = os.path.join(outdir, f"{outFile}.sfs")
        if os.path.exists(out_file):
            mode = 'a'
        else:
            mode = 'w'
        with open(out_file, mode) as out:
            out.write(f"{header}\n")
            sfs_list = asfsObsStats([pos, gt, sample_ix])
            out.write(f"{sfs_list}")

    if "jsfs" in stats:
        header = header_fx(pairs, "", jsfs=True)
        out_file = os.path.join(outdir, f"{outFile}.jsfs")
        if os.path.exists(out_file):
            mode = 'a'
        else:
            mode = 'w'
        with open(out_file, mode) as out:
            out.write(f"{header}\n")
            jsfs_list = jsfsObsStats([pos, gt, sample_ix])
            out.write(f"{jsfs_list}")
    if "afibs" in stats:
        header = header_fx(pairs, sample_ix, afibs=True, obs=True)
        out_file = os.path.join(outdir, f"{outFile}.afibs")
        if os.path.exists(out_file):
            mode = 'a'
        else:
            mode = 'w'
        with open(out_file, mode) as out:
            out.write(f"{header}\n")
            afibs_list = afibsObsStats([pos, gt, sample_ix, window, chrlen])
            out.write(f"{afibs_list}")
    if "filet" in stats:
        if anc_fasta:
            anc_arr = makeAncArray(calls, pos, chr_arm, anc_fasta)
        filet_header = header_fx(pairs, "", filet=True)
        out_file = os.path.join(outdir, f"{outFile}.filet")
        if os.path.exists(out_file):
            mode = 'a'
        else:
            mode = 'w'
        with open(out_file, mode) as out:
            out.write(f"{filet_header}\n")
            filet_list = filetObsStats([pos, gt, sample_ix, chrlen], unmasked,
                                       anc_arr, unmskfrac, window, filet_path)
            out.write(f"{filet_list}")


def parse_args(args_in):
    """Parse args.

    Parameters
    ----------
    args_in : TYPE
        DESCRIPTION.

    Returns
    -------
    argsDict : TYPE
        DESCRIPTION.

    """
    parser = argparse.ArgumentParser(description="calculate summary statistics"
                                     " from simulated data or observed data")
    # parser._positionals.title = f"enter 'python {sys.argv[0]} modeName -h' for modeName's help message"
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('sim', help="Generate stats from sim data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_b = subparsers.add_parser('obs', help="Generate stats from data in a VCF",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_a.set_defaults(mode='sim')
    # parser_a._positionals.title = "required arguments"
    parser_a.add_argument('ms_file', help="path to simulation output file"
                          "must be same format used by Hudson\'s ms")
    parser_a.add_argument('--out_dir', default="./", help="path to file where stats "
                          "will be written")
    parser_a.add_argument("--split", action="store_true",
                          help="split is preprocessing")
    parser_a.add_argument("--pairs", nargs='+', default="all",
                          help="list of pairs separate by hyphen, 0 indexed")
    parser_a.add_argument("--stats", nargs='+', default="filet",
                          choices=["pop", "sfs", "jsfs", "filet", "afibs"],
                          help="which stats to calculate")
    parser_a.add_argument("--nprocs", type=int, default=1, help="try to run"
                          " with parallel. If > 1 will parallel on current machine"
                          " else will run through each file(s) in order")
    parser_a.add_argument('--filet_path', type=str, default="current_dir",
                          help="path to FILET dir w/ programs")
    parser_a.add_argument('--window', type=int, default=0,
                          help="window size for FILET calculations. If not given"
                          " defaults to contig size if smaller than 100kb, else"
                          " windows of 100kb.")

    # calculate stats from a VCF file
    parser_b.set_defaults(mode='obs')
    # parser_b._positionals.title = "required arguments"
    parser_b.add_argument('vcfFileIn', help="VCF format file containing data"
                          "for one chromosome arm (other arms will be ignored)")
    parser_b.add_argument('chr_arm', help="Exact name of the chromosome arm for"
                          "which feature vectors will be calculated")
    parser_b.add_argument('chr_len', type=int, help="Length of the chromosome arm")
    parser_b.add_argument('out_file', help="path to file where feature vectors "
                          "will be written")
    parser_b.add_argument("--pairs", nargs='+',
                          help="list of pops")
    parser_b.add_argument('pops_file', help="individual names to pops as found in"
                          "VCF file")
    parser_b.add_argument("--stats", nargs='+', default="filet",
                          choices=["sfs", "jsfs", "filet", "afibs", "all"],
                          help="which stats to calculate")
    parser_b.add_argument('--win_size', type=int, default=10000,
                          help="Length of the window")
    parser_b.add_argument('--win_slide', type=int, default=0,
                          help="overlap/slide between windows")
    parser_b.add_argument('--mask_file', default=None,
                          help="Path to a fasta-formatted file that contains"
                          "masking information (marked by \'N\'). If specified,"
                          "simulations will be masked in a manner mirroring"
                          "windows drawn from this file.")
    parser_b.add_argument('--unmasked_frac_cutoff', type=float, default=0.25,
                          help="Minimum fraction of unmasked sites, if masking simulated data")
    parser_b.add_argument('--ancestral_fasta', default=None,
                          help="Path to a fasta-formatted file that contains"
                          " inferred ancestral states")
    parser_b.add_argument('--filet_path', type=str, default="current_dir",
                          help="path to FILET dir w/ programs")
    args = parser.parse_args(args_in)
    argsDict = vars(args)

    return argsDict


def main():
    """Run main function."""
    argsDict = parse_args(sys.argv[1:])
    # =========================================================================
    #  Gather args
    # =========================================================================
    if argsDict["mode"] == "sim":
        ms_file = os.path.abspath(argsDict["ms_file"])
        out_dir = os.path.abspath(argsDict["out_dir"])
        split = argsDict["split"]
        pairs = argsDict["pairs"]
        stats = argsDict["stats"]
        processors = argsDict["nprocs"]
        filet_path = os.path.abspath(argsDict["filet_path"])
        window = argsDict["window"]
    else:
        vcf = argsDict["vcfFileIn"]
        chrom = argsDict["chr_arm"]
        chrom_len = argsDict["chr_len"]
        out_file = argsDict["out_file"]
        mask_file = argsDict["mask_file"]
        anc_fasta = argsDict["ancestral_fasta"]
        win_size = argsDict["win_size"]
        win_slide = argsDict["win_slide"]
        unmasked_frac = argsDict["unmasked_frac_cutoff"]
        pairs = argsDict["pairs"]
        pops_file = argsDict["pops_file"]
        stats = argsDict["stats"]
        filet_path = argsDict["filet_path"]
    # =========================================================================
    #  Main executions
    # =========================================================================
    if "-" not in pairs[0]:
        npairs = []
        for p1, p2 in combinations(pairs, 2):
            npairs.append(f"{p1}-{p2}")
    else:
        npairs = pairs
    if argsDict["mode"] == "sim":
        # get path to file/dir
        if os.path.isdir(ms_file):
            # will open many files, suffix with msout
            msfiles = glob.glob("*.msout")
        else:
            msfiles = [ms_file]
        # for each file
        for ms in msfiles:
            print("reading file ...")
            start = timer()
            msdict = ms_parse(ms)
            end = timer()
            print(f"total time to read: {end - start}")
            if split:
                pair_split(msdict, out_dir, npairs)
            else:
                print("starting stats ...")
                start = timer()
                calc_simstats(ms,
                              out_dir,
                              msdict,
                              npairs,
                              stats,
                              filet_path,
                              processors,
                              window)
                end = timer()
                print(f"total time for stats: {end - start}")
    elif argsDict["mode"] == "obs":
        calc_obsStats(vcf, chrom, chrom_len, out_file, mask_file, anc_fasta,
                      [win_size, win_slide], unmasked_frac, npairs, pops_file,
                      stats, filet_path)


if __name__ == "__main__":
    main()
