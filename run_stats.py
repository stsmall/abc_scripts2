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
import os
import numpy as np

from project.sim_modules.readconfig import read_config_stats
from project.stat_modules.write_stats import headers, stats_out
from project.stat_modules.sequtils import read_ms
from project.stat_modules.sumstats import PopSumStats

import multiprocessing
import glob
from math import ceil
from tqdm import tqdm
# set globals
chunk = 100
chunkopt = True


def calc_simstats(ms_file):
    infile = os.path.split(ms_file)[-1]
    length_bp = stats_dt["length"]
    pos, haps, counts = read_ms(infile, length_bp, seq_error=True)
    # calc true stats
    stats_list = []
    for stat in stats_dt["calc_stats"]:
        stat_fx = getattr(PopSumStats, stat)
        stats_list.append(stat_fx(pos, haps, counts, stats_dt))

    return stats_list


def run_simstats(ms_files, out_dir, nprocs):
    """ """
    # write headers
    pops_outfile = open(f"{out_dir}.pop_stats.txt", 'w')
    pops_outfile = headers(pops_outfile, stats_dt)
    # chunk and MP
    nk = 10
    chunk_list = [ms_files[i:i + nk] for i in range(0, len(ms_files), nk)]
    chunksize = ceil(len(chunk_list[0])/nprocs)
    pool = multiprocessing.Pool(nprocs)
    for args in tqdm(chunk_list):
        pop_stats_arr = pool.map(calc_simstats, args, chunksize=chunksize)
        pops_out = stats_out(pop_stats_arr, pops_outfile)
    pool.close()
    pops_out.close()


# def calc_obsStats(vcfFile, chr_arm, chrlen, outFile, maskFile, anc_fasta,
#                   window, unmskfrac, pairs, sampleFile, stats, filet_path):
#     """Calculate Obs stats.

#     vcfFile
#     pops
#     chr_arm
#     genome_file
#     gff3_file
#     meta_file
#     stats

#     Returns
#     -------
#     None.

#     """
#     vcf_path = os.path.abspath(vcfFile)
#     outdir = os.path.dirname(vcf_path)
#     calls = allel.read_vcf(vcfFile)
#     chroms = calls["variants/CHROM"]
#     positions = np.extract(chroms == chr_arm, calls["variants/POS"])
#     #
#     samples = calls["samples"]
#     sample_pop = readSampleToPopFile(sampleFile)
#     sub_pops = list(dict.fromkeys([item for x in [x.split("-") for x in pairs] for item in x]))
#     sample_ix = []
#     for pop in sub_pops:
#         sample_ix.append([i for i in range(len(samples)) if sample_pop.get(samples[i], "popNotFound!") == pop])
#     #
#     rawgenos = np.take(calls["calldata/GT"], [i for i in range(len(chroms)) if chroms[i] == chr_arm], axis=0)
#     #
#     if maskFile:
#         unmasked = readMaskDataForScan(maskFile, chr_arm)
#         assert len(unmasked) == chrlen
#     #


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
    parser_a.add_argument('-cfg', "--configFile",
                          help="path to config file, see examples")
    parser_a.add_argument('-o', "--outfile", default="./",
                          help="path to file where stats will be written")
    parser_a.add_argument("--nprocs", type=int, default=1,
                          help="processors for MP")

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
    parser_b.add_argument('pops_file', help="individual names to pops as found in"
                          "VCF file")
    parser_b.add_argument('-cfg', "--configFile",
                          help="path to config stats file, see examples")
    parser_b.add_argument('--coords_bed', default=None,
                          help="Path to a bed file of coordinates for stats")
    parser_b.add_argument('--mask_bed', default=None,
                          help="Path to a bed file of masked data sites")
    parser_b.add_argument('--masked_frac', default=0.25,
                          help="cut-off for skipping window based on proportion of"
                          " masked sites")
    parser_b.add_argument('--gff', default=None,
                          help="Path to a gff file, for selecting coding/noncoding regions")
    parser_b.add_argument('--gff_filter', default=None, nargs='+',
                          help="remove sites matching this keyword")
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
        configFile = argsDict["configFile"]
        out_dir = os.path.abspath(argsDict["outfile"])
        nprocs = argsDict["nprocs"]

    else:
        vcf = argsDict["vcfFileIn"]
        chrom = argsDict["chr_arm"]
        chrom_len = argsDict["chr_len"]
        out_file = argsDict["out_file"]
        configFile = argsDict["configFile"]
        pops_file = argsDict["pops_file"]
        mask_bed = argsDict["mask_bed"]
        gff_file = argsDict["gff"]
        masked_frac = argsDict["masked_frac"]
        gff_filter = argsDict["gff_filter"]
    # =========================================================================
    #  Config parser
    # =========================================================================
    global stats_dt
    stats_dt = read_config_stats(configFile)
    # =========================================================================
    #  Main executions
    # =========================================================================
    if argsDict["mode"] == "sim":
        # get path to file/dir
        if os.path.isdir(ms_file):
            # will open many files, suffix with msout
            ms_files = glob.glob("*.msout")
        else:
            ms_files = [ms_file]
        run_simstats(ms_files, out_dir, nprocs)

    elif argsDict["mode"] == "obs":
        pass
        #calc_obsStats(vcf, chrom, chrom_len, out_file, pops_file, unmasked_frac=0.25)


if __name__ == "__main__":
    main()
