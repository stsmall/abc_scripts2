# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:05:00 2020
@author: Scott T. Small

Main script for generating statistics for ABC and ML training.
depends: parse_sims.py, sims_stats.py, obs_stats.py

Example
-------

run_stats.py sim tests/msout/test.ms -cfg docs/examples/example.stats.ms.cfg
    --nprocs 1 --ms msmove

run_stats.py obs --infile vcf/h5 --fasta foo.mask.fa --gff foo.gff --pops 4
    --pairs 0-1 0-2 0-3 --stats filet --anc_fasta anc.fasta


Notes
-----

Creates summary stats from coalescent simulations: ms, msmove, discoal, msprime
There are two main modes: sims and obs

 Mode 'sims' is for ms-style formats in any file, those produced by run_sims.py

 Mode 'obs' is for generating the same stats but with a starting vcf.


"""
import sys
import argparse
from pathlib import Path
import numpy as np
import multiprocessing
import glob
from math import ceil
from tqdm import tqdm

from project.sim_modules.readconfig import read_config_stats
from project.stat_modules.write_stats import headers, stats_out
from project.stat_modules.sequtils import read_ms, add_seqerror
from project.stat_modules.sumstats import PopSumStats


def calc_simstats(ms):
    # calc stats
    stat_mat = np.zeros([len(ms), header_len])
    length_bp = stats_dt["length_bp"]
    pfe = stats_dt["perfixder"]
    i = 0
    for pos, haps in zip(*ms):
        pos, haps, counts = add_seqerror(pos, haps, length_bp, pfe, seq_error=True)
        stats_ls = []
        popsumstats = PopSumStats(pos, haps, counts, stats_dt)
        for stat in stats_dt["calc_stats"]:
            stat_fx = getattr(popsumstats, stat)
            try:
                ss = stat_fx()
                # print(f"{stat} =  {len(ss)}")
            except IndexError:
                ss = [np.nan] * len(stats_dt["pw_quants"])
            stats_ls.extend(ss)
        stat_mat[i, :] = stats_ls
        i += 1
    pop_stats_arr = np.nanmean(stat_mat, axis=0)

    return pop_stats_arr


def run_simstats(ms_files, msexe, outpath, nprocs):
    """ """
    global header_len
    global header
    # read in all the files
    length_bp = stats_dt["length_bp"]
    nhaps = stats_dt["num_haps"]
    ms_dict = read_ms(ms_files, msexe, nhaps, length_bp)
    sim_number = len(ms_dict)
    # write headers
    outfile = outpath.parent / f"{outpath.stem}.pop_stats.txt"
    pops_outfile = open(outfile, 'w')
    pops_outfile, header_, header_ls = headers(pops_outfile, stats_dt)
    header_len = header_
    header = header_ls
    if nprocs == 1:
        for ms in tqdm(ms_dict.values()):
            pop_stats_arr = calc_simstats(ms)
            pops_outfile = stats_out(pop_stats_arr, pops_outfile, nprocs)
        pops_outfile.close()
    else:
        # chunk and MP
        nk = nprocs * 10
        ms_vals = list(ms_dict.values())
        chunk_list = [ms_vals[i:i + nk] for i in range(0, len(ms_vals), nk)]
        chunksize = ceil(nk/nprocs)
        pool = multiprocessing.Pool(nprocs)
        for i, args in enumerate(chunk_list):
            pop_stats_arr = pool.map(calc_simstats, args, chunksize=chunksize)
            pops_outfile = stats_out(pop_stats_arr, pops_outfile, nprocs)
            print(i)
        pool.close()
        pops_outfile.close()


def calc_obsStats(vcfpath, chrom, pops, coord_bed, zarrpath, outpath):
    """Calculate stats from a VCF file."""
    # if reuse_zarr is true
    if zarrpath.exists():
        zarfile = zarrpath
    else:
        # save zarr
        zarrfile = zarrpath
        # make zarr
        # load zarr
    # load popdf
    # keep only inds in the popdf
    # separate df for each population ... keep position

    ## record indexes, rejoin as (inds, pos)
    # with open(coords_bed) OR use a step-size:
        # s = 0
        # e = coord_bed
        # s = e
        # e += coord_bed

        # select range, loc_ranges()
        # pos, haps, counts =




    # 1) apply gff_file & filter use locate_range
    #     -subset of filtered gt and pos
    # 2) turn into hap/pos/counts
    # 3) pass to stats in coordinate chunks
    #     using pos > X > pos
    #     check that same coordinate range has less than 25% missing sites from maskbed
    #     ADIVISE doing a SLIDING WINDOW here to better find locations that meet all criteria
    # 4) print stats out to file
    # chrom start stop sites STATS

    return None

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
    parser_a.add_argument("--ms", type=str, required=True,
                          help=" full path to discoal/msbgs/scrm exe")
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
    parser_b.add_argument('--pops_file', help="individual names to pops as found in"
                          "VCF file")
    parser_b.add_argument('-cfg', "--configFile",
                          help="path to config stats file, see examples")
    parser_b.add_argument('--coords_bed', required=True,
                          help="Path to a bed file of coordinates for stats")
    parser_b.add_argument('--zarr_path', type=str
                          help="Path to a zarr file. If exists will reuse if not"
                          " it will build one at that location")
    parser_b.add_argument('-o', "--outfile", default="./",
                          help="path to file where stats will be written")
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
        mspath = Path(argsDict["ms_file"])
        configFile = argsDict["configFile"]
        ms = argsDict["ms"]
        outpath = Path(argsDict["outfile"])
        nprocs = argsDict["nprocs"]

    else:
        vcfpath = Path(argsDict["vcfFileIn"])
        chrom = argsDict["chr_arm"]
        configFile = argsDict["configFile"]
        pops = argsDict["pops_file"]
        coord_bed = argsDict["coords_bed"]
        zarrpath = Path(argsDict["coords_bed"])
        outpath = Path(argsDict["outfile"])
    # =========================================================================
    #  Config parser
    # =========================================================================
    global stats_dt
    stats_dt = read_config_stats(configFile)
    # =========================================================================
    #  Main executions
    # =========================================================================
    if argsDict["mode"] == "sim":
        if mspath.is_dir():
            # will open many files, suffix with msout
            ms_files = list(mspath.glob("*.msout"))
        else:
            ms_files = [mspath]
        run_simstats(ms_files, ms, outpath, nprocs)

    elif argsDict["mode"] == "obs":
        calc_obsStats(vcfpath, chrom, pops, coord_bed, zarrpath, outpath)


if __name__ == "__main__":
    main()
