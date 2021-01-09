#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:31:12 2020.
@author: Scott T. Small.

Main script for generating simulations for ABC and ML training.
depends: sim_models.py, sim_params.py

Example
-------

abc_sims.py -cfg examples/example.cfg -i 100000 --ms msmove --out test

generates a file with a random name that has 5,000 lines. each line is a call
to msmove under the CONFIG specifications.

Notes
-----
This relies on a config file and a model file. It can also use a distribution
of pi and rho (the population recombination rate). See the github for examples.


"""
import sys
import argparse
import os
import pandas as pd
from project.sim_modules.readconfig import read_config_sims
from project.sim_modules.models import parse_model
from project.sim_modules.sim_discoal import simulate_discoal
from project.sim_modules.sim_msbgs import simulate_msbgs
from project.sim_modules.sim_msprime import simulate_msprime
from project.sim_modules.sim_scrm import simulate_scrm


def write_params(param_df, outfile, sim_number, dryrun):
    """Write priors to file.

    Parameters
    ----------
    eventkey_dict : TYPE
        DESCRIPTION.
    params_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    priors_str : TYPE
        DESCRIPTION.

    """
    headers = []
    for ep in param_df.itertuples():
        i = 0
        base = f"{ep.event}_{''.join(ep.pops)}"
        col = f"{base}_{i}"
        while col in headers:
            i += 1
            col = f"{base}_{i}"
        headers.append(f"{col}")
    iterables = [headers, ["time", "value"]]
    m_ix = pd.MultiIndex.from_product(iterables, names=['tbi', 'params'])
    param_ls = []
    for tbi in param_df.itertuples():
        param_ls.append(tbi.time)
        param_ls.append(tbi.value)
    df = pd.DataFrame(list(zip(*param_ls)), index=range(sim_number), columns=m_ix)

    if dryrun:
        print(f"{df.describe()}\n\n")
        import seaborn as sns
        df_col = df.columns
        for c in df_col:
            if len(df[c[0]][c[1]].unique()) > 1:
                sns_plot = sns.displot(data=df[c[0]], x=c[1])
                sns_plot.savefig(f"{outfile}.{c[0]}_{c[1]}.pdf")
    else:
        df.to_csv(f"{outfile}.priors.out")


def parse_args(args_in):
    """Parse args."""
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cfg', "--configFile", required=True,
                        help="path to config file")
    parser.add_argument('-m', "--modelFile", required=True,
                        help="path to model file")
    parser.add_argument('-i', "--iterations", type=int, default=1,
                        help="number of iterations, number of lines. If you want"
                        " replications alter loci in config file")
    parser.add_argument("--ms", type=str, default="msprime",
                        help=" full path to discoal/msbgs/scrm exe")
    parser.add_argument("--approx", type=str, default='',
                        help="approx to coal w/ recomb for scrm")
    parser.add_argument("--out", type=str, default="model",
                        help="outfilename to write simulations")
    parser.add_argument("--ploidy", type=float, default=1,
                        help="options: hap=0.5, sex=0.75, auto=1")
    parser.add_argument("--set_priors", type=str,
                        help="provide a list of priors to be reused will overide"
                        " all other parameters")
    parser.add_argument("--nprocs", type=int, default=1,
                        help="processors for MP")
    parser.add_argument("--dryrun", action="store_true",
                        help="run debugger and plot priors")
    parser.add_argument("--stats_config", type=str, default=None,
                        help="config file for calc stats on the fly")
    return(parser.parse_args(args_in))


def main():
    """Execute main."""
    args = parse_args(sys.argv[1:])
    # =========================================================================
    #  Gather args
    # =========================================================================
    configFile = args.configFile
    model_file = args.modelFile
    sim_number = args.iterations
    ms_path = args.ms
    outfile = args.out
    ploidy = args.ploidy
    priors_df = args.set_priors
    nprocs = args.nprocs
    dry_run = args.dryrun
    stats_config = args.stats_config

    model_dir = os.path.abspath(model_file)
    out_path = os.path.split(model_dir)[0]
    if not outfile:
        outfile = os.path.split(model_dir)[1]
    sim_path = os.path.join(out_path, f"{outfile}.{sim_number}")
    # =========================================================================
    #  Config parser
    # =========================================================================
    model_dt = read_config_sims(configFile, ms_path)
    model_dt["ploidy"] = ploidy
    # =========================================================================
    #  Main executions
    # =========================================================================
    # parses the model file and draw all parameters
    demo_df, param_df = parse_model(model_file, sim_number)
    if priors_df:
        assert os.path.exists(priors_df), "no priors file found"
        param_df = pd.read_csv(priors_df, header=[0, 1], index_col=0, skipinitialspace=True)
    else:
        # write priors
        priors_outfile = f"{sim_path}"
        write_params(param_df, priors_outfile, sim_number, dry_run)

    if "discoal" in ms_path:
        simulate_discoal(ms_path, model_dt, demo_df, param_df, sim_number, sim_path, nprocs, stats_config, dry_run)
    elif "msbgs" in ms_path:
        simulate_msbgs(model_dt, demo_df, param_df, ms_path, sim_path, sim_number, outfile)
    elif "scrm" in ms_path:
        simulate_scrm(ms_path, model_dt, demo_df, param_df, sim_number, sim_path, nprocs, stats_config, dry_run, args.approx)
    elif "msprime" in ms_path:
        simulate_msprime(model_dt, demo_df, param_df, sim_number, sim_path, nprocs, stats_config, dry_run)


if __name__ == "__main__":
    main()
