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
import numpy as np
import configparser
import pandas as pd

from parse_models import parse_model
from sim_discoal import simulate_discoal
from sim_msbgs import simulate_msbgs
from sim_msprime import simulate_msprime


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
    headers = [f"{ep.event}_{''.join(ep.pops)}" for ep in param_df.itertuples()]
    iterables = [headers, ["time", "value"]]
    m_ix = pd.MultiIndex.from_product(iterables, names=['tbi', 'params'])
    param_ls = []
    for tbi in param_df.itertuples():
        param_ls.append(tbi.time)
        param_ls.append(tbi.value)
    df = pd.DataFrame(list(zip(*param_ls)), index=range(sim_number), columns=m_ix)
    # to access multi_index: e.g., df.loc[0]["Ne_1"]["value"]
    df.to_csv(f"{outfile}.priors.out")

    if dryrun:
        pass
        #plot using seaborn as sns


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
                        help=" full path to discoal/msbgs exe")
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

    model_dir = os.path.abspath(model_file)
    out_path = os.path.split(model_dir)[0]
    if not outfile:
        outfile = os.path.split(model_dir)[1]
    sim_path = os.path.join(out_path, f"{outfile}.{sim_number}.sims.out")
    # =========================================================================
    #  Config parser
    # =========================================================================
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(configFile)
    config_path = os.path.split(os.path.abspath(configFile))[0]

    # simulation section
    sim = "simulation"
    contig_len = config.getint(sim, "contiglen")
    num_loci = config.getint(sim, "loci")

    effective_size = config.get(sim, "effective_population_size")
    if "," in effective_size:
        effective_size = list(map(int, effective_size.split(",")))
    else:
        effective_size = int(effective_size)

    recomb_rate = config.get(sim, "recombination_rate")
    if recomb_rate:
        if "," in recomb_rate:
            recomb_rate = list(map(float, recomb_rate.split(",")))
        elif recomb_rate[0].isalpha():
            if os.path.exists(recomb_rate):
                print(f"loading {recomb_rate} ...")
                recomb_rate = np.loadtxt(recomb_rate)
            else:
                print(f"loading {os.path.join(config_path, recomb_rate)} ...")
                recomb_rate = np.loadtxt(os.path.join(config_path, recomb_rate))
        elif recomb_rate[0].isdigit():
            recomb_rate = float(recomb_rate)
        else:
            print("recombination not given, setting to 0")
    mutation_rate = config.get(sim, "mutation_rate")
    if mutation_rate:
        if "," in mutation_rate:
            mutation_rate = list(map(float, mutation_rate.split(",")))
        elif mutation_rate[0].isalpha():
            if os.path.exists(mutation_rate):
                print(f"loading {mutation_rate} ...")
                mutation_rate = np.loadtxt(mutation_rate)
            else:
                print(f"loading {os.path.join(config_path, mutation_rate)} ...")
                mutation_rate = np.loadtxt(os.path.join(config_path, mutation_rate))
        elif mutation_rate[0].isdigit():
            mutation_rate = float(mutation_rate)
        else:
            raise ValueError("must provide mutation rate")

    # initialize section
    init = "initialize"
    sample_sizes = list(map(int, config.get(init, "sample_sizes").split(",")))
    npops = len(sample_sizes)
    initial_sizes = list(map(int, config.get(init, "initial_sizes").split(",")))
    gene_conversion = list(map(float, config.get(init, "gene_conversion").split(",")))
    mig_file = config.get(init, "migration_matrix")
    if mig_file:
        migration_matrix = np.genfromtxt(mig_file, delimiter=",")
        assert len(sample_sizes) == migration_matrix.shape[0], "require an entry for each population in mig matrix"
        mig_list = migration_matrix.tolist()
        migration_matrix = [val for sublist in mig_list for val in sublist]
    else:
        migration_matrix = np.zeros([npops, npops])
    # selection section
    sel_dict = {}
    if config.has_section("positive_selection"):
        assert "discoal" in ms_path, "discoal needed for positive selection"
        sel = "positive_selection"
        hide = config.getboolean(sel, "hide")
        alpha = config.get(sel, "sel_coeff")
        freq = config.get(sel, "soft_freq")
        sweep_stop = config.get(sel, "sweep_time")
        sweep_site = config.get(sel, "sweep_site")
        part_freq = config.get(sel, "partial_sweep")
        adapt = config.get(sel, "adapt_mutrate")
        left_rho = config.get(sel, "leftRho")
        rrh_left = config.get(sel, "Lrecurrent")
        rrh_loc = config.get(sel, "Rrecurrent")
        pop0_Ne = config.get(sel, "pop1_effective_size")
        sweep_Ne = config.get(sel, "sweep_effective_size")

        sel_dict = {"alpha": alpha,
                    "freq": freq,
                    "sweep_stop": sweep_stop,
                    "sweep_site": sweep_site,
                    "part_freq": part_freq,
                    "adapt": adapt,
                    "left_rho": left_rho,
                    "rrh_left": rrh_left,
                    "rrh_loc": rrh_loc,
                    "pop0_Ne": pop0_Ne,
                    "sweep_Ne": sweep_Ne
                    }

        for key in sel_dict.keys():
            if "," in sel_dict[key]:
                sel_dict[key] = list(map(float, sel_dict[key].split(",")))
            else:
                if "." in sel_dict[key]:
                    sel_dict[key] = float(sel_dict[key])
                else:
                    sel_dict[key] = int(sel_dict[key])
        sel_dict["hide"] = hide

    elif config.has_section("background_selection"):
        assert "msbgs" in ms_path, "bgsms needed for background selection"
        sel = "background_selection"
        # Nr /μ is not small (>10, say) and Ne s is > 1
        # sel_def = "10,20"
        # region_def = "[0,1,2,3,4];[5,6,7,8,9]"
        # sel_def = "10"
        # region_def = "[(0_3_500),(1_3_500)]"
        # sel_def = "0"
        # region_def = "neutral"
        selection = config.get(sel, "sel_def")
        region = config.get(sel, "region_def")
        sel_dict = {"sel_def": selection,
                    "region_def": region}

    # =========================================================================
    #  Build model dictionary
    # =========================================================================
    model_dict = {"contig_length": contig_len,
                  "eff_size": effective_size,
                  "recombination_rate": recomb_rate,
                  "mutation_rate": mutation_rate,
                  "sampleSize": sample_sizes,
                  "initialSize": initial_sizes,
                  "gene_conversion": gene_conversion,
                  "migmat": migration_matrix,
                  "loci": num_loci,
                  "ploidy": ploidy,
                  "sel_dict": sel_dict
                  }
    # =========================================================================
    #  Main executions
    # =========================================================================
    # parses the model file and draw all parameters
    demo_df, param_df = parse_model(model_file, sim_number)
    if priors_df:
        if os.path.exists(priors_df):
            param_df = pd.read_csv(priors_df)
    else:
        # write priors
        priors_outfile = f"{sim_path}"
        write_params(param_df, priors_outfile, sim_number, dry_run)

    if "discoal" in ms_path:
        simulate_discoal(model_dict, demo_df, param_df, ms_path, sim_path, sim_number, outfile)
    elif "msbgs" in ms_path:
        simulate_msbgs(model_dict, demo_df, param_df, ms_path, sim_path, sim_number, outfile)
    elif "msprime" in ms_path:
        simulate_msprime(model_dict, demo_df, param_df, sim_number, f"{sim_path}{outfile}", nprocs, dry_run)


if __name__ == "__main__":
    main()
