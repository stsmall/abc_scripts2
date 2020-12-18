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

import logging
from datetime import datetime
# =========================================================================
#  Globals
# =========================================================================
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y.%H_%M_%S")
logout = f"{dt_string}.log"
logging.basicConfig(filename=logout, filemode='w', encoding='utf-8', level=logging.DEBUG)
# =========================================================================


def write_priors(param_df, outfile):
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
    if eventkey_dict:
        header_list = []
        for event in eventkey_dict.keys():
            for i, eve in enumerate(eventkey_dict[event]):
                if "tm" in event:
                    pop1 = event[-2]
                    pop2 = event[-1]
                    header_list.append(f"time_ev_{pop1}-{pop2}\tev_prop")
                elif "es" in event:
                    pop1 = event[-3]
                    pop2 = event[-2]
                    pop3 = event[-1]
                    header_list.append(f"time_es_{pop1}-{pop2}-{pop3}\tes_prop")
                elif "ej" in event:
                    pop1 = event[-2]
                    pop2 = event[-1]
                    header_list.append(f"time_ej_{pop1}-{pop2}")
                elif "Ne" in event:
                    pop1 = event[-1]
                    header_list.append(f"Ne_time_{pop1}_{i}\tNe_size_{pop1}_{i}\tNe_grow_{pop1}_{i}")
        return("\t".join(header_list))
    else:
        params_list = list(params_dict.keys())
        params_list.sort(key=lambda x: int(x[3:]))
        par_list = []
        for event in params_list:
            for params in params_dict[event]:
                if type(params) is list:
                    for param in params:
                        par_list.append(param)
                else:
                    par_list.append(params)
            priors_list = map(str, par_list)
        priors_str = "\t".join(priors_list)
        return priors_str


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
    initial_sizes = list(map(int, config.get(init, "initial_sizes").split(",")))
    gene_conversion = list(map(float, config.get(init, "gene_conversion").split(",")))
    mig_file = config.get(init, "migration_matrix")
    if mig_file:
        migration_matrix = np.genfromtxt(mig_file, delimiter=",")
        if np.sum(migration_matrix) > 0:
            assert len(sample_sizes) == migration_matrix.shape[0], "require an entry for each population in mig matrix"
            mig_list = migration_matrix.tolist()
            migration_matrix = [val for sublist in mig_list for val in sublist]
    else:
        migration_matrix = ''

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
                  "migMat": migration_matrix,
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
        priors_outfile = f"{sim_path}.priors"
        write_priors(param_df, priors_outfile)
        param_df.to_csv(f"{outfile}.param_df.csv")

    if "discoal" in ms_path:
        simulate_discoal(model_dict, demo_df, param_df, ms_path, sim_path, sim_number, outfile)
    elif "msbgs" in ms_path:
        simulate_msbgs(model_dict, demo_df, param_df, ms_path, sim_path, sim_number, outfile)
    elif "msprime" in ms_path:
        simulate_msprime(model_dict, demo_df, param_df, sim_path, sim_number, outfile)


if __name__ == "__main__":
    main()
