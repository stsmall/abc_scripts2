#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:32:28 2020
@author: Scott T. Small

"""
import msprime as msp
import numpy as np
import pandas as pd
import multiprocessing

# =========================================================================
# change this to allow trees to be written to file using tree.dump()
# WARNING! this will produce a lot of files! == sim_number
run_stats = False
# =========================================================================


def pop_config():
    """Set pop config for msprime.

    Parameters
    ----------
    model_dt : Dict
        global declared dict of model values
    ploidy : int
        ploidy of samples
    Returns
    -------
    pops_ls : List
        List containing msprime PopConfig objects

    """
    sample_sizes = model_dt["sampleSize"]
    init_sizes = [size * ploidy for size in model_dt["initialSize"]]
    pops_ls = []
    for ss, init in zip(sample_sizes, init_sizes):
        pops_ls.append(msp.PopulationConfiguration(sample_size=ss, initial_size=init))
    return pops_ls


def demo_config(params):
    """Create model with no migration for discoal.

    Parameters
    ----------
    ord_events : TYPE
        DESCRIPTION.
    model_dict : TYPE
        DESCRIPTION.
    ms_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    dem_list : TYPE
        DESCRIPTION.

    """
    demo_param_df = pd.concat([demo_df, pd.DataFrame(params)])
    demo_param_df_srt = demo_param_df.set_index("time")
    demo_param_df_srt.sort_index(inplace=True)
    dem_list = []
    sourcelist = []

    # "time": float, "event": [Ne, ej, tes, tm], "pop": [0-9], "value": float
    for time, row in demo_param_df_srt.iterrows():
        event = row["event"]
        if time > hybrid_switch_over:  # add hybrid switch-over here
            dem_list.append(msp.SimulationModelChange(time=time, model=hybrid_model))
        if "Ne" in event:
            [pop1] = row["pops"]
            pop1 = int(pop1)
            if type(row["value"]) is list:
                if len(row["value"]) > 1:
                    low = row["value"][0]
                    high = row["value"][1]
                    size = np.random.randint(low, high)
                else:
                    size = row["value"][0]
            else:
                size = row["value"]
            dem_list.append(msp.PopulationParametersChange(time=time, initial_size=size, population=pop1, growth_rate=0))
        elif "ej" in event:
            pop1, pop2 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            # pop1 -> pop2
            if pop1 not in sourcelist:
                dem_list.append(msp.MassMigration(time=time, source=pop1, destination=pop2, proportion=1.0))
                dem_list.append(msp.MigrationRateChange(time, 0, (pop1, pop2)))
                sourcelist.append(pop1)
        elif "es" in event:
            # es34
            pop1, pop2 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            if not any(i in sourcelist for i in [pop1, pop2]):
                prop = row["value"]
                dem_list.append(msp.MassMigration(time=time, source=pop2, destination=pop1, proportion=prop))
        elif "ea" in event:
            # ea345: daughter, parent1, parent2
            pop1, pop2, pop3 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            pop3 = int(pop3)
            if not any(i in sourcelist for i in [pop1, pop2, pop3]):
                prop = row["value"]
                dem_list.append(msp.MassMigration(time=time, source=pop2, destination=pop1, proportion=prop))
                dem_list.append(msp.MassMigration(time=time, source=pop3, destination=pop1, proportion=prop))
        elif "m" in event:
            pop1, pop2 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            mig = row["value"]
            if not any(i in sourcelist for i in [pop1, pop2]):
                if "ms" in event:  # set as symmetrical
                    dem_list.append(msp.MigrationRateChange(time=time, rate=mig, matrix_index=(pop1, pop2)))
                    dem_list.append(msp.MigrationRateChange(time=time, rate=mig, matrix_index=(pop2, pop1)))
                elif "ma" in event:
                    dem_list.append(msp.MigrationRateChange(time=time, rate=mig, matrix_index=(pop1, pop2)))
    return dem_list


def run_simulation(param_df):
    """Run msprime simulation.

    Parameters
    ----------
    param_df : TYPE
        DESCRIPTION.
    check_demo : TYPE, optional
        DESCRIPTION. The default is True.
    run_stats : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    ts : TYPE
        DESCRIPTION.

    """
    # rescale
    pops = pop_config()

    # build demographic command line
    demo_events = demo_config(param_df)
    # check demo
    if dry_run:
        checkDemo(pops, demo_events)
    else:
        tree = msp.simulate(
                          Ne=np.random.choice(scaled_Ne),
                          recombination_rate=np.random.choice(rec),
                          mutation_rate=np.random.choice(mu),
                          num_replicates=model_dt["loci"],
                          length=model_dt["contig_length"],
                          population_configurations=pops,
                          migration_matrix=model_dt["migmat"],
                          demographic_events=demo_events,
                          model=initial_model)
        if run_stats:
            pass
            # pipe directly to run_stats.py sims --config FOO --file None --format msprime
            # this will run as MP with the sims, should preload the config as dict or a global
            # so I dont have to keep passing it. Then need a different command line pass-by
            # to prevent it looking for all the options
        else:
            for i, ts in enumerate(tree):
                ts.dump(f"{outfile_tree}.{i}.tree")


def checkDemo(pops, demo_events):
    """Print out demography for debugging."""
    dd = msp.DemographyDebugger(
                      Ne=np.random.choice(scaled_Ne),
                      population_configurations=pops,
                      migration_matrix=model_dt["migmat"],
                      demographic_events=demo_events,
                      model=initial_model)
    dd.print_history()


def simulate_msprime(model_dict, demo_dataframe, param_df, sim_number, outfile, nprocs, dryrun):
    """Main code for simulating msprime

    Parameters
    ----------
    model_dict : Dict
        Dict holding information on model specs from config file
    demo_dataframe : DataFrame
        Dataframe with info from model file
    param_df : DataFrame
        Dataframe that holds tbi values and draws
    sim_path : str
        file path
    sim_number : int
        how man independent sims to run
    outfile : str
        file name for output
    nprocs : int
        how many processors to run with MP

    Returns
    -------
    Writes a trees file from mpsrime to a file

    """
    # =========================================================================
    #  Globals
    # =========================================================================
    global demo_df
    global model_dt
    global nhaps
    global mu
    global rec
    global ploidy
    global scaled_Ne
    global initial_model
    global hybrid_model
    global hybrid_switch_over
    global outfile_tree
    global dry_run
    # =========================================================================
    # declare globals
    dry_run = dryrun
    initial_model = "hudson"
    hybrid_model = "hudson"  # dtwf, smc, smc_prime
    hybrid_switch_over = 500
    outfile_tree = outfile
    model_dt = model_dict
    demo_df = demo_dataframe
    nhaps = sum(model_dt["sampleSize"])
    # set mutation rate
    mut_rate = model_dt["mutation_rate"]
    if type(mut_rate) == list:
        if len(mut_rate) == 2:
            low, high = mut_rate
            mu = np.random.uniform(low, high, sim_number)
    else:
        mu = [mut_rate]
    # set recombination rate
    rec_rate = model_dt["recombination_rate"]
    if type(rec_rate) == list:
        if len(rec_rate) == 2:
            low, high = mut_rate
            rec = np.random.uniform(low, high, sim_number)
    else:
        rec = [rec_rate]
    # set Ne
    ploidy = model_dt["ploidy"]
    effective_size = model_dt["eff_size"]
    if type(effective_size) == list:
        low, high = effective_size
        scaled_Ne = np.random.randint(low, high, sim_number) * ploidy
    else:
        scaled_Ne = effective_size * ploidy

    # set up generator fx
    event = param_df["event"].values
    pops = param_df["pops"].values
    time_arr = list(zip(*param_df["time"].values))
    value_arr = list(zip(*param_df["value"].values))
    param_gen = ({"time": time_arr[i], "event":event, "pops":pops, "value": value_arr[i]} for i in range(sim_number))

    # check nprocs
    if nprocs > multiprocessing.cpu_count():  # check that there are not more requested than available
        print("not {nprocs} processors available, setting to {multiprocessing.cpu_count()}")
        nprocs = multiprocessing.cpu_count()
    if dry_run:
        for param in param_gen:
            run_simulation(param)
            break
    else:
        with multiprocessing.Pool(processes=nprocs) as pool:
            pool.map(run_simulation, param_gen)
