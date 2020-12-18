#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:32:28 2020
@author: Scott T. Small

"""
import msprime as msp
import tskit as tsk
import numpy as np
import pandas as pd
import tqdm


def sim_syntax(model_dict):
    """Create parameters for specific model.

    Parameters
    ----------
    model_dict : TYPE
        DESCRIPTION.
    ms_path : str
        path to simulator
    Returns
    -------
    ms_dict : Dict
        BOO

    """
    ms_dict = {}
    # populations
    sample_sizes = model_dict["sampleSize"]
    npops = len(sample_sizes)
    ploidy = model_dict["ploidy"]
    locus_len = model_dict["contig_length"]
    effective_size = model_dict["eff_size"] * ploidy

    # calc theta
    mut_rate = model_dict["mutation_rate"]
    if type(mut_rate) == list:
        if len(mut_rate) == 2:
            low, high = mut_rate
            mu = np.random.uniform(low, high)
        else:
            mu = np.random.choice(mut_rate)
    else:
        mu = mut_rate
    theta_loc = 4 * effective_size * mu * locus_len

    # calc rho rate
    rec_rate = model_dict["recombination_rate"]
    if type(rec_rate) == list:
        if len(rec_rate) == 2:
            low, high = mut_rate
            rho = np.random.uniform(low, high)
        else:
            rho = np.random.choice(rec_rate)
    elif rec_rate is None:
        rho = 0
    else:
        rho = rec_rate
    rho_loc = 4 * effective_size * rho * locus_len

    # gene conversion
    gen_conversion = model_dict["gene_conversion"][0]
    if gen_conversion > 0:
        tract = model_dict["gene_conversion"][1]
        gen_cov = f"-gr {gen_conversion / rec_rate} {tract}"
    else:
        gen_cov = ""

    # subops
    init_sizes = [size * ploidy for size in model_dict["initialSize"]]
    # TODO: growth rate
    grow_rate = model_dict["growthRate"]
    mig_mat = model_dict["migMat"]
    subpops = f"-p {npops} {' '.join(map(str, sample_sizes))}"
    ne_sub_pops = [f"-en 0 {i} {pop_ne/effective_size}" for i, pop_ne in enumerate(init_sizes)]
    ne_subpop = " ".join(ne_sub_pops)
    grow_subpop = []
    if mig_mat:
        mig = []
        mig_matrix = zip(*mig_mat)
        for p, pop_m in enumerate(mig_matrix):
            for i, m in pop_m:
                if p != i and m > 0:
                    mig.append(f"-m {p} {i} {4*effective_size*m}")
    else:
        mig_matrix = ""

    ms_dict = {"npops": npops,
               "subpop": subpops,
               "theta_loc": theta_loc,
               "scaled_Ne": effective_size,
               "rho_loc": rho_loc,
               "gen_cov": gen_cov,
               "ne_subpop": ne_subpop,
               "grow_subpop": grow_subpop,
               "mig_matrix": mig_matrix}

    return ms_dict


def model_msprime(model_dict, ms_dict, demo_df):
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
    scaled_Ne = ms_dict["scaled_Ne"]
    init_size = list(model_dict["initialSize"])
    dem_list = []
    sourcelist = []

    demo_df_srt = demo_df.set_index("time")
    demo_df_srt.sort_index(inplace=True)
    # "time": float, "event": [Ne, ej, tes, tm], "pop": [0-9], "value": float
    for time, row in demo_df_srt.iterrows():
        new_time = time / (4*scaled_Ne)  # 4N0gens
        event = row["event"]
        if "Ne" in event:
            pop = int(row["pop"])
            size = row["value"][0]
            grow = row["value"][1]
            init_size[pop] = size
            if pop not in sourcelist:
                new_Ne = size / scaled_Ne
                if grow > 0:
                    dem_list.append(f"-en {new_time} {pop} {new_Ne} {grow}")
                else:
                    dem_list.append(f"-en {new_time} {pop} {new_Ne}")
        elif "ej" in event:
            pop1, pop2 = row["pop"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            # pop1 -> pop2
            if pop1 not in sourcelist:
                dem_list.append(f"-ed {new_time} {pop1} {pop2}")
                sourcelist.append(pop1)
        elif "es" in event:
            # es_343; daughter, parent1, parent2. can be same
            pop1, pop2, pop3 = row["pop"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            pop3 = int(pop3)
            if not any(i in sourcelist for i in [pop1, pop2, pop3]):
                prop = 1 - row["value"]
                dem_list.append(f"-ea {new_time} {pop1} {pop2} {pop3} {prop}")
        elif "m" in event:
            pop1, pop2 = row["pop"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            mig1 = row["value"]*4*scaled_Ne
            mig2 = row["value"]*4*scaled_Ne
            if not any(i in sourcelist for i in [pop1, pop2]):
                if "ms" in event:  # set as symmetrical
                    dem_list.append(f"-m {new_time} {pop1} {pop2} {mig1}")
                    dem_list.append(f"-m {new_time} {pop2} {pop1} {mig2}")
                elif "ma" in event:
                    dem_list.append(f"-m {new_time} {pop1} {pop2} {mig1}")

    return dem_list


def command_line(model_dict, demo_df, ms_path, seed):
    """Create a single instance of a call to simulator.

    Parameters
    ----------
    model_dict: Dict
        contains info from config file
    demo_dict: Dict
        dict of pop sizes and times
    par_dict: Dict
        generator of distributions
    event_dict: List
        list of demographic events, column 2 in model
    ms_path: str
        location of ms exe

    Returns
    -------
    mscmd: str
        full call to ms/msmove
    params: list
        list of random parameters for that simulation

    """
    # TODO: set seed to iteration to run in parallel
    seed_ls = [seed, np.random.randint(1, 2**32-1, 1), np.random.randint(1, 2**32-1, 1)]
    # rescale
    ms_dict = sim_syntax(model_dict)

    # build demographic command line
    dem_events = model_msprime(model_dict, ms_dict, demo_df)

    # gather command line args
    ms_params = {
                'ms': ms_path,
                'nhaps': sum(model_dict["sampleSize"]),
                'loci': model_dict["loci"],
                'theta': ms_dict["theta_loc"],
                'rho': ms_dict['rho_loc'],
                'gen_cov': ms_dict['gen_cov'],
                'basepairs': model_dict["contig_length"],
                'subpops': ms_dict["subpop"],
                'ne_subpop': ms_dict["ne_subpop"],
                'growth_subpop': ms_dict["grow_subpop"],
                'migmat': ms_dict["mig_matrix"],
                'demo': " ".join(dem_events)
                }

    ms_base = ("{ms} {nhaps} {loci} {basepairs} -t {theta} -r {rho} "
               "{gen_cov} {subpops} {ne_subpop} {demo} {sel}")
    mscmd = ms_base.format(**ms_params)
    ms_cmd = " ".join(mscmd.split())
    return ms_cmd


def simulate_msprime(model_dict, demo_df, param_df, sim_path, sim_number, outfile):
    """Main simulate code for discoal.

    Parameters
    ----------
    model_dict : TYPE
        DESCRIPTION.
    demo_df : TYPE
        DESCRIPTION.
    param_df : TYPE
        DESCRIPTION.
    ms_path : TYPE
        DESCRIPTION.
    sim_path : TYPE
        DESCRIPTION.
    sim_number : TYPE
        DESCRIPTION.
    outfile : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    event = param_df.event.values
    pop = param_df.pop.values
    time_arr = param_df.time.values
    value_arr = param_df.value.values
    param_gen = ({"time": time_arr[i], "event":event, "pop":pop, "value": value_arr[i]} for i in sim_number)
    pbar = tqdm.tqdm(total=sim_number, desc="Sim Number", unit='sim')
    with open(sim_path, 'w') as sims_outfile:
        # use MP pool here
        for params in param_gen:
            pbar.update(1)
            demo_df = pd.concat([demo_df, pd.DataFrame(params)])
            ts = command_line(model_dict, demo_df)
            sims_outfile.write(ts)
    pbar.close()
