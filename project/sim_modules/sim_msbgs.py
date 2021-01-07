#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:32:28 2020
@author: Scott T. Small

"""
import numpy as np
import pandas as pd
from tqdm import trange


def selection_parse(model_dict, ms_dict):
    """Parse selection dict for discoal.

    Parameters
    ----------
    model_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sel_ls = []
    scaled_Ne = ms_dict["scaled_Ne"]
    rho_loc = ms_dict["rho_loc"],
    # sel params
    sel_dt = model_dict["sel_dict"]
    sel_def = sel_dt["sel_def"]
    region_def = sel_dt["region_def"]

    return sel_ls


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

    # get Ne
    effective_size = model_dict["eff_size"]
    if type(effective_size) == list:
        low, high = effective_size
        scaled_Ne = np.random.randint(low, high)
    else:
        scaled_Ne = effective_size
    # TODO: msbgs expects the haploid effective size, do I need to rescale to conform with other sims?
    scaled_Ne = scaled_Ne * ploidy

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
    theta_loc = 4 * scaled_Ne * mu * locus_len

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
    rho_loc = 4 * scaled_Ne * rho * locus_len

    gen_conversion = model_dict["gene_conversion"][0]
    if gen_conversion > 0:
        tract = model_dict["gene_conversion"][1]
        gen_cov = f"-gr {gen_conversion / rec_rate} {tract}"
    else:
        gen_cov = ""

    # subops
    init_sizes = [size * ploidy for size in model_dict["initialSize"]]
    mig_mat = model_dict["migMat"]
    subpops = f"-I {npops} {' '.join(map(str, sample_sizes))}"
    ne_sub_pops = [f"-n 0 {i} {pop_ne/scaled_Ne}" for i, pop_ne in enumerate(init_sizes)]
    ne_subpop = " ".join(ne_sub_pops)

    if mig_mat:
        mig = []
        mig_matrix = zip(*mig_mat)
        for p, pop_m in enumerate(mig_matrix):
            for i, m in pop_m:
                if p != i and m > 0:
                    mig.append(f"-m {p} {i} {4*scaled_Ne*m}")
    else:
        mig_matrix = ""

    ms_dict = {"npops": npops,
               "subpop": subpops,
               "theta_loc": theta_loc,
               "scaled_Ne": effective_size,
               "rho_loc": rho_loc,
               "gen_cov": gen_cov,
               "ne_subpop": ne_subpop,
               "mig_matrix": mig_matrix}

    return ms_dict


def model_msbgs(model_dict, ms_dict, demo_df):
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
    npops = ms_dict["npops"]
    dem_list = []
    sourcelist = []

    demo_df_srt = demo_df.set_index("time")
    demo_df_srt.sort_index(inplace=True)
    # "time": float, "event": [Ne, ej, tes, tm], "pop": [0-9], "value": float
    for time, row in demo_df_srt.iterrows():
        new_time = time / (4*scaled_Ne)
        event = row["event"]
        if "Ne" in event:
            pop = int(row["pop"])
            if len(row["value"]) > 1:
                low, high = row["value"]
                size = np.random.randint(low, high)
            else:
                size = row["value"][0]
            init_size[pop] = size
            new_Ne = size / scaled_Ne
            dem_list.append(f'-eN "{new_time}, {new_Ne}"')
        elif "ej" in event:
            pop1, pop2 = row["pop"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            # pop1 -> pop2
            if pop1 not in sourcelist:
                dem_list.append(f"-ej {new_time} {pop1} {pop2}")
                sourcelist.append(pop1)
        elif "es" in event:
            # es_34; es in ms/msmove
            pop1, pop2 = row["pop"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            if pop1 not in sourcelist:
                prop = 1 - row["value"]  # 1-prop are admixed from pop2
                npops += 1
                dem_list.append(f"-es {new_time} {pop1} {prop}")
                dem_list.append(f"-ej {new_time} {npops} {pop2}")
                sourcelist.append(npops)
        elif "m" in event:
            pop1, pop2 = row["pop"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            mig1 = row["value"]*4*scaled_Ne
            mig2 = row["value"]*4*scaled_Ne
            if not any(i in sourcelist for i in [pop1, pop2]):
                if "ms" in event:  # set as symmetrical
                    dem_list.append(f"-em {new_time} {pop1} {pop2} {mig1}")
                    dem_list.append(f"-em {new_time} {pop2} {pop1} {mig2}")
                elif "ma" in event:
                    dem_list.append(f"-em {new_time} {pop1} {pop2} {mig1}")
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
    # TODO: add seed to run parallel
    seed_ls = seed
    # rescale
    ms_dict = sim_syntax(model_dict)

    # build selection command line
    if model_dict["sel_dict"]:
        sel_list = selection_parse(model_dict, ms_dict)
    else:
        sel_list = ''

    # build demographic command line
    dem_events = model_msbgs(model_dict, ms_dict, demo_df)

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
                'migmat': ms_dict["mig_matrix"],
                'demo': " ".join(dem_events),
                'sel_def': " ".join(sel_list),
                'region_def': " ".join(sel_list),
                }

    ms_base = ("{ms} {nhaps} {basepairs} {loci} -t {theta} -r {rho} "
               "-sel {sel_def} -region {region_def} {gen_cov} {subpops} "
               "{ne_subpop} {demo} -ms_format ")
    mscmd = ms_base.format(**ms_params)
    ms_cmd = " ".join(mscmd.split())
    return ms_cmd


def simulate_msbgs(model_dict, demo_df, param_df, ms_path, sim_path, sim_number, outfile):
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
    param_dt = {}
    for tbi, row in param_df.iterrows():
        param_dt[tbi] = list(zip(row["time"], row["value"]))
    demo_df = pd.concat([demo_df, param_df])

    with open(sim_path, 'w') as sims_outfile:
        for i in trange(sim_number):
            for tbi in param_dt.keys():
                ptime, pvalue = param_dt[tbi][i]
                demo_df.loc[tbi]["time"] = ptime
                demo_df.loc[tbi]["value"] = pvalue
            mscmd = command_line(model_dict, demo_df, ms_path, i)
            sims_outfile.write(f"{mscmd} >> {outfile}\n")
