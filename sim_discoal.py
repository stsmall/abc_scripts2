#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:32:28 2020
@author: Scott T. Small

"""
import sys
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
    scaled_Ne = ms_dict["scaled_Ne"]
    rho_loc = ms_dict["rho_loc"],
    # sel params
    sel_dict = model_dict["sel_dict"]
    pop0_Ne = sel_dict["pop0_Ne"]
    if pop0_Ne == 0:
        pop0_Ne = scaled_Ne
    sweep_Ne = sel_dict["sweep_Ne"]
    alpha = sel_dict["alpha"]
    freq = sel_dict["freq"]
    sweep_stop = sel_dict["sweep_stop"]
    sweep_site = sel_dict["sweep_site"]
    part_freq = sel_dict["part_freq"]
    adapt = sel_dict["adapt"]
    hide = sel_dict["hide"]

    sel_list = []

    if sweep_Ne > 0:
        sel_list.append(f"-N {sweep_Ne}")
    # sweep time
    if type(sweep_stop) == list:
        ws_l = sweep_stop[0]/(4 * scaled_Ne)
        ws_h = sweep_stop[1]/(4 * scaled_Ne)
        sel_list.append(f"-ws 0 -Pu {ws_l} {ws_h}")
    else:
        tau = sweep_stop/(4 * scaled_Ne)
        sel_list.append(f"-ws {tau}")
    # sel coeff
    if type(alpha) == list:
        a_low = alpha[0] * 2 * pop0_Ne
        a_high = alpha[1] * 2 * pop0_Ne
        sel_list.append(f"-Pa {a_low} {a_high}")
    else:
        a = alpha * 2 * pop0_Ne
        sel_list.append(f"-a {a}")
    # offscreen sweep
    left_rho = sel_dict["left_rho"]
    if left_rho[1] > 0:
        time, scale = left_rho
        sel_list.append(f"-ls {time/(4*scaled_Ne)} {scale*rho_loc}")
    # recurrent to the left
    rrh_left = sel_dict["rrh_left"]
    if rrh_left > 0:
        sel_list.append(f"-L {rrh_left}")
    # recurrent at locus
    rrh_loc = sel_dict["rrh_loc"]
    if rrh_loc > 0:
        sel_list.append(f"-R {rrh_loc}")
    # starting freq
    if type(freq) == list:
        f_l, f_h = freq
        assert f_l >= 0
        assert f_h <= 1
        assert f_l < f_h
        sel_list.append(f"-Pf {f_l} {f_h}")
    elif freq > 0:
        assert freq <= 1
        sel_list.append(f"-f {freq}")
    else:
        # hard sweep
        pass
    # sweep site
    if type(sweep_site) == list:
        s_l, s_h = sweep_site
        assert s_l >= 0
        assert s_h <= 1
        assert s_l < s_h
        sel_list.append(f"-Px {s_l} {s_h}")
    else:
        assert 0 <= sweep_site <= 1
        sel_list.append(f"-x {sweep_site}")
    # partial sweep freq
    if type(part_freq) == list:
        p_l, p_h = part_freq
        assert p_l >= 0
        assert p_h <= 1
        assert p_l < p_h
        sel_list.append(f"-Pc {p_l} {p_h}")
    elif part_freq > 0:
        assert part_freq <= 1
        sel_list.append(f"-c {part_freq}")
    else:
        # no partial sweep
        pass
    # reccurrent adaptive mut
    if type(adapt) == list:
        a_l, a_h = adapt
        sel_list.append(f"-PuA {a_l} {a_h}")
    elif adapt > 0:
        sel_list.append(f"-uA {adapt}")
    else:
        # no recurrent adaptive mutation
        pass
    if hide:
        sel_list.append("-h")
    return sel_list


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
    ploidy = model_dict["ploidy"] * 2
    locus_len = model_dict["contig_length"]
    effective_size = model_dict["eff_size"]

    # need theta
    theta_arr = model_dict["theta"]
    if len(theta_arr) == 1:
        # command line or just 1 entry in file
        theta_nuc = theta_arr
    else:
        theta_nuc = np.random.choice(theta_arr)
    theta_loc = theta_nuc * locus_len

    # need scaled Ne
    mut_rate = model_dict["mutation_rate"]
    if mut_rate:
        if type(mut_rate) == list:
            low, high = mut_rate
            mut_rate = np.random.uniform(low, high)
        scaled_Ne = (ploidy/4.0) * int(np.round((theta_nuc/(4*mut_rate))))
    elif effective_size:
        scaled_Ne = effective_size
    else:
        raise ValueError("must provide mutation rate or effective size")
        return None

    # recombination rate
    rho_arr = model_dict["rho"]
    rho_mu = model_dict["rho_mu"]
    rec_rate = model_dict["recombination_rate"]
    if len(rho_arr) == 1:
        rho = rho_arr * locus_len
    elif rho_mu:
        rho = theta_loc * rho_mu
    elif len(rho_arr) > 1:
        rho = np.random.choice(rho_arr) * locus_len
    elif rec_rate:
        if type(rec_rate) == list:
            low, high = rec_rate
            rec_rate = np.random.uniform(low, high)
        rho = ploidy*scaled_Ne*rec_rate * locus_len
    else:
        rho = 0

    # gene conversion
    gen_conversion = model_dict["gene_conversion"][0]
    if gen_conversion > 0:
        tract = model_dict["gene_conversion"][1]
        gen_cov = f"-gr {gen_conversion*rho} {tract}"
    else:
        gen_cov = ""

    # subops
    init_sizes = [size * (ploidy/4.0) for size in model_dict["initialSize"]]
    grow_rate = model_dict["growthRate"]
    mig_mat = model_dict["migMat"]
    subpops = f"-p {npops} {' '.join(map(str, sample_sizes))}"
    ne_sub_pops = [f"-en 0 {i} {pop_ne/scaled_Ne}" for i, pop_ne in enumerate(init_sizes)]
    ne_subpop = " ".join(ne_sub_pops)
    grow_subpop = []
    if mig_mat:
        mig = []
        mig_matrix = zip(*mig_mat)
        for p, pop_m in enumerate(mig_matrix):
            for i, m in pop_m:
                if p != i and m > 0:
                    mig.append(f"-m {p} {i} {m}")
    else:
        mig_matrix = ""

    ms_dict = {"npops": npops,
               "subpop": subpops,
               "theta_loc": theta_loc,
               "scaled_Ne": scaled_Ne,
               "rho_loc": rho,
               "gen_cov": gen_cov,
               "ne_subpop": ne_subpop,
               "grow_subpop": grow_subpop,
               "mig_matrix": mig_matrix}

    return ms_dict


def model_discoal(model_dict, ms_dict, demo_df):
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
    ploidy = model_dict["ploidy"] * 2
    scaled_Ne = ms_dict["scaled_Ne"]
    init_size = list(model_dict["initialSize"])
    dem_list = []
    sourcelist = []

    demo_df_srt = demo_df.set_index("time")
    demo_df_srt.sort_index(inplace=True)
    # "time": float, "event": [Ne, ej, tes, tm], "pop": [0-9], "value": float
    for time, row in demo_df_srt.iterrows():
        new_time = time / (ploidy*scaled_Ne)
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
            mig1 = row["value"]*ploidy*init_size[pop1]
            mig2 = row["value"]*ploidy*init_size[pop2]
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
    seed_ls = [seed, np.random.randint(1, 2**32-1, 1), np.random.randint(1, 2**32-1, 1)]
    # rescale
    ms_dict = sim_syntax(model_dict)

    # build selection command line
    if model_dict["sel_dict"]:
        sel_list = selection_parse(model_dict, ms_dict)
    else:
        sel_list = ''

    # build demographic command line
    dem_events = model_discoal(model_dict, ms_dict, demo_df)

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
                'demo': " ".join(dem_events),
                'sel': " ".join(sel_list)
                }

    ms_base = ("{ms} {nhaps} {loci} {basepairs} -t {theta} -r {rho} "
               "{gen_cov} {subpops} {ne_subpop} {demo} {sel}")
    mscmd = ms_base.format(**ms_params)
    ms_cmd = " ".join(mscmd.split())
    return ms_cmd


def simulate_discoal(model_dict, demo_df, param_df, ms_path, sim_path, sim_number, outfile):
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
