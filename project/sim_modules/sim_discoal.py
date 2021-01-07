#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:32:28 2020
@author: Scott T. Small

"""
import numpy as np
import pandas as pd
import tqdm


def selection_parse(ms_dt):
    """Parse selection dict for discoal.

    Parameters
    ----------
    model_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ne0 = ms_dt["Ne0"]
    rho_loc = ms_dt["rho_loc"]
    # sel params
    sel_dict = model_dt["sel_dict"]
    pop0_Ne = sel_dict["pop0_Ne"]
    if not pop0_Ne:
        pop0_Ne = ne0
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
        ws_l = sweep_stop[0]/(4 * ne0)
        ws_h = sweep_stop[1]/(4 * ne0)
        sel_list.append(f"-ws 0 -Pu {ws_l} {ws_h}")
    else:
        tau = sweep_stop/(4 * ne0)
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
        sel_list.append(f"-ls {time/(4*ne0)} {scale*rho_loc}")
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


def sim_syntax():
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
    ms_dt = {}
    locus_len = model_dt["contig_length"]
    ne0 = np.random.choice(scaled_Ne)
    mu_t = np.random.choice(mu)
    rec_t = np.random.choice(rec)
    # calc theta
    theta_loc = 4 * ne0 * mu_t * locus_len
    # calc rho rate
    rho_loc = 4 * ne0 * rec_t * locus_len

    # gene conversion
    gen_conversion = model_dt["gene_conversion"][0]
    if gen_conversion > 0:
        tract = model_dt["gene_conversion"][1]
        gen_cov = f"-gr {gen_conversion / rec_t} {tract}"
    else:
        gen_cov = ""

    # subops
    mig_mat = model_dt["migmat"]
    subpops = f"-p {npops} {' '.join(map(str, sample_sizes))}"
    ne_sub_pops = [f"-en 0 {i} {pop_ne/ne0}" for i, pop_ne in enumerate(init_sizes)]
    ne_subpop = " ".join(ne_sub_pops)
    if np.sum(mig_mat) > 0:
        mig = []
        mig_matrix = zip(*mig_mat)
        for p, pop_m in enumerate(mig_matrix):
            for i, m in pop_m:
                if p != i and m > 0:
                    mig.append(f"-m {p} {i} {4*ne0*m}")
    else:
        mig_matrix = ""

    ms_dt = {"npops": npops,
             "subpop": subpops,
             "theta_loc": theta_loc,
             "Ne0": ne0,
             "rho_loc": rho_loc,
             "gen_cov": gen_cov,
             "ne_subpop": ne_subpop,
             "mig_matrix": mig_matrix}

    return ms_dt


def model_discoal(params, ne0):
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
        new_time = time / (4*ne0)  # 4N0gens
        event = row["event"]
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
            if pop1 not in sourcelist:
                new_Ne = size / ne0
                dem_list.append(f"-en {new_time} {pop1} {new_Ne}")
        elif "ej" in event:
            pop1, pop2 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            # pop1 -> pop2
            if pop1 not in sourcelist:
                dem_list.append(f"-ed {new_time} {pop1} {pop2}")
                sourcelist.append(pop1)
        elif "es" in event:
            # es34 is read as es_343 in discoal: daughter, parent1, parent2
            pop1, pop2 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            pop3 = pop1
            if not any(i in sourcelist for i in [pop1, pop2]):
                prop = row["value"]
                dem_list.append(f"-ea {new_time} {pop1} {pop2} {pop3} {prop}")
        elif "ea" in event:
            # ea345: daughter, parent1, parent2
            pop1, pop2, pop3 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            pop3 = int(pop3)
            if not any(i in sourcelist for i in [pop1, pop2, pop3]):
                prop = row["value"]
                dem_list.append(f"-ea {new_time} {pop1} {pop2} {pop3} {prop}")
        elif "m" in event:
            pop1, pop2 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            mig1 = row["value"]*4*ne0
            if not any(i in sourcelist for i in [pop1, pop2]):
                if "ms" in event:  # set as symmetrical
                    mig2 = row["value"]*4*ne0
                    dem_list.append(f"-m {new_time} {pop1} {pop2} {mig1}")
                    dem_list.append(f"-m {new_time} {pop2} {pop1} {mig2}")
                elif "ma" in event:
                    dem_list.append(f"-m {new_time} {pop1} {pop2} {mig1}")

    return dem_list


def command_line(params):
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
    # rescale
    ms_dt = sim_syntax()

    # build selection command line
    if model_dt["sel_dict"]:
        sel_list = selection_parse(ms_dt)
    else:
        sel_list = ''

    # build demographic command line
    dem_events = model_discoal(params, ms_dt["Ne0"])

    # gather command line args
    ms_params = {
                'ms': ms_exe,
                'nhaps': nhaps,
                'loci': model_dt["loci"],
                'theta': ms_dt["theta_loc"],
                'rho': ms_dt['rho_loc'],
                'gen_cov': ms_dt['gen_cov'],
                'basepairs': model_dt["contig_length"],
                'subpops': ms_dt["subpop"],
                'ne_subpop': ms_dt["ne_subpop"],
                'migmat': ms_dt["mig_matrix"],
                'demo': " ".join(dem_events),
                'sel': " ".join(sel_list)
                }

    ms_base = ("{ms} {nhaps} {loci} {basepairs} -t {theta} -r {rho} "
               "{gen_cov} {subpops} {ne_subpop} {demo} {sel}")
    mscmd = ms_base.format(**ms_params)
    ms_cmd = " ".join(mscmd.split())
    return ms_cmd


def simulate_discoal(model_dict, demo_dataframe, param_df, ms_path, sim_path, sim_number, outfile):
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
    # =========================================================================
    #  Globals
    # =========================================================================
    global demo_df
    global model_dt
    global nhaps
    global mu
    global rec
    global scaled_Ne
    global initial_model
    global hybrid_model
    global outfile_tree
    global ms_exe
    global sample_sizes
    global npops
    global ploidy
    global init_sizes
    # =========================================================================
    # declare globals
    ms_exe = ms_path
    model_dt = model_dict
    demo_df = demo_dataframe
    nhaps = sum(model_dt["sampleSize"])
    sample_sizes = model_dt["sampleSize"]
    npops = len(sample_sizes)
    ploidy = model_dt["ploidy"]
    init_sizes = [size * ploidy for size in model_dt["initialSize"]]
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
        scaled_Ne = [effective_size * ploidy]

    # set up generator fx
    event = param_df["event"].values
    pops = param_df["pops"].values
    time_arr = list(zip(*param_df["time"].values))
    value_arr = list(zip(*param_df["value"].values))
    param_gen = ({"time": time_arr[i], "event":event, "pops":pops, "value": value_arr[i]} for i in range(sim_number))
    progressbar = tqdm.tqdm(total=sim_number, unit='sims')
    with open(sim_path, 'w') as sims_outfile:
        for param in param_gen:
            progressbar.update(1)
            mscmd = command_line(param)
            sims_outfile.write(f"{mscmd} >> {outfile}\n")
    progressbar.close()