# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:32:28 2020
@author: Scott T. Small

"""
import multiprocessing
import subprocess
from math import ceil

import numpy as np
import pandas as pd
from tqdm import tqdm

from project.sim_modules.readconfig import read_config_stats
from project.stat_modules.sequtils import read_ms_stream
from project.stat_modules.sumstats import PopSumStats
from project.stat_modules.write_stats import headers, stats_out


def selection_parse(ms_dt):
    """Parse selection parameters.

    Parameters
    ----------
    ms_dt : TYPE
        DESCRIPTION.

    Returns
    -------
    sel_list : list
        parameters in discoal syntax

    """
    sel_list = []
    ne0 = ms_dt["Ne0"]
    # sel params
    sel_dict = model_dt["sel_dict"]
    pop0_Ne = sel_dict["pop0_Ne"]
    if not pop0_Ne:
        pop0_Ne = ne0

    sweep_Ne = sel_dict["sweep_Ne"]
    hide = sel_dict["hide"]
    if hide:
        sel_list.append("-h")
    if sweep_Ne:
        sel_list.append(f"-N {sweep_Ne}")

    # =========================================================================
    #  required
    # =========================================================================
    alpha = sel_dict["alpha"]
    sweep_stop = sel_dict["sweep_stop"]
    sweep_site = sel_dict["sweep_site"]

    # *sweep time
    if type(sweep_stop) == list:
        ws_l = sweep_stop[0]/(4 * ne0)
        ws_h = sweep_stop[1]/(4 * ne0)
        sel_list.append(f"-ws 0 -Pu {ws_l} {ws_h}")
    else:
        tau = sweep_stop/(4 * ne0)
        sel_list.append(f"-ws {tau}")

    # *sel coeff
    if type(alpha) == list:
        a_low = alpha[0] * 2 * pop0_Ne
        a_high = alpha[1] * 2 * pop0_Ne
        sel_list.append(f"-Pa {a_low} {a_high}")
    else:
        a = alpha * 2 * pop0_Ne
        sel_list.append(f"-a {a}")

    # *sweep site
    if type(sweep_site) == list:
        s_l, s_h = sweep_site
        assert s_l >= 0
        assert s_h <= 1
        assert s_l < s_h
        sel_list.append(f"-Px {s_l} {s_h}")
    else:
        assert 0 <= sweep_site <= 1
        sel_list.append(f"-x {sweep_site}")

    # =========================================================================
    # soft and partial sweeps
    # =========================================================================
    freq = sel_dict["freq"]
    part_freq = sel_dict["part_freq"]

    # starting freq for soft sweep
    if freq:
        if type(freq) == list:
            f_l, f_h = freq
            assert f_l >= 0
            assert f_h <= 1
            assert f_l < f_h
            sel_list.append(f"-Pf {f_l} {f_h}")
        elif freq > 0:
            assert freq <= 1
            sel_list.append(f"-f {freq}")

    # partial sweep freq
    if part_freq:
        if type(part_freq) == list:
            p_l, p_h = part_freq
            assert p_l >= 0
            assert p_h <= 1
            assert p_l < p_h
            sel_list.append(f"-Pc {p_l} {p_h}")
        elif part_freq > 0:
            assert part_freq <= 1
            sel_list.append(f"-c {part_freq}")

    # =========================================================================
    # linked selection
    # =========================================================================
    # offscreen sweep
    rho_loc = ms_dt["rho_loc"]
    left_rho = sel_dict["left_rho"]

    if left_rho:
        if left_rho[1]:
            time, scale = left_rho
            sel_list.append(f"-ls {time/(4*ne0)} {scale*rho_loc}")
    # recurrent to the left
    rrh_left = sel_dict["rrh_left"]
    if rrh_left:
        sel_list.append(f"-L {rrh_left}")
    # recurrent at locus
    rrh_loc = sel_dict["rrh_loc"]
    if rrh_loc:
        sel_list.append(f"-R {rrh_loc}")
    # =========================================================================
    # adapative recurrent
    # =========================================================================
    # reccurrent adaptive mut
    adapt = sel_dict["adapt"]
    if adapt:
        if type(adapt) == list:
            a_l, a_h = adapt
            sel_list.append(f"-PuA {a_l} {a_h}")
        elif adapt > 0:
            sel_list.append(f"-uA {adapt}")

    return sel_list


def sim_syntax():
    """Create discoal syntax.

    Returns
    -------
    ms_dt : Dict
        contains syntax in discoal format

    """
    ms_dt = {}
    locus_len = model_dt["contig_length"]
    ne0 = np.random.choice(scaled_Ne)
    mu_t = np.random.choice(mu)
    rec_t = np.random.choice(rec)
    pfileout.write(f"{ne0}\t{mu_t}\t{rec_t}\n")
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


def model_discoal(params, ne0: int):
    """Parse model into syntax.

    Parameters
    ----------
    params : pd.DataFrame
        DESCRIPTION.
    ne0 : int
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
            if all(i not in sourcelist for i in [pop1, pop2]):
                prop = row["value"]
                dem_list.append(f"-ea {new_time} {pop1} {pop2} {pop3} {prop}")
        elif "ea" in event:
            # ea345: daughter, parent1, parent2
            pop1, pop2, pop3 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            pop3 = int(pop3)
            if all(i not in sourcelist for i in [pop1, pop2, pop3]):
                prop = row["value"]
                dem_list.append(f"-ea {new_time} {pop1} {pop2} {pop3} {prop}")
        elif "m" in event:
            pop1, pop2 = row["pops"]
            pop1 = int(pop1)
            pop2 = int(pop2)
            mig1 = row["value"]*4*ne0
            if all(i not in sourcelist for i in [pop1, pop2]):
                if "ms" in event:  # set as symmetrical
                    mig2 = row["value"]*4*ne0
                    dem_list.append(f"-m {new_time} {pop1} {pop2} {mig1}")
                    dem_list.append(f"-m {new_time} {pop2} {pop1} {mig2}")
                elif "ma" in event:
                    dem_list.append(f"-m {new_time} {pop1} {pop2} {mig1}")

    return dem_list


def run_simulation(params):
    """Run sims and capture output for stats.

    Parameters
    ----------
    params : list
        DESCRIPTION.

    Returns
    -------
    pop_stats_arr : np.array
        calculated summary stats
    ms_cmd : str
        command line to print to file, later run with bash

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

    seed_arr = np.random.randint(1, 2**31, 2)
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
                'sel': " ".join(sel_list),
                'seed': f"{seed_arr[0]} {seed_arr[1]}"
                }

    ms_base = ("{ms} {nhaps} {loci} {basepairs} -t {theta} -r {rho} "
               "{gen_cov} {subpops} {ne_subpop} {migmat} {demo} {sel} -d {seed} ")
    mscmd = ms_base.format(**ms_params)
    if dry_run:
        ms_cmd = " ".join(mscmd.split())
        print(ms_cmd)
    elif statsconfig:
        length_bp = stats_dt["length_bp"]
        pfe = stats_dt["perfixder"]
        # run sims
        output = subprocess.check_output(mscmd, shell=True)
        pos_ls, hap_ls, count_ls = read_ms_stream(output, nhaps, length_bp, pfe, seq_error=True)
        # calc stats
        i = 0
        stat_mat = np.zeros([model_dt["loci"], header_len])
        for pos, haps, counts in zip(pos_ls, hap_ls, count_ls):
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
    else:
        ms_cmd = " ".join(mscmd.split())
        return ms_cmd


def simulate_discoal(ms_path, model_dict, demo_dataframe, param_df, sim_number,
                     outfile, nprocs, stats_config, dryrun):
    """
    Main simulate.

    Parameters
    ----------
    ms_path : TYPE
        DESCRIPTION.
    model_dict : TYPE
        DESCRIPTION.
    demo_dataframe : TYPE
        DESCRIPTION.
    param_df : TYPE
        DESCRIPTION.
    sim_number : TYPE
        DESCRIPTION.
    outfile : TYPE
        DESCRIPTION.
    nprocs : TYPE
        DESCRIPTION.
    stats_config : TYPE
        DESCRIPTION.
    dryrun : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # =========================================================================
    #  Globals for model params
    # =========================================================================
    global ms_exe
    ms_exe = ms_path
    global demo_df
    demo_df = demo_dataframe
    global model_dt
    model_dt = model_dict
    global dry_run
    dry_run = dryrun

    # model pops
    global nhaps
    nhaps = sum(model_dt["sampleSize"])
    global sample_sizes
    sample_sizes = model_dt["sampleSize"]
    global npops
    npops = len(sample_sizes)

    # set mutation rate
    global mu
    mut_rate = model_dt["mutation_rate"]
    if type(mut_rate) is list:
        if len(mut_rate) == 2:
            low, high = mut_rate
            mu = np.random.uniform(low, high, sim_number)
        else:
            mu = mut_rate
    else:
        mu = [mut_rate]

    # set recombination rate
    global rec
    rec_rate = model_dt["recombination_rate"]
    if type(rec_rate) is list:
        if len(rec_rate) == 2:
            low, high = rec_rate
            rec = np.random.uniform(low, high, sim_number)
        else:
            rec = rec_rate
    else:
        # rec = np.random.exponential(rec_rate, sim_number)
        rec = [rec_rate]

    # set effective population size
    global ploidy
    ploidy = model_dt["ploidy"]
    global init_sizes
    init_sizes = [size * ploidy for size in model_dt["initialSize"]]
    global scaled_Ne
    effective_size = model_dt["eff_size"]
    if type(effective_size) is list:
        if len(effective_size) == 2:
            low, high = effective_size
            scaled_Ne = np.random.randint(low, high, sim_number) * ploidy
        else:
            scaled_Ne = effective_size
    else:
        scaled_Ne = [effective_size * ploidy]
    global pfileout
    pfileout = open(f"{outfile}.{sim_number}.ne_mu_rec.out", 'w')
    # =========================================================================
    #  Main simulations
    # =========================================================================
    # set up generator fx for MP
    event = param_df["event"].values
    pops = param_df["pops"].values
    time_arr = list(zip(*param_df["time"].values))
    value_arr = list(zip(*param_df["value"].values))
    param_gen = ({"time": time_arr[i], "event": event, "pops": pops, "value": value_arr[i]} for i in range(sim_number))
    param_gen = list(param_gen)

    # check nprocs
    if nprocs > multiprocessing.cpu_count():  # check that there are not more requested than available
        print("not {nprocs} processors available, setting to {multiprocessing.cpu_count()}")
        nprocs = multiprocessing.cpu_count()

    global statsconfig
    statsconfig = ''
    global stats_dt
    global header_len
    global header
    # perform sims
    if dry_run:
        for param in param_gen:
            run_simulation(param)
            break
    elif stats_config:
        stats_dt = read_config_stats(stats_config)
        statsconfig = stats_config
        # write headers
        pops_outfile = open(f"{outfile}.{sim_number}.pop_stats.txt", 'w')
        pops_outfile, header_, header_ls = headers(pops_outfile, stats_dt)
        header_len = header_
        header = header_ls
        if nprocs == 1:
            for param in tqdm(param_gen):
                pop_stats_arr = run_simulation(param)
                pops_outfile = stats_out(pop_stats_arr, pops_outfile, nprocs)
        else:
            # chunk and MP
            nk = nprocs * 10
            chunk_list = [param_gen[i:i + nk] for i in range(0, len(param_gen), nk)]
            chunksize = ceil(nk/nprocs)
            pool = multiprocessing.Pool(nprocs)
            for i, args in enumerate(chunk_list):
                pop_stats_arr = pool.map(run_simulation, args, chunksize=chunksize)
                pops_outfile = stats_out(pop_stats_arr, pops_outfile, nprocs)
                print(i)
            pool.close()
        pops_outfile.close()
    else:
        with open(f"{outfile}.{sim_number}.sims.cmd.txt", 'w') as sims_outfile:
            for param in tqdm(param_gen):
                mscmd = run_simulation(param)
                sims_outfile.write(f"{mscmd} >> {outfile}\n")
    pfileout.close()
