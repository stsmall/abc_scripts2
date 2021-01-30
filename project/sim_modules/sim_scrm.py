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
    pfileout.write(f"{ne0}\t{mu_t}\t{rec_t}\n")
    # calc theta
    theta_loc = 4 * ne0 * mu_t * locus_len
    # calc rho rate
    rho_loc = 4 * ne0 * rec_t * locus_len

    # subops
    mig_mat = model_dt["migmat"]
    subpops = f"-I {npops} {' '.join(map(str, sample_sizes))}"
    ne_sub_pops = [f"-n {i+1} {pop_ne/ne0}" for i, pop_ne in enumerate(init_sizes)]
    ne_subpop = " ".join(ne_sub_pops)
    if np.sum(mig_mat) > 0:
        mig_matrix = f"-ma {' '.join(map(str, mig_mat))}"
    else:
        mig_matrix = ""

    ms_dt = {"npops": npops,
             "subpop": subpops,
             "theta_loc": theta_loc,
             "Ne0": ne0,
             "rho_loc": rho_loc,
             "ne_subpop": ne_subpop,
             "mig_matrix": mig_matrix}

    return ms_dt


def model_scrm(params, ne0):
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
    # ej, mig from i set to 0, growth rates unchanged
    # en/eN, growth rates set to 0
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
            pop1 = int(pop1) + 1
            if type(row["value"]) is list:  # TODO: check this
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
            pop1 = int(pop1) + 1
            pop2 = int(pop2) + 1
            # pop1 -> pop2
            if pop1 not in sourcelist:
                dem_list.append(f"-ej {new_time} {pop1} {pop2}")
                sourcelist.append(pop1)
        elif "es" in event:
            # es34 is read as es_343 in discoal: daughter, parent1, parent2
            pop1, pop2 = row["pops"]
            pop1 = int(pop1) + 1
            pop2 = int(pop2) + 1
            if all(i not in sourcelist for i in [pop1, pop2]):
                prop = row["value"]
                dem_list.append(f"-eps {new_time} {pop1} {pop2} {prop}")
        elif "m" in event:
            pop1, pop2 = row["pops"]
            pop1 = int(pop1) + 1
            pop2 = int(pop2) + 1
            mig1 = row["value"]*4*ne0
            if all(i not in sourcelist for i in [pop1, pop2]):
                if "ms" in event:  # set as symmetrical
                    mig2 = row["value"]*4*ne0
                    dem_list.append(f"-em {new_time} {pop1} {pop2} {mig1}")
                    dem_list.append(f"-em {new_time} {pop2} {pop1} {mig2}")
                elif "ma" in event:
                    dem_list.append(f"-em {new_time} {pop1} {pop2} {mig1}")
                elif "ema" in event:
                    # dem_list.append(f"-ema {new_time} {pop1} {pop2} {mig1}")
                    # set migration matrix ... have to read in another one??
                    pass
    return dem_list


def run_simulation(param_df):
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

    # build demographic command line
    dem_events = model_scrm(param_df, ms_dt["Ne0"])

    # gather command line args
    ms_params = {
                'ms': ms_exe,
                'nhaps': nhaps,
                'loci': model_dt["loci"],
                'theta': ms_dt["theta_loc"],
                'rho': ms_dt['rho_loc'],
                'basepairs': model_dt["contig_length"],
                'subpops': ms_dt["subpop"],
                'ne_subpop': ms_dt["ne_subpop"],
                'migmat': ms_dt["mig_matrix"],
                'demo': " ".join(dem_events),
                }

    if dry_run:
        ms_base = ("{ms} {nhaps} 1 -t 1 -r 1 1000 "
                   "{subpops} {ne_subpop} {demo} --print-model")
        mscmd = ms_base.format(**ms_params)
        output = subprocess.check_output(mscmd, shell=True)
        for line in output.splitlines():
            if line.startswith(b"//"):
                break
            else:
                print(line.decode())
    elif statsconfig:
        # default is -l 500.
        # smc' is -l 0.
        # exact -l -1 or 100000
        # good choices: scrm_l = "-l 100r"; scrm_l = "-l 250r"
        ms_base = ("{ms} {nhaps} {loci} -t {theta} -r {rho} {basepairs} "
                   "{subpops} {ne_subpop} {migmat} {demo} " + scrm_l)
        mscmd = ms_base.format(**ms_params)
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
        ms_base = ("{ms} {nhaps} {loci} -t {theta} -r {rho} {basepairs} "
                   "{subpops} {ne_subpop} {demo}")
        mscmd = ms_base.format(**ms_params)
        ms_cmd = " ".join(mscmd.split())
        return ms_cmd


def simulate_scrm(ms_path, model_dict, demo_dataframe, param_df, sim_number,
                  outfile, nprocs, stats_config, dryrun, approx):
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
    #  Globals for model params
    # =========================================================================
    global demo_df
    demo_df = demo_dataframe
    global model_dt
    model_dt = model_dict
    global ms_exe
    ms_exe = ms_path
    global dry_run
    dry_run = dryrun
    global scrm_l
    scrm_l = f"-l {approx}" if approx else ''
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
    # set Ne
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
    pfileout = open(f"{outfile}.ne_mu_rec.out", 'w')
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
        pops_outfile = open(f"{outfile}.pop_stats.txt", 'w')
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
        with open(f"{outfile}.sims.cmd.txt", 'w') as sims_outfile:
            for param in tqdm(param_gen):
                mscmd = run_simulation(param)
                sims_outfile.write(f"{mscmd} >> {outfile}.sims.out\n")
    pfileout.close()
