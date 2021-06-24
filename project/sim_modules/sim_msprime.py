# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:32:28 2020
@author: Scott T. Small

"""
import multiprocessing
from math import ceil

import msprime as msp
import numpy as np
import pandas as pd
from tqdm import tqdm

from project.sim_modules.readconfig import read_config_stats
from project.stat_modules.sequtils import read_trees
from project.stat_modules.sumstats import PopSumStats
from project.stat_modules.write_stats import headers, stats_out


def demo_config(params, demo_events):
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
    # mass migration does not reset migration rates
    # population size changes do not make growth rates 0

    demo_param_df = pd.concat([demo_df, pd.DataFrame(params)])
    demo_param_df_srt = demo_param_df.set_index("time")
    demo_param_df_srt.sort_index(inplace=True)
    sourcelist = []
    poplist = [f"pop_{i}" for i in range(len(model_dt["sampleSize"]))]
    # "time": float, "event": [Ne, ej, tes, tm], "pop": [0-9], "value": float
    for time, row in demo_param_df_srt.iterrows():
        event = row["event"]
        if "Ne" in event:
            pix = row["pops"][0]
            pop = f"pop_{pix}"
            if type(row["value"]) is list:
                if len(row["value"]) > 1:
                    low = row["value"][0]
                    high = row["value"][1]
                    size = np.random.randint(low, high)
                else:
                    size = row["value"][0]
            else:
                size = row["value"]
            if pop not in sourcelist and pop in poplist:
                demo_events.add_population_parameters_change(time=time, initial_size=size, population=pop, growth_rate=0)
        elif "ej" in event:
            # add_population_split(time=int, derived=["pop_0", "pop_1"], acestral="pop_01")
            pix, p2ix, ancix = row["pops"]
            pop1 = f"pop_{pix}"
            pop2 = f"pop_{p2ix}"
            anc = f"pop_{ancix}"
            if anc not in poplist:
                # get Ne for anc pop
                ne = demo_param_df_srt[demo_param_df_srt["event"] == "Ne"]
                ne_anc = ne.iloc[np.where(pd.Series([x for _list in ne["pops"] for x in _list ]) == ancix)]
                size_anc = ne_anc[ne_anc.index > time].iloc[0]["value"]
                # need value >= time
                #ne_anc = ne.iloc[i for i, pop in enumerate(ne["pops"].values) if anc in pop]["value"]
                if type(size_anc) is list:
                    if len(size_anc) > 1:
                        low = size_anc[0]
                        high = size_anc[1]
                        size = np.random.randint(low, high)
                    else:
                        size = size_anc[0]
                else:
                    size = size_anc
                demo_events.add_population(name=anc, initial_size=size)  # first instance of anc in df
                poplist.append(anc)
            if pop1 not in sourcelist and pop2 not in sourcelist:
                demo_events.add_population_split(time=time, derived=[pop1, pop2], ancestral=anc)
                sourcelist.append(pop1)
                sourcelist.append(pop2)
        elif "ev" in event:  # add_mass_migration
            # add_mass_migration(time=int, source=0, dest=1, proportion=0.5)  # does not change mig rates
            # ev12
            pix, p2ix = row["pops"]
            pop1 = f"pop_{pix}"
            pop2 = f"pop_{p2ix}"
            if all(i not in sourcelist for i in [pop1, pop2]):
                prop = row["value"]
                demo_events.add_mass_migration(time=time, source=pop1, dest=pop2, proportion=prop)
        elif "ea" in event:
            # add_admixture(time=int, derived="pop_{}", ancestral=["pop_0", "pop_1"], proportions=[0.25, 0.75])
            # ea345: derived, parent1, parent2
            pix, p2ix, p3ix = row["pops"]
            pop1 = f"pop_{pix}"
            pop2 = f"pop_{p2ix}"
            pop3 = f"pop_{p3ix}"
            if all(i not in sourcelist for i in [pop1, pop2, pop3]):
                prop = row["value"]
                demo_events.add_admixture(time=time, derived=pop1, ancestral=[pop2, pop3], proportion=[1-prop, prop])
                sourcelist.append(pop1)
        elif "m" in event:
            # add_migration_rate_change(time=int, source= , dest= , rate=flt)  # if None will set all
            # add_symmetric_migration_rate_change(time=int, populations=[pop_0, pop_1], rate=float)
            pix = row["pops"]
            pops_ = [f"pop_{ix}" for ix in pix]
            mig = row["value"]
            if all(i not in sourcelist for i in pops_):
                if "ms" in event:  # set as symmetrical
                    demo_events.add_symmetric_migration_rate_change(time=time, populations=pops_, rate=mig)
                elif "ma" in event:
                    demo_events.add_migration_rate_change(time=time, source=pops_[0] , dest=pops_[1], rate=mig)
                elif "m0" in event:
                    demo_events.add_migration_rate_change(time=time, rate=0)
    return demo_events


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
    demo_events = msp.Demography()

    # set samples sizes, here in diploids. so nsam/2
    sample_sizes = model_dt["sampleSize"]
    samples = {f'pop_{i}':sample_size/2 for i, sample_size in enumerate(sample_sizes)}

    # set population sizes
    init_sizes = [size * ploidy for size in model_dt["initialSize"]]
    for i, init in enumerate(init_sizes):
        demo_events.add_population(name=f"pop_{i}", initial_size=init)

    # set migration rates from migration matrix if > 0
    mig_mat = model_dt["migmat"]
    if np.sum(mig_mat) > 0:
        sym_rates = [model_dt["migmat"][i, j] for i, j in zip(*np.where(~np.eye(model_dt["migmat"].shape[0],dtype=bool)))]
        if sym_rates.count(sym_rates[0]) == len(sym_rates):
            demo_events.set_migration_rate(source=None, dest=None, rate=sym_rates[0])
        else:
            mig_matrix = zip(*mig_mat)
            for p, pop_m in enumerate(mig_matrix):
                for i, m in pop_m:
                    if p != i and m > 0:
                        demo_events.set_migration_rate(source=p, dest=i, rate=m)

    # build demographic command line
    demo_events = demo_config(param_df, demo_events)

    # set hybrid models
    if hybrid_switch_over:
        model_list = [msp.DiscreteTimeWrightFisher(duration=hybrid_switch_over),
                      msp.StandardCoalescent(),]
    else:
        model_list = msp.StandardCoalescent()

    # check demo
    if dry_run:
        checkDemo(demo_events)
        return None

    trees = msp.sim_ancestry(samples,
                             recombination_rate=param_df["rec_t"],
                             demography=demo_events,
                             num_replicates=model_dt["loci"],
                             sequence_length=model_dt["contig_length"],
                             model=model_list)
    # calc stats
    stat_mat = np.zeros([model_dt["loci"], header_len])
    length_bp = stats_dt["length_bp"]
    pfe = stats_dt["perfixder"]
    for i, tree in enumerate(trees):
        tree = msp.sim_mutations(tree,
                                 rate=param_df["mu_t"],
                                 model="binary")
        stats_ls = []
        pos, haps, counts, bp = read_trees(tree, length_bp, pfe, seq_error=True)
        stats_dt["breakpoints"] = bp
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

    return np.nanmean(stat_mat, axis=0)


def checkDemo(demo_events):
    """Print out demography for debugging."""
    dd = demo_events.debug()
    print(dd)
    #dd.population_size_trajectory()
    #dd.possible_lineage_locations()
    #dd.lineage_probabilities()
    #dd.coalescence_rate_trajectory()
    #dd.mean_coalescence_time()
    # Inverse Instantaneous coalescence rates
    #https://tskit.dev/msprime/docs/stable/demography.html#sec-demography-numerical


def simulate_msprime(model_dict, demo_dataframe, param_df, sim_number: int,
                     outfile: str, nprocs: int, stats_config: str, dryrun: bool,
                     order: bool):
    """Run code for simulating msprime.

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
    #  Globals for model params
    # =========================================================================
    # set info dicts
    global model_dt
    model_dt = model_dict
    global demo_df
    demo_df = demo_dataframe

    # set dryrun
    global dry_run
    dry_run = dryrun

    # set models and switching
    global initial_model
    initial_model = "hudson"
    global hybrid_model
    hybrid_model = "hudson"  # dtwf, smc, smc_prime
    global hybrid_switch_over
    if dryrun:
        hybrid_switch_over = ''  # demo debug does not handle hybrid models
    else:
        hybrid_switch_over = ''  # int of gens, e.g., 500

    # set mutation rate
    global mu
    l_mu = np.nan
    mut_rate = model_dt["mutation_rate"]
    if type(mut_rate) is list:
        if len(mut_rate) == 2:
            low, high = mut_rate
            mu = np.random.uniform(low, high, sim_number)
        else:
            if len(mut_rate) < sim_number:
                mu = np.random.choice(mut_rate, sim_number)
            elif order:
                l_mu = len(mu)
            else:
                mu = mut_rate

    else:
        mu = [mut_rate] * sim_number

    # set recombination rate
    global rec
    l_rec = np.nan
    rec_rate = model_dt["recombination_rate"]
    if type(rec_rate) is list:
        if len(rec_rate) == 2:
            low, high = rec_rate
            rec = np.random.uniform(low, high, sim_number)
            # rec = np.random.exponential(rec_rate, sim_number)
        else:
            if len(rec_rate) < sim_number:
                rec = np.random.choice(rec_rate, sim_number)
            elif order:
                l_rec = len(rec)
            else:
                rec = rec_rate
    else:
        rec = [rec_rate] * sim_number

    # set ploidy
    global ploidy
    ploidy = model_dt["ploidy"]

    # set effective pop size
    global scaled_Ne
    l_ne = np.nan
    effective_size = model_dt["eff_size"]
    if type(effective_size) is list:
        if len(effective_size) == 2:
            low, high = effective_size
            scaled_Ne = np.random.randint(low, high, sim_number) * ploidy
        else:
            if len(effective_size) < sim_number:
                scaled_Ne = np.random.choice(effective_size, sim_number)
                scaled_Ne = list(scaled_Ne * ploidy)
            elif order:
                l_ne = len(scaled_Ne)
                scaled_Ne = list(effective_size * ploidy)
            else:
                scaled_Ne = list(effective_size * ploidy)
    else:
        scaled_Ne = [effective_size * ploidy] * sim_number
    # =========================================================================
    #  Main simulations
    # =========================================================================
    # set up generator fx for MP
    if order:
        l_min = np.nanmin([l_mu, l_rec, l_ne])
        sim_number = int(l_min)
        print(f"order requested, setting sim_number to shortest param file: {l_min}")

    with open(f"{outfile}.ne_mu_rec.out", 'w') as pfile:
        pfile.write("Ne\tmu\trec\n")
        for i in range(sim_number):
            pfile.write(f"{int(scaled_Ne[i])}\t{mu[i]}\t{rec[i]}\n")

    event = param_df["event"].values
    pops = param_df["pops"].values
    time_arr = list(zip(*param_df["time"].values))
    value_arr = list(zip(*param_df["value"].values))
    param_gen = ({"ne_t": scaled_Ne[i], "mu_t": mu[i], "rec_t": rec[i],
                  "time": time_arr[i], "event": event, "pops": pops, "value": value_arr[i]} for i in range(sim_number))
    # param_gen = ({"time": time_arr[i], "event": event, "pops": pops, "value": value_arr[i]} for i in range(sim_number))
    param_gen = list(param_gen)

    # check nprocs
    if nprocs > multiprocessing.cpu_count():  # check that there are not more requested than available
        print("not {nprocs} processors available, setting to {multiprocessing.cpu_count()}")
        nprocs = multiprocessing.cpu_count()
    # perform sims
    global stats_dt
    global header_len
    global header

    if dry_run:
        for param in param_gen:
            run_simulation(param)
            break
    elif stats_config:
        stats_dt = read_config_stats(stats_config)
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
            nk = nprocs * 10  # tricky, how many jobs for each processor
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
        print("No stats file given with msprime")
