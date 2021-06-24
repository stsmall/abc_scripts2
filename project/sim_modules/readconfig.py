#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 09:02:36 2020
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

"""
import os
import configparser
import numpy as np
from project.stat_modules.ld import ld_intervals


def read_config_sims(configFile, ms_path):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(configFile)
    config_path = os.path.split(os.path.abspath(configFile))[0]

    # simulation section
    sim = "simulation"
    contig_len = int(config.getfloat(sim, "contiglen"))
    num_loci = config.getint(sim, "loci")

    ne_size = config.get(sim, "effective_population_size")
    if "," in ne_size:
        effective_size = list(map(float, ne_size.split(",")))
        effective_size = list(map(int, effective_size))
    elif ne_size[0].isalpha():
        if os.path.exists(ne_size):
            print(f"loading {ne_size} ...")
            effective_size = list(np.loadtxt(ne_size))
        else:
            print(f"loading {os.path.join(config_path, ne_size)} ...")
            effective_size = list(np.loadtxt(os.path.join(config_path, ne_size)))
    else:
        effective_size = int(float(ne_size))

    recomb_rate = config.get(sim, "recombination_rate")
    if recomb_rate:
        if "," in recomb_rate:
            recomb_rate = list(map(float, recomb_rate.split(",")))
        elif recomb_rate[0].isalpha():
            if os.path.exists(recomb_rate):
                print(f"loading {recomb_rate} ...")
                recomb_rate = list(np.loadtxt(recomb_rate))
            else:
                print(f"loading {os.path.join(config_path, recomb_rate)} ...")
                recomb_rate = list(np.loadtxt(os.path.join(config_path, recomb_rate)))
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
                mutation_rate = list(np.loadtxt(mutation_rate))
            else:
                print(f"loading {os.path.join(config_path, mutation_rate)} ...")
                mutation_rate = list(np.loadtxt(os.path.join(config_path, mutation_rate)))
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
        pop0_Ne = config.get(sel, "sweep_population_Ne")
        mode = config.get(sel, "mode")
        alpha = config.get(sel, "alpha")
        sweep_start = config.get(sel, "sweep_start")
        sweep_end = config.get(sel, "sweep_end")
        allele_freq = config.get(sel, "allele_freq")
        partial_freq = config.get(sel, "partial_sweep")
        sweep_site = config.get(sel, "sweep_site")
        hide = config.getboolean(sel, "hide")
        sweep_effective_size = config.get(sel, "sweep_effective_size")
        adapt = config.get(sel, "adapt_mutrate")
        left_rho = config.get(sel, "leftRho")
        rrh_left = config.get(sel, "Lrecurrent")
        rrh_loc = config.get(sel, "Rrecurrent")
        sel_dict = {
                    "mode": mode,
                    "pop0_Ne": pop0_Ne,
                    "alpha": alpha,
                    "sweep_start": sweep_start,
                    "sweep_stop": sweep_end,
                    "freq": allele_freq,
                    "part_freq": partial_freq,
                    "sweep_site": sweep_site,
                    "sweep_Ne": sweep_effective_size,
                    "adapt": adapt,
                    "left_rho": left_rho,
                    "rrh_left": rrh_left,
                    "rrh_loc": rrh_loc
                    }

        for key in sel_dict.keys():
            if sel_dict[key]:
                if "," in sel_dict[key]:
                    sel_dict[key] = list(map(float, sel_dict[key].split(",")))
                elif "." in sel_dict[key]:
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
                  "sel_dict": sel_dict
                  }

    return model_dict


def read_config_stats(configFile):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(configFile)

    # simulation section
    sim = "simulation"
    n_haps = config.getint(sim, "sample_size")
    pop_list = list(map(int, config.get(sim, "populations").split(",")))
    assert n_haps == sum(pop_list)
    popconfig = []
    ind = 0
    for pop in pop_list:
        popconfig.append(list(range(ind, pop+ind)))
        ind += pop
    length_bp = config.getfloat(sim, "length")
    reps = config.getint(sim, "loci")
    recombination_rate = config.getfloat(sim, "recombination_rate")
    per_fix_derived = config.getfloat(sim, "percent_fixed_derived")

    sim_dt = {}
    # stats 1 pop
    pop1 = "single_pop"
    win_size = config.getfloat(pop1, "win_size")
    sim_dt["pi"] = True
    sim_dt["tajd"] = True
    sim_dt["hap_het"] = config.getboolean(pop1, "compute_haplotype_het")
    sim_dt["afibs"] = config.getboolean(pop1, "compute_AFIBS")
    afibsfold = config.getboolean(pop1, "fold_AFIBS")
    sim_dt["ibs"] = config.getboolean(pop1, "compute_IBS")
    if sim_dt["ibs"]:
        prob_list = list(map(float, config.get(pop1, "prob_list").split(",")))
        assert len(prob_list) > 0
        size_list = list(map(int, config.get(pop1, "size_list").split(",")))
        assert len(size_list) > 0
        assert max(size_list) <= n_haps
    sim_dt["spatial_sfs"] = config.getboolean(pop1, "compute_spatial_SFS")
    spatfold = config.getboolean(pop1, "fold_spatial_SFS")
    sim_dt["sfs"] = config.getboolean(pop1, "compute_SFS")
    sfsfold = config.getboolean(pop1, "fold_SFS")
    sim_dt["ld"] = config.getboolean(pop1, "compute_LD")
    if sim_dt["ld"]:
        nb_times = config.getint(pop1, "pop_time_changes")
        assert nb_times > 0
        tmax = config.getfloat(pop1, "time_max_demo")
        assert tmax > 0
    # stats for 2 pops
    pop2 = "joint_pop"
    win_size_2 = config.getfloat(pop2, "win_size")
    sim_dt["jsfs"] = config.getboolean(pop2, "compute_JSFS")
    sim_dt["ld_2pop"] = config.getboolean(pop2, "compute_cross_pop_LD")
    quant_list = list(map(float, config.get(pop2, "quantile_list").split(",")))
    filet_pairs = list(config.get(pop2, "compute_pairs").split(","))
    for fs in filet_pairs:
        sim_dt[fs.strip()] = True
    # =========================================================================
    #  Build stat dictionary
    # =========================================================================
    stat_list = [k for k, j in sim_dt.items() if j]
    stats_dt = {"num_haps": n_haps,
                "pop_config": popconfig,
                "length_bp": int(length_bp),
                "reps": reps,
                "recombination_rate": recombination_rate,
                "perfixder": per_fix_derived,
                "win_size1": int(win_size),
                "win_size2": int(win_size_2),
                "calc_stats": stat_list,
                "afibs_fold": afibsfold,
                "spat_fold": spatfold,
                "sfs_fold": sfsfold,
                "pw_quants": quant_list
                }
    if "ibs" in stat_list:
        stats_dt["ibs_params"] = (prob_list, size_list)
    if "ld" in stat_list:
        intervals_list = ld_intervals(nb_times, tmax, recombination_rate, length_bp)
        stats_dt["ld_params"] = intervals_list
    return stats_dt
