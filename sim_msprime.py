#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:32:28 2020
@author: Scott T. Small

"""
import sys
import argparse
import msprime as msp
import tskit as tsk

def sim_syntax(ms_path, model_dict):
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
        if "discoal" in ms_path:
            gen_cov = f"-gr {gen_conversion*rho} {tract}"
        else:
            gen_cov = f"-c {gen_conversion*rho} {tract}"
    else:
        gen_cov = ""

    # subops
    init_sizes = [size * (ploidy/4.0) for size in model_dict["initialSize"]]
    grow_rate = model_dict["growthRate"]
    mig_mat = model_dict["migMat"]

    if "discoal" in ms_path:
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
    else:
        subpops = f"-I {npops} {' '.join(map(str, sample_sizes))}"
        ne_sub_pops = [f"-n {i+1} {pop_ne/scaled_Ne}" for i, pop_ne in enumerate(init_sizes)]
        ne_subpop = " ".join(ne_sub_pops)
        if sum(grow_rate) == 0:
            grow_subpop = ""
        else:
            grow_sub_pops = [f"-g {i+1} {g}" for i, g in enumerate(grow_rate)]
            grow_subpop = " ".join(grow_sub_pops)
        if mig_mat:
            mig_matrix = f"-ma {' '.join(map(str, mig_mat))}"
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


def simulate_discoal(model_dict, demo_dict, par_dict, eventkey_dict, cond_list, ms_path):
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
    ms_dict = sim_syntax(ms_path, model_dict)

    # build selection command line
    if model_dict["sel_dict"]:
        sel_list = selection_parse(model_dict, ms_dict)
    else:
        sel_list = ''

    # draw and organize for parameters and check conditions
    params_dict, time_dict = org_params(par_dict, demo_dict, eventkey_dict, cond_list)

    # build demographic command line
    model = Model()
    dem_events = model.genDem(ms_dict, model_dict, time_dict, ms_path)

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

    if "discoal" in ms_path:
        ms_base = ("{ms} {nhaps} {loci} {basepairs} -t {theta} -r {rho} "
                   "{gen_cov} {subpops} {ne_subpop} {demo} {sel}")
    else:
        ms_base = ("{ms} {nhaps} {loci} -t {theta} -r {rho} {basepairs} "
                   "{gen_cov} {subpops} {ne_subpop} {growth_subpop} {migmat} {demo}")
    # ms/msmove/discoal command line
    mscmd = ms_base.format(**ms_params)
    ms_cmd = " ".join(mscmd.split())
    return ms_cmd, params_dict

def main():
    with open(sim_path, 'w') as sims_outfile:
        for i in trange(sim_number):
            mscmd, params_dict = simulate(model_dict,
                                          demo_dict.copy(),
                                          par_dict,
                                          eventkey_dict,
                                          conditions,
                                          ms_path)
            priors_list = write_priors([], params_dict)
            priors_outfile.write(f"{priors_list}\n")
            sims_outfile.write(f"{mscmd} >> {outfile}\n")