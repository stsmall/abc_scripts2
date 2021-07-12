# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:05:00 2020
@author: Scott T. Small

"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, epps_singleton_2samp
from scipy.special import rel_entr, kl_div
from scipy.spatial.distance import jensenshannon
# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# pdf edits
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
colours = sns.color_palette("colorblind")


def calc_ks(df, pop, stat, col2, epps=False):
    """Compare two distributions.
    
    ks_1samp;kstest(x, "norm")
    anderson_ksample(x)  # mixed distribution
    epps_singleon_2samp(x, y)  # discrete

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    pop : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    epps : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    dfpop = df[df["pops"] == pop]
    stats_list = dfpop[stat].unique()
    sig_count = 0
    for i in stats_list:
        obs = dfpop[(dfpop[stat] == i) & (dfpop["df_id"] == "obs")]
        sim = dfpop[(dfpop[stat] == i) & (dfpop["df_id"] == "sim")]
        if epps:
            ts, pval = epps_singleton_2samp(obs[col2], sim[col2])
        else:
            ts, pval = ks_2samp(obs[col2], sim[col2])
        if pval < 0.001:
            print(f"{stat} {i}: {pval} **")
            sig_count += 1
        else:
            print(f"{stat} {i}: {pval}")
    print(f"total perc sig: {sig_count/len(stats_list)}")


def calc_kl(df, pop, stat, col2):
    """Compare two prob distributions.
    
    https://machinelearningmastery.com/divergence-between-probability-distributions/

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    pop : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dfpop = df[df["pops"] == pop]
    stats_list = dfpop[stat].unique()
    for i in stats_list:
        obs = dfpop[(dfpop[stat] == i) & (dfpop["df_id"] == "obs")]
        sim = dfpop[(dfpop[stat] == i) & (dfpop["df_id"] == "sim")]
        ent = sum(rel_entr(obs[col2].values, sim[col2].values))
        kl = sum(kl_div(obs[col2].values, sim[col2].values))
        js = jensenshannon(obs[col2].values, sim[col2].values, base=2)
        print(f"{stat} {i}: rel_entr {ent}, KL_div {kl}, js_dist {js} bits")


def box_pop(df, stat, col2="distance"):
    """Box plot for one data frame.

    spatialsfs{n}_pop{k}
    sfs{n}_pop{k}
    afibs{n}_pop{k}
    afibsstd{n}_pop{k}
    jsfs{n}_pop{kj}
    # ld{n}_pop{k} :: load these distance from ldintervals.txt

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE, optional
        DESCRIPTION. The default is "distance".

    Returns
    -------
    None.

    """
    stat_cols = [col for col in df.columns if stat == col.split("_")[0].rstrip('0123456789')]
    df_stat = pd.melt(df.filter(regex=f"{stat}"), value_vars=stat_cols, var_name=stat, value_name=col2)
    # add pop column
    pop = df_stat[stat].str.split("_", n=-1, expand=True)
    df_stat["pops"] = pop[pop.columns[-1]]
    df_stat[stat] = pop[0].str.split(f"{stat}", n=1, expand=True)[1]
    # plotting if just obs or just sims
    ax = sns.catplot(data=df_stat, x=stat, y=col2, col="pops", kind="box", hue="pops",
                     fliersize=.5, height=4, aspect=2, sharex=False)
    for axes in ax.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), fontsize=7)
    ax.savefig(f"{stat}.catplot.pdf")


def box_compare(df_list, stat, col2="distance", title=None):
    """Compare two dataframes with boxplots.
    
    spatialsfs{n}_pop{k}
    sfs{n}_pop{k}
    afibs{n}_pop{k}
    afibsstd{n}_pop{k}
    jsfs{n}_pop{kj}
    # ld{n}_pop{k} :: load these distance from ldintervals.txt
    
    Parameters
    ----------
    df_list : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE, optional
        DESCRIPTION. The default is "distance".
    title : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df_stat_combine : TYPE
        DESCRIPTION.

    """
    df_stat_ls = []
    df_id = ["obs", "sim"]
    for i, df in enumerate(df_list):
        stat_cols = [col for col in df.columns if stat == col.split("_")[0].rstrip('0123456789')]
        df_stat = pd.melt(df.filter(regex=f"{stat}"), value_vars=stat_cols, var_name=stat, value_name=col2)
        # add pop column
        pop = df_stat[stat].str.split("_", n=-1, expand=True)
        df_stat["pops"] = pop[pop.columns[-1]]
        df_stat[stat] = pop[0].str.split(f"{stat}", n=1, expand=True)[1]
        df_stat["df_id"] = df_id[i]
        df_stat_ls.append(df_stat)
    # combine 2 dfs
    df_stat_combine = pd.concat(df_stat_ls).reset_index(drop=True)
    # plotting if comparing
    if title is None:
        title = stat
    for pop in df_stat_combine["pops"].unique():
        fig, ax = plt.subplots(1, figsize=(14, 4))
        ax = sns.boxplot(data=df_stat_combine.query(f"pops=='{pop}'"), x=stat, y=col2, hue="df_id", fliersize=.5)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        ax.set_title(f"{pop}")
        fig.savefig(f"{stat}.{pop}.compare.boxplot.pdf", bbox_inches='tight')
    return df_stat_combine


def hist_pop(df, stat, col2, cum=False, title=None):
    """Histograms for stats from one dataframe.
    
    pi_pop{k}
    pistd_pop{k}
    tajd_pop{k}
    tajdstd_pop{k}
    haphet_pop{k}
    haphetstd_pop{k}

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    cum : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    stat_cols = [col for col in df.columns if stat == col.split("_")[0]]
    df_stat = pd.melt(df.filter(regex=f"{stat}_"), value_vars=stat_cols, var_name=stat, value_name=col2)
    # add pop column
    pop = df_stat[stat].str.split("_", n=-1, expand=True)
    df_stat["pops"] = pop[pop.columns[-1]]
    # plotting if just obs or just sims
    if title is None:
        title = stat
    name = ''
    if cum:
        ax = sns.histplot(data=df_stat, x=col2, hue="pops", element="step", fill=False, cumulative="True", stat="density", common_norm="False")
        name = "cum"
    else:
        ax = sns.histplot(data=df_stat, x=col2, hue="pops", multiple="dodge", shrink=.8)
    ax.set_title(f"{title}")
    ax.figure.savefig(f"{stat}.{name}hist.pdf", bbox_inches='tight')


def hist_compare(df_list, stat, col2, cum=False, title=None):
    """Compare two dataframes with histograms.
    
    pi_pop{k}
    pistd_pop{k}
    tajd_pop{k}
    tajdstd_pop{k}
    haphet_pop{k}
    haphetstd_pop{k}

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    cum : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    df_stat_ls = []
    df_id = ["obs", "sim"]
    for i, df in enumerate(df_list):
        stat_cols = [col for col in df.columns if stat == col.split("_")[0]]
        df_stat = pd.melt(df.filter(regex=f"{stat}_"), value_vars=stat_cols, var_name=stat, value_name=col2)
        # add pop column
        pop = df_stat[stat].str.split("_", n=-1, expand=True)
        df_stat["pops"] = pop[pop.columns[-1]]
        df_stat["df_id"] = df_id[i]
        df_stat_ls.append(df_stat)
    # combine 2 dfs
    df_stat_combine = pd.concat(df_stat_ls).reset_index(drop=True)
    # plotting if just obs or just sims
    if title is None:
        title = stat
    name = ""
    for pop in df_stat_combine["pops"].unique():
        fig, ax = plt.subplots(1, figsize=(14, 4))
        if cum:
            sns.histplot(data=df_stat_combine.query(f"pops=='{pop}'"), x=col2, hue="df_id", element="step", fill=False, cumulative="True", stat="density", common_norm=False)
            name = "cum"
        else:
            sns.histplot(data=df_stat_combine.query(f"pops=='{pop}'"), x=col2, hue="df_id", multiple="dodge", shrink=.8, stat="density", common_norm=False)
        ax.set_title(f"{pop}")
        ax.figure.savefig(f"{stat}.{pop}.{name}hist.pdf", bbox_inches='tight')
    return df_stat_combine


def hist_pair(df, stat, col2, cum=False, title=None):
    """Histogram of population pair stats.
    
    dXY{n}_pop{kj}
    dmin{n}_pop{kj}
    FST{n}_pop{kj}
    gmin{k}_pop{kj}
    Zx{n}_pop{kj}
    IBSmaxXY_pop{kj}

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    cum : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    stat_cols = [col for col in df.columns if stat == col.split("_")[0].rstrip('0123456789')]
    df_stat = pd.melt(df.filter(regex=f"{stat}"), value_vars=stat_cols, var_name=stat, value_name=col2)
    # add pop column
    pop = df_stat[stat].str.split("_", n=-1, expand=True)
    df_stat["pops"] = pop[pop.columns[-1]]
    df_stat[stat] = pop[pop.columns[0]].str.rstrip('0123456789')
    # plotting if just obs or just sims
    if title is None:
        title = stat
    name = ""
    if cum:
        g = sns.histplot(data=df_stat, x=col2, hue="pops", element="step", fill=False, cumulative="True", stat="density", common_norm="False")
        name = "cum"
    else:
        g = sns.histplot(data=df_stat, x=col2, hue="pops")
    g.set_title(f"{title}")
    g.figure.savefig(f"{stat}.{name}histpair.pdf", bbox_inches='tight')


def hist_pair_compare(df_list, stat, col2, cum=False, title=None):
    """Compare two dataframes with population pair statistics.
    
    dXY{n}_pop{kj}
    dmin{n}_pop{kj}
    FST{n}_pop{kj}
    gmin{k}_pop{kj}
    Zx{n}_pop{kj}
    IBSmaxXY_pop{kj}

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    cum : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    df_stat_ls = []
    df_id = ["obs", "sim"]
    for i, df in enumerate(df_list):
        stat_cols = [col for col in df.columns if stat == col.split("_")[0].rstrip('0123456789')]
        df_stat = pd.melt(df.filter(regex=f"{stat}"), value_vars=stat_cols, var_name=stat, value_name=col2)
        # add pop column
        pop = df_stat[stat].str.split("_", n=-1, expand=True)
        df_stat["pops"] = pop[pop.columns[-1]]
        df_stat[stat] = pop[pop.columns[0]].str.rstrip('0123456789')
        df_stat["df_id"] = df_id[i]
        df_stat_ls.append(df_stat)
    # combine 2 dfs
    df_stat_combine = pd.concat(df_stat_ls).reset_index(drop=True)
    # plotting if just obs or just sims
    if title is None:
        title = stat
    name = ''
    for pop in df_stat_combine["pops"].unique():
        fig, ax = plt.subplots(1, figsize=(14, 4))
        if cum:
            g=sns.histplot(data=df_stat_combine.query(f"pops=='{pop}'"), x=col2, hue="df_id", element="step", fill=False, cumulative="True", stat="density", common_norm="False")
            name = "cum"
        else:
            g=sns.histplot(data=df_stat_combine.query(f"pops=='{pop}'"), x=col2, hue="df_id", stat="density", common_norm=False)
        g.set_title(f"{title}")
        g.figure.savefig(f"{stat}.{pop}.{name}histpair.pdf", bbox_inches='tight')
    return df_stat_combine


def hist_pair2(df, stat, col2, cum=False, title=None):
    """Histogram of population pair stats.

    dd12_{n}_{k}_pop{kj}
    ddRank12_{n}_{k}_pop{kj}

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    cum : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    stat_cols = [col for col in df.columns if stat == col.split("_")[0]]
    df_stat = pd.melt(df.filter(regex=f"{stat}"), value_vars=stat_cols, var_name=stat, value_name=col2)
    # add pop column
    pop = df_stat[stat].str.split("_", n=-1, expand=True)
    df_stat["pops"] = pop[pop.columns[-1]]
    df_stat["subpop"] = pop[2]
    df_stat[stat] = pop[0]
    # plotting if just obs or just sims
    if title is None:
        title = stat
    name = ""
    if cum:
        g = sns.histplot(data=df_stat, x=col2, hue="subpop", col="pops", kind="ecdf")
        name = "cum"
    else:
        g = sns.displot(data=df_stat, x=col2, hue="subpop", col="pops", kind="hist")
    g.savefig(f"{stat}.{name}histpair.pdf", bbox_inches='tight')


def hist_pair2_compare(df_list, stat, col2, cum=False, title=None):
    """Compare two dataframes with population pair statistics.
    
    dd12_{n}_{k}_pop{kj}
    ddRank12_{n}_{k}_pop{kj}

    Parameters
    ----------
    df_list : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    cum : TYPE, optional
        DESCRIPTION. The default is False.
    title : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df_stat_combine : TYPE
        DESCRIPTION.

    """
    df_stat_ls = []
    df_id = ["obs", "sim"]
    for i, df in enumerate(df_list):
        stat_cols = [col for col in df.columns if stat == col.split("_")[0]]
        df_stat = pd.melt(df.filter(regex=f"{stat}"), value_vars=stat_cols, var_name=stat, value_name=col2)
        # add pop column
        pop = df_stat[stat].str.split("_", n=-1, expand=True)
        df_stat["pops"] = pop[pop.columns[-1]]
        df_stat["subpop"] = pop[2]
        df_stat[stat] = pop[0]
        df_stat["df_id"] = df_id[i]
        df_stat_ls.append(df_stat)
    # combine 2 dfs
    df_stat_combine = pd.concat(df_stat_ls).reset_index(drop=True)
    # plotting if just obs or just sims
    if title is None:
        title = stat
    name = ""
    for pop in df_stat_combine["pops"].unique():
        if cum:
            g = sns.displot(data=df_stat_combine.query(f"pops=='{pop}'"), x=col2, hue="df_id", col="subpop", kind="ecdf")
            name = "cum"
        else:
            g = sns.displot(data=df_stat_combine.query(f"pops=='{pop}'"), x=col2, hue="df_id", col="subpop", kind="hist")
        g.savefig(f"{stat}.{pop}.{name}histpair.pdf", bbox_inches='tight')
    return df_stat_combine


def ibs_pop(df, stat, col2, probs, haps, num_pops=2, sample=20):
    """Calculate IBS stats for one dataframe.

    ibs{n}_pop{k}
    prob_list = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    size_list = [16, 8, 4, 2]

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    probs : TYPE
        DESCRIPTION.
    haps : TYPE
        DESCRIPTION.
    num_pops : TYPE, optional
        DESCRIPTION. The default is 2.
    sample : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    None.

    """
    prob_list = probs  # [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    size_list = haps  # [16, 8, 4, 2]
    iterations = df.shape[0]
    stat_cols = [col for col in df.columns if col.startswith(f"{stat}")]
    fig, axes = plt.subplots(len(size_list), num_pops, figsize=(12, 8))
    rand_its = np.random.choice(range(0, iterations), sample, replace=False)
    for its in rand_its:
        dff = df.iloc[its:its+1]
        df_stat = pd.melt(dff.filter(regex=f"{stat}"), value_vars=stat_cols, var_name=stat, value_name=col2)
        # add pop column
        pop = df_stat[stat].str.split("_", n=-1, expand=True)
        df_stat["pops"] = pop[pop.columns[-1]]
        df_stat["prob"] = (prob_list * len(size_list)) * len(df_stat["pops"].unique())
        df_stat["hap"] = list(np.repeat(size_list, len(prob_list))) * len(df_stat["pops"].unique())
        df_stat["hap"] = df_stat["hap"].astype('category')
        # lineplot
        for i, h in enumerate(size_list[::-1]):
            for j, p in enumerate(df_stat["pops"].unique()):
                g=sns.lineplot(data=df_stat[(df_stat["pops"] == p) & (df_stat["hap"] == h)],
                               hue="hap", x=col2, y="prob",markers=True, dashes=False, ax=axes[i, j])
                g.legend([],[], frameon=False)
    fig.savefig("ibs.pdf", bbox_inches='tight')


def ibs_compare(df_list, stat, col2, probs, haps, num_pops, sample=20):
    """Compare IBS from two dataframes.
    
    ibs{n}_pop{k}
    prob_list = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    size_list = [16, 8, 4, 2]

    Parameters
    ----------
    df_list : TYPE
        DESCRIPTION.
    stat : TYPE
        DESCRIPTION.
    col2 : TYPE
        DESCRIPTION.
    probs : TYPE
        DESCRIPTION.
    haps : TYPE
        DESCRIPTION.
    num_pops : TYPE
        DESCRIPTION.
    sample : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    df_tots : TYPE
        DESCRIPTION.

    """
    df_id = ["obs", "sim"]
    df_tots = pd.DataFrame()
    iterations = min([df.shape[0] for df in df_list])
    rand_its = np.random.choice(range(0, iterations), sample, replace=False)
    # set plot size, subplot
    fig, axes = plt.subplots(len(haps), num_pops, figsize=(12, 8))
    for its in rand_its:
        df_stat_ls = []
        for i, df in enumerate(df_list):
            dff = df.iloc[its:its+1]
            stat_cols = [col for col in df.columns if col.startswith(f"{stat}")]
            df_stat = pd.melt(dff.filter(regex=f"{stat}"), value_vars=stat_cols, var_name=stat, value_name=col2)
            # add pop column
            pop = df_stat[stat].str.split("_", n=-1, expand=True)
            df_stat["pops"] = pop[pop.columns[-1]]
            df_stat["prob"] = (probs * len(haps)) * len(df_stat["pops"].unique())
            df_stat["hap"] = list(np.repeat(haps, len(probs))) * len(df_stat["pops"].unique())
            df_stat["hap"] = df_stat["hap"].astype('category')
            df_stat["df_id"] = df_id[i]
            df_stat_ls.append(df_stat)
        df_stat_combine = pd.concat(df_stat_ls).reset_index(drop=True)
        df_tots = pd.concat([df_tots, df_stat_combine]).reset_index(drop=True)
        #calc_ks(dff, "pop0", "hap", col2)
        # lineplot
        for i, h in enumerate(haps[::-1]):
            for j, p in enumerate(df_stat_combine["pops"].unique()):
                g = sns.lineplot(data=df_stat_combine[(df_stat_combine["pops"] == p) & (df_stat_combine["hap"] == h)], x=col2, y="prob",
                               hue="df_id", style="hap", markers=True, ax=axes[i, j])
                g.legend([],[], frameon=False)
    fig.savefig("ibs.compare.pdf", bbox_inches='tight')

    return df_tots


def parse_args(args_in):
    """Parse args."""
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, required=True,
                        help="path to file with data")
    parser.add_argument("--data_obs", type=str, default=None,
                        help="path to file when using compare")
    parser.add_argument("--stat", type=str, required=True,
                        choices=("spatialsfs", "sfs", "afibs", "afibsstd", "jsfs",
                                 "ld", "pi", "pistd", "tajd", "tajdstd", "haphet",
                                 "haphetstd", "dXY", "dmin", "FST", "gmin", "Zx",
                                 "IBSmaxXY", "dd12", "ddRank12", "ibs"),
                        help="which stat to plot")
    parser.add_argument("--compare", action="store_true",
                        help="compare data_obs to data")
    parser.add_argument("--probs", type=float, nargs="*", action="append",
                        default=[0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999],
                        help="list of probabilities for calculating IBS")
    parser.add_argument("--haps", type=int, nargs="*", action="append",
                        default=[16, 8, 4, 2],
                        help="list of haplotype combinations when calculating IBS")
    parser.add_argument("--pops", type=int, default=2,
                        help="number of pops in data")
    parser.add_argument("--sep", type=str, default="\t")
    return(parser.parse_args(args_in))


def main():
    """Run main function."""
    args = parse_args(sys.argv[1:])
    # =========================================================================
    #  Gather args
    # =========================================================================
    df_path = args.data
    stat = args.stat
    df_obs_path = args.data_obs
    num_pops = args.pops
    prob_list = args.probs  # [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    size_list = args.haps  # [16, 8, 4, 2]
    sep = args.sep
    # =========================================================================
    #  Main executions
    # =========================================================================
    if args.compare:
        assert df_obs_path is not None

    df = pd.read_csv(df_path, sep=sep)
    if stat in ["spatialsfs", "sfs", "afibs", "afibsstd", "jsfs", "ld"]:
        if args.compare:
            dff = box_compare([df_obs_path, df], stat)
        else:
            box_pop(df, stat)
    elif stat in ["pi", "pistd", "tajd", "tajdstd", "haphet", "haphetstd"]:
        if args.compare:
            dff = hist_compare([df_obs_path, df], stat, f"{stat}_val")
        else:
            hist_pop(df, stat, f"{stat}_val")

    elif stat in ["dXY", "dmin", "FST", "gmin", "Zx", "IBSmaxXY"]:
        if args.compare:
            dff = hist_pair_compare([df_obs_path, df], stat, f"{stat}_val")
        else:
            hist_pair(df, stat, f"{stat}_val")

    elif stat in ["dd12", "ddRank12"]:
        if args.compare:
            dff = hist_pair2_compare([df_obs_path, df], stat, f"{stat}_val")
        else:
            hist_pair2(df, stat, f"{stat}_val")

    elif stat == "ibs":
        if args.compare:
            dff = ibs_compare([df_obs_path, df], stat, "distance", prob_list, size_list, num_pops)
            for p in dff["pops"].unique():
                for h in size_list[::-1]:
                    df_hap = dff[dff["hap"] == h]
                    print(f"\n pop {p} and number haps: {h}\n")
                    calc_ks(df_hap, p, "prob", "distance")
        else:
            ibs_pop(df, stat, "distance", prob_list, size_list)

if __name__ == "__main__":
    main()
