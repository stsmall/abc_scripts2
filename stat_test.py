# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:10:49 2021
@author: Scott T. Small

"""
import allel
import msprime as msp
import numpy as np
from project.stat_modules.sequtils import read_trees
from project.stat_modules.sequtils import read_ms, add_seqerror
from project.stat_modules import afibs
from project.stat_modules import afs
from project.stat_modules import ibs
from project.stat_modules import ld
from project.stat_modules import popstats
from project.stat_modules import pwpopstats


# option 1: run msprime w/ set seed
pops = [msp.PopulationConfiguration(sample_size=10, initial_size=10000),
        msp.PopulationConfiguration(sample_size=10, initial_size=10000)]
demo_events = [msp.MassMigration(time=10000, source=0, destination=1, proportion=1.0)]
ts_reps = list(msp.simulate(population_configurations=pops,
                            Ne=10000,
                            recombination_rate=1e-8,
                            mutation_rate=1e-8,
                            length=100000,
                            num_replicates=1,
                            demographic_events=demo_events,
                            random_seed=1000))

# option 2
#ms_dict = read_ms("tests/msout/test.msout", "msmove", 20, 10000)
#pos, haps, counts = add_seqerror(ms_dict[pos], ms_dict[haps], 100000, 0.0, seq_error=False)
# =========================================================================
#  Globals
# =========================================================================
pos, haps, counts, bp = read_trees(ts_reps[0], 100000, 0.0, seq_error=False)

pop_config = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
pop1 = pop_config[0]
gt = allel.HaplotypeArray(haps.T)
pos_s = allel.SortedIndex(pos)
gtpop = gt.take(pop1, axis=1)
win_size = 20000
length_bp = 100000
quants = [0, 0.25, 0.50, 0.75, 1.0]


def split_pop(pop_ix):
    hap_p = haps[pop_ix, :]
    seg = hap_p.sum(axis=0).astype(int)
    seg_mask = seg > 0
    pos_p = pos[seg_mask]
    hap_p = hap_p[:, seg_mask]
    counts_p = hap_p.sum(axis=0).astype(int)
    return hap_p, pos_p, counts_p


hap_p, pos_p, counts_p = split_pop(pop1)

# =========================================================================
# Tests
# =========================================================================
def test_seqerror():
    """Use msprime output."""
    pos_se, h, counts_se, b = read_trees(ts_reps[0], 100000, 0.0, seq_error=True)
    se = set(pos_se) - set(pos)  # errors
    cse = counts_se[np.isin(pos_se, list(se))]  # should all be 1
    assert(all(cse == 1) is True)


def test_seqerror2():
    """Use read_ms output."""
    pos_se, h, counts_se = add_seqerror(pos, haps, 100000, 0.0, seq_error=True)
    se = set(pos) - set(pos_se)  # errors
    cse = counts_se[np.isin(pos_se, list(se))]  # should all be 1
    assert(all(cse == 1) is True)


def test_afibs():
    stats_ls = []
    durbin = True
    res_afibs = afibs.distrib_afibs(hap_p, pos_p, counts_p, durbin)
    afibs_m = np.round(res_afibs[::2])  # mean afibs
    afibs_std = np.round(res_afibs[1::2])  # afibs_std
    # unfold
    stats_ls.extend(afibs_m)  # mean afibs
    stats_ls.extend(afibs_std)  # afibs_std
    ss = np.array([12878., 5664., 10330., 6950., 2280., 4879., 2896., 2175.,
                   6411., 5724., 1834., 3938., 1589., 2948., 0., 2316.])
    assert(all(np.isclose(ss, stats_ls)) is True)
    # fold
    stats_ls = []
    for a in [afibs_m, afibs_std]:
        afibs_flip = np.flip(a, axis=0)
        afibs_fold = (afibs_flip + a)
        haps = int(len(pop1)/2)
        stats_ls.extend(afibs_fold[0:haps])
    ss = np.array([15053.0, 8560.0, 15209.0, 9230.0, 9230.0, 8727.0, 5724.0,
                   4782.0, 5527.0, 5527.0])
    assert(all(np.isclose(ss, stats_ls)) is True)


def test_sfs():
    # unfold
    sfs = afs.asfs_stats(gtpop, pos_s, False, False)
    ss = np.array([0.43859649, 0.16666667, 0.14912281, 0.01754386, 0.05263158,
                   0.04385965, 0.0877193, 0.01754386, 0.02631579])
    assert(all(np.isclose(ss, sfs)) is True)
    # fold
    sfs = afs.asfs_stats(gtpop, pos_s, True, False)
    ss = np.array([0.46491228, 0.18421053, 0.23684211, 0.06140351, 0.05263158])
    assert(all(np.isclose(ss, sfs)) is True)
    # agg
    sfs = afs.asfs_stats(gtpop, pos_s, False, True)
    ss = np.array([0.43859649, 0.16666667, 0.14912281, 0.20175439, 0.01754386,
                   0.02631579])
    assert(all(np.isclose(ss, sfs)) is True)


def test_ibs():
    size = [8, 4, 2]
    prob = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    stats_ls = []
    for m in size:
        hap_pp = hap_p[0:m, :]
        res_ibs = ibs.ibs_quantiles_from_data(m, pos_p, hap_pp, prob, length_bp)
        stats_ls.extend(np.round(res_ibs))
    ss = np.array([14.0, 14.0, 18.0, 78.0, 243.0, 606.0, 1202.0, 1716.0, 3660.0,
                   4941.0, 5080.0, 14.0, 14.0, 18.0, 78.0, 243.0, 606.0, 1202.0,
                   1716.0, 3660.0, 4941.0, 5080.0, 14.0, 14.0, 18.0, 78.0, 243.0,
                   606.0, 1202.0, 1716.0, 3660.0, 4941.0, 5080.0])
    assert(all(np.isclose(ss, np.array(stats_ls))) is True)


def test_pi():
    pi_, pi_std = popstats.pi_window(pos_s, gtpop, win_size, length_bp)
    assert(np.round(pi_, 5) == 0.00038)
    assert(np.round(pi_std, 5) == 8e-05)


def test_tajd():
    tajd_, tajd_std = popstats.tajimaD(pos_s, gtpop, win_size, length_bp)
    assert(np.round(tajd_, 5) == -0.32115)
    assert(np.round(tajd_std, 5) == 0.49679)


def test_hap_het():
    res_hh = popstats.haplo_win(pos_p, hap_p, win_size, length_bp)
    res = np.round(res_hh, 5)
    ss = np.array([0.94444, 0.03333])
    assert(all(np.isclose(ss, res)) is True)


def test_spatial_sfs():
    n_haplo = len(pop1)
    spat_ = afs.spatial_histo_fast(pos_p, counts_p, n_haplo-1)
    spat_ = np.round(spat_)
    # unfold
    ss = np.array([1419., 7868., 5648., 0., 12476., 32643., 10216., 0., 12462.])
    assert(all(np.isclose(ss, spat_)) is True)
    # fold
    spat_flip = np.flip(spat_, axis=0)
    spat_fold = (spat_flip + spat_)
    haps = int(len(pop1)/2)
    spat_fold = spat_fold[0:haps]
    spat_f = spat_fold
    ss = np.array([13881.,  7868., 15864., 32643., 24952.])
    assert(all(np.isclose(ss, spat_f)) is True)
    # agg
    spat_ag = list(spat_[0:3])
    spat_ag.append(np.mean(spat_[3:-2]))
    spat_ag.append(spat_[-2])
    spat_ag.append(spat_[-1])
    spat_a = np.array(spat_ag)
    ss = np.array([1419.0, 7868.0, 5648.0, 13833.75, 0.0, 12462.0])
    assert(all(np.isclose(ss, spat_a)) is True)


def test_ld():
    intervals = [[55269, 61087], [25367, 28038], [11887, 13138], [5623, 6215],
                 [2672, 2953], [1272, 1406], [617, 681]]
    pw_ld = ld.ld_pop_mp(pos_p, hap_p, intervals, 0.05, 1000)  # momentsLD 100 snps
    ss = np.array([0.00126, 0.00044, -0.00031, 0.00386, 0.00349, 0.0066, 0.00357])
    assert(all(np.isclose(ss, np.round(pw_ld, 5))) is True)


def test_jsfs():
    # fold
    props = afs.jsfs_stats(len(pop1), gt, pos_s, True)
    ss = np.array([0.29101, 0.3545 , 0., 0., 0.09524, 0.12169, 0.00529,
                0., 0., 0., 0., 0.02646, 0.01058, 0., 0.04762, 0.01058,
                0.01587, 0.00529, 0.00529, 0., 0., 0.01058, 0.])
    assert(all(np.isclose(ss, np.round(props, 5))) is True)
    # unfold
    props = afs.jsfs_stats(len(pop1), gt, pos_s, False)
    ss = np.array([0.25397, 0.32804, 0.10582, 0.1164, 0.00529, 0.01058,
                   0.00529, 0.02646, 0.00529, 0.00529, 0.01058, 0., 0.,
                   0.0582, 0., 0., 0., 0., 0.00529, 0., 0.02116, 0.03175,
                   0.01058])
    assert(all(np.isclose(ss, np.round(props, 5))) is True)


def test_ld_2pop():
    pw_ld = ld.ld_pop2(len(pop1), pos, haps, quants)
    ss = np.array([-0.025, -0.00833333, 0., 0. , 0.00833333])
    assert(all(np.isclose(ss, pw_ld)) is True)


def test_FST():
    flt = pwpopstats.fst(len(pop1), pos_s, gt, quants)
    ss = np.array([-0.11111111, -0.08695652, -0.02564103, 0.12698413, 0.49494949])
    assert(all(np.isclose(ss, flt)) is True)


def test_dXY():
    flt = pwpopstats.dxy(len(pop1), pos_s, gt, win_size, length_bp)
    stats_ls = np.quantile(flt, quants)
    ss = np.array([0.000405, 0.000412, 0.000423, 0.000531, 0.000621])
    assert(all(np.isclose(ss, stats_ls)) is True)


def test_dmin():
    flt = pwpopstats.dmin(len(pop1), pos_s, gt, win_size, length_bp)
    stats_ls = np.quantile(flt, quants)
    ss = np.array([5.0e-05, 1.5e-04, 2.0e-04, 2.0e-04, 2.5e-04])
    assert(all(np.isclose(ss, stats_ls)) is True)


def test_gmin():
    flt = pwpopstats.gmin(len(pop1), pos_s, gt, win_size, length_bp)
    stats_ls = np.quantile(flt, quants)
    ss = np.array([0.09416196, 0.35460993, 0.40257649, 0.48543689, 0.49382716])
    assert(all(np.isclose(ss, stats_ls)) is True)


def test_dd12():
    flt = pwpopstats.dd1_2(len(pop1), pos_s, gt, win_size, length_bp, quants)
    ss = np.array([0.09978, 0.47535, 0.53254, 0.60976, 0.70312, 0.1312 , 0.35809,
                0.73469, 0.74503, 0.82569])
    assert(all(np.isclose(ss, np.round(flt, 5))) is True)


def test_ddRank12():
    flt = pwpopstats.ddRank1_2(len(pop1), pos_s, gt, win_size, length_bp, quants)
    ss = np.array([12.78, 19.44, 20., 26.11, 37.22, 15., 23.89, 23.89, 27.22, 32.78])
    assert(all(np.isclose(ss, np.round(flt, 2))) is True)


def test_Zx():
    flt = pwpopstats.zx(len(pop1), pos, haps, win_size, length_bp)
    stats_ls = np.nanquantile(flt, quants)
    ss = np.array([0. , 0.21724138, 0.65702271, 1.21374723, 1.78181818])
    assert(all(np.isclose(ss, stats_ls)) is True)


def test_IBS_maxXY():
    flt = pwpopstats.ibs_maxxy(len(pop1), pos, haps, length_bp)
    assert(flt == 13443.0)


if __name__ == "__main__":
    test_seqerror()
    test_afibs()
    test_sfs()
    test_ibs()
    test_pi()
    test_tajd()
    test_hap_het()
    test_spatial_sfs()
    test_ld()
    test_jsfs()
    test_ld_2pop()
    test_FST()
    test_dXY()
    test_dmin()
    test_gmin()
    test_dd12()
    test_ddRank12()
    test_Zx()
    test_IBS_maxXY()
