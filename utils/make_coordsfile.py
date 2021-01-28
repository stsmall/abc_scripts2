# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:50:08 2021
@author: Scott T. Small

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

"""
import sys
import argparse
import numpy as np


def load_gff(chrom, stop, gff, gff_filt):
    """Open gff file and keep coordinate list of features."""
    feat_ls = ["gene", "exon", "CDS", "mRNA", "five_prime_UTR", "three_prime_UTR"]
    gff_ls = []
    e = 1
    intron_s = 0
    gs = 0
    with open(gff, 'r') as gf:
        for line in gf:
            if not line.startswith("#"):
                g_lin = line.split("\t")
                if g_lin[0] == chrom:
                    feat = g_lin[2]
                    if "intergenic" in gff_filt:
                        if feat == "gene":
                            if gs == 0:
                                gff_ls.append((gs, int(g_lin[3])))
                            else:
                                gff_ls.append((gs-1, int(g_lin[3])))
                            gs = int(g_lin[4])
                    elif feat in feat_ls:
                        s = int(g_lin[3])
                        e = int(g_lin[4])
                        if feat in gff_filt:
                            gff_ls.append((s-1, e))
                        if "intron" in gff_filt:
                            if "exon" in feat:
                                if intron_s == 0:
                                    intron_s = e
                                else:
                                    gff_ls.append((intron_s-1, s))
                                    intron_s = e
    # catch last window for intergenic and introns using chrom_len or stop
    return gff_ls


def load_mask(chrom, mask_file):
    """Open mask file and store positions in array."""
    mask_ls = []
    bed = False
    with open(mask_file, 'r') as mask:
        for line in mask:
            m_lin = line.split()
            if len(m_lin) > 2:
                bed = True
            if bed:
                if m_lin[1].isdigit():
                    if m_lin[0] == chrom:
                        pos1 = int(m_lin[1])
                        pos2 = int(m_lin[2])
                        if pos2 - pos1 == 1:
                            mask_ls.append(pos1)
                        else:
                            mask_ls.extend(list(range(pos1, pos2)))
            else:
                if m_lin[1].isdigit():
                    if m_lin[0] == chrom:
                        pos1 = int(m_lin[1])  # assuming this is 1 based
                        mask_ls.append(pos1-1)
    mask_arr = np.array(mask_ls)

    return mask_arr


def make_bed_gff(chrom, window, step, stop, gff_ls, mask_arr, mask_frac):
    """Make coordinates from a gff file"""
    f = open(f"{chrom}.{window}.{step}.gff.bed", 'w')
    f.write("chrom\tchromStart\tchromEnd\tsites\n")
    for feat in gff_ls:
        s, e = feat
        if s > e:  # reverse strand
            e, s = feat
        feat_size = e - s
        if feat_size > window:
            ew = s + window
            sw = s
            while ew <= feat_size:
                if mask_arr is not None:
                    n_mask = np.count_nonzero((mask_arr >= sw) & (mask_arr <= ew))
                    w_len = ew - sw
                    n_frac = n_mask/w_len
                    if n_frac < mask_frac:
                        sites = w_len - n_mask
                        f.write(f"{chrom}\t{sw}\t{ew}\t{sites}\n")
                else:
                    w_len = ew - sw
                    f.write(f"{chrom}\t{sw}\t{ew}\t{w_len}\n")
                sw += step
                ew += step
            if feat_size < ew:
                # last window
                # TODO: possible overstep on last window since I dont use stop_len
                ew = feat_size
                if mask_arr is not None:
                    n_mask = np.count_nonzero((mask_arr >= sw) & (mask_arr <= ew))
                    w_len = ew - (sw - 1)
                    n_frac = n_mask/w_len
                    if n_frac < mask_frac:
                        sites = w_len - n_mask
                        f.write(f"{chrom}\t{sw}\t{ew}\t{sites}\n")
                else:
                    w_len = ew - sw
                    f.write(f"{chrom}\t{sw}\t{ew}\t{w_len}\n")
        else:
            if mask_arr is not None:
                n_mask = np.count_nonzero((mask_arr >= s) & (mask_arr <= e))
                w_len = e - s
                n_frac = n_mask/w_len
                if n_frac < mask_frac:
                    sites = w_len - n_mask
                    f.write(f"{chrom}\t{s}\t{e}\t{sites}\n")
            else:
                w_len = e - s
                f.write(f"{chrom}\t{s}\t{e}\t{w_len}\n")

    f.close()


def make_bed(chrom, start, stop, window, step, mask_arr, mask_frac):
    f = open(f"{chrom}.{window}.{step}.bed", 'w')
    f.write("chrom\tchromStart\tchromEnd\tsites\n")
    s = start
    e = start + window
    while e <= stop:
        if mask_arr is not None:
            n_mask = np.count_nonzero((mask_arr >= s) & (mask_arr <= e))
            w_len = e - s
            n_frac = n_mask/w_len
            if n_frac < mask_frac:
                sites = w_len - n_mask
                f.write(f"{chrom}\t{s}\t{e}\t{sites}\n")
        else:
            w_len = e - s
            f.write(f"{chrom}\t{s}\t{e}\t{w_len}\n")
        s += step
        e += step
    if stop < e:
        # last window
        e = stop
        if mask_arr is not None:
            n_mask = np.count_nonzero((mask_arr >= s) & (mask_arr <= e))
            w_len = e - (s - 1)
            n_frac = n_mask/w_len
            if n_frac < mask_frac:
                sites = w_len - n_mask
                f.write(f"{chrom}\t{s}\t{e}\t{sites}\n")
        else:
            w_len = e - s
            f.write(f"{chrom}\t{s}\t{e}\t{w_len}\n")
    f.close()


def parse_args(args_in):
    """Parse args."""
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('chr_arm', type=str,
                        help="Exact name of the chromosome arm for which feature"
                        " vectors will be calculated")
    parser.add_argument('chr_len', type=int,
                        help="Length of the chromosome arm")
    parser.add_argument('--window', type=float, default=1e6,
                        help="window size")
    parser.add_argument('--step', type=int, default=None,
                        help="step_size")
    parser.add_argument('--start', type=int, default=0,
                        help="start pos, if other than 1")
    parser.add_argument('--stop', type=int, default=None,
                        help="stop position, if other than chr_len")
    parser.add_argument('--mask', type=str, default=None,
                        help="Path to a mask file, can be bed or single list")
    parser.add_argument('--masked_frac', type=float, default=0.25,
                        help="cut-off for skipping window based on proportion of"
                        " masked sites")
    parser.add_argument('--gff', type=str, default=None,
                        help="Path to a gff file, for selecting coding/noncoding "
                        "regions")
    parser.add_argument('--gff_filter', default=None, nargs='+',
                        help="remove sites matching this keyword")
    return(parser.parse_args(args_in))


def main():
    """Run main function."""
    args = parse_args(sys.argv[1:])
    # =========================================================================
    #  Gather args
    # =========================================================================
    chrom = args.chr_arm
    chr_len = args.chr_len
    start = args.start
    stop = args.stop
    if stop is None:
        stop = chr_len
    window = int(args.window)
    step = args.step
    if step is None:
        step = window
    mask_file = args.mask
    mask_frac = args.masked_frac
    gff = args.gff
    gff_filter = args.gff_filter
    # =========================================================================
    #  Main executions
    # =========================================================================
    # load mask
    if mask_file:
        mask_arr = load_mask(chrom, mask_file)
    else:
        mask_arr = None

    if gff:
        gff_ls = load_gff(chrom, stop, gff, gff_filter)
        make_bed_gff(chrom, window, step, stop, gff_ls, mask_arr, mask_frac)
    else:
        make_bed(chrom, start, stop, window, step, mask_arr, mask_frac)


if __name__ == "__main__":
    main()
