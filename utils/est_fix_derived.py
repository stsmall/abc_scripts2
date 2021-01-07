#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:48:08 2020
@author: Scott T. Small

Useful only for when we add errors to the sequence:
Get an appproximation of the percentage of fixed derived sites because they are
 not present simulated data)

python est_fix_derived.py 332 5e6 5e5 2.8E-9
Percent fixed-derived: 0.01163252945261868
"""
import sys

n_haplo = int(sys.argv[1])
t_ancestor = float(sys.argv[2])
longtermNe = float(sys.argv[3])
mu = float(sys.argv[4])

delta = (t_ancestor + 2 * longtermNe - (2 * (1 - 1 / n_haplo) * 2 * longtermNe))
sampleTreeTot = (sum([1.0/i for i in range(1, n_haplo)]) * 2 * 2 * longtermNe)
pr_fix_der_given_fix = (delta * mu / (1 - sampleTreeTot * mu))
print(f"Percent fixed-derived: {pr_fix_der_given_fix}")
