#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:53:16 2020
@author: Scott T. Small
"""
import numpy as np
from math import log
import os

class DrawDist(object):
    """draws values from distribution."""

    def __init__(self):
        """Create command line for coalescent sims.

        Returns
        -------
        None.

        """
        return(None)

    def unif(self, low, high, size):
        """Draws a random value from a uniform distribution, float.

        Parameters
        ----------
        low: float
            lower bound
        high: float
            upper bound

        """
        return np.random.uniform(low, high, size)

    def log_unif(self, low, high, size, base=10):
        """Draws a random value from a log uniform distribution, float.

        Parameters
        ----------
        low: float
            lower bound
        high: float
            upper bound

        """
        log_low = log(low, base)
        log_high = log(high, base)
        rn = np.power(base, np.random.uniform(log_low, log_high, size))
        return rn

    def unif_int(self, low, high, size, trunc=-2):
        """Draws a random value from a uniform distribution, int.

        Parameters
        ----------
        low: int
            lower bound
        high: int
            upper bound

        """
        if type(low) is int or type(low) is float:
            trunc = np.floor((len(str(round(low))) - 1) / 2) * -1
        else:
            trunc = np.floor((len(str(round(low[0]))) - 1) / 2) * -1
        nn = np.random.randint(low, np.array(high)+1, size)
        return np.around(nn, int(trunc))

    def log_unif_int(self, low, high, size, base=10):
        """Draws a random value from a log uniform distribution, int.

        Parameters
        ----------
        low: float
            lower bound
        high: float
            upper bound

        """
        log_low = log(low, base)
        log_high = log(high, base)
        rn = np.power(base, np.random.uniform(log_low, log_high, size))
        return np.rint(rn)

    def norm_int(self, mu, sigma, size):
        """Draws a random value from a normal distribution, int.

        Parameters
        ----------
        mu: int
            mean
        sigma: int
            variance

        """
        nn = np.abs(np.round(sigma*np.random.randn(size)+mu))
        return nn

    def log_norm(self, mu, sigma, size):
        """Draws a random value from a log normal distribution, float.

        migration rates as 4Nm

        Parameters
        ----------
        mu: float
            log mean
        sigma: float
            variance

        """
        return sigma*np.random.lognormal(size=size)+mu

    def log_norm_int(self, mu, sigma, size):
        """Draws a random value from a log normal distribution, int.

        Parameters
        ----------
        mu: float
            log mean
        sigma: float
            variance

        """
        return np.round(sigma*np.random.lognormal(size=size)+mu)

    def exp(self, scale, size):
        """Draw from exponential distribution, float.

        Mutation or recombination rates

        Parameters
        ----------
        scale : TYPE
            DESCRIPTION.
        size : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # scale = beta = 1/lambda
        return np.random.exponential(scale, size)

    def beta_plot(self, alpha, beta, size=100000):
        """Draws a random value from a log normal distribution.

        Parameters
        ----------
        mu: float
            log mean
        sigma: float
            variance

        """
        import matplotlib.pyplot as plt
        points = np.random.beta(alpha, beta, size=size)
        count, bins, ignored = plt.hist(points, 100, normed=True, align='mid')

    def beta(self, alpha, beta, size, plot=False):
        """Draws a random value from a log normal distribution.

        Parameters
        ----------
        a: float
            beta
        b: float
            beta

        """
    #    alpha = ((((1-mu)/var) - (1/mu))*mu)**2
    #    beta = alpha*((1/mu) - 1)
        mu = alpha / (alpha + beta)
        var = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
        mode = (alpha - 1) / (alpha + beta - 2)
        assert 0 < mu < 1
        assert 0 < var < .25
        if plot:
            self.beta_plot(alpha, beta, size)
            print(f"mode:{mode}, mu:{mu}, var: {var}")
        return np.random.beta(alpha, beta, size)

    def constant(self, low, high, size):
        """Return a constant.

        Parameters
        ----------
        c: float, int

        """
        assert low == high, f"constant needs to be equal {low} != {high}"
        return np.full(size, low)

    def posterior(self, file, column, size):
        """Draws directly from a file with posteriors of parameter."""
        if os.path.exists(file):
            posterior_dist = np.loadtxt(file, delimiter="\t")
        posterior_param = posterior_dist[:, column]
        return np.random.choice(posterior_param, size)
