#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:53:16 2020
@author: Scott T. Small
"""
import numpy as np
from math import log


class Ddist(object):
    """draws values from distribution."""

    def __init__(self, low, high, size):
        """Create command line for coalescent sims.

        Returns
        -------
        None.

        """
        self.low = 0
        self.high = 1
        self.size = 1
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
        """Draws a random value from a uniform distribution, float.

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

    def unif_int(self, low, high, size):
        """Draws a random value from a uniform distribution, int.

        Parameters
        ----------
        low: int
            lower bound
        high: int
            upper bound

        """
        return np.random.randint(low, high+1, size)

    def norm_int(self, mu, sigma, size):
        """Draws a random value from a normal distribution, int.

        Parameters
        ----------
        mu: int
            mean
        sigma: int
            variance

        """
        return np.round(sigma*np.random.randn()+mu, size)

    def log_norm_int(self, low, high, size, base=10):
        """Draws a random value from a log normal distribution.

        Parameters
        ----------
        mu: float
            log mean
        sigma: float
            variance

        """
        log_low = log(low, base)
        log_high = log(high, base)
        ln = np.power(base, np.random.lognormal(log_low, log_high, size))
        return ln

    def beta_plot(self, alpha, beta, size):
        """Draws a random value from a log normal distribution.

        Parameters
        ----------
        mu: float
            log mean
        sigma: float
            variance

        """
        import matplotlib.pyplot as plt
        points = np.random.beta(alpha, beta, size=100000)
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
