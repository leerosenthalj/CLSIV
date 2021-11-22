import os
import csv
import pdb
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.optimize as op
import scipy.special as spec
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
import scipy.ndimage
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import kstest

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns
import corner

import astropy
from astropy import stats as astrostats
from astropy.timeseries import LombScargle

import emcee
#import celerite
import radvel

import rvsearch

class Completeness(object):
    """Object to handle a suite of injection/recovery tests

    Args:
        recoveries (DataFrame): DataFrame of injection/recovery tests from Injections class
        xcol (string): (optional) column name for independent variable. Completeness grids and
            interpolator will work in these axes
        ycol (string): (optional) column name for dependent variable. Completeness grids and
            interpolator will work in these axes

    """
    def __init__(self, recoveries, xcol='inj_au', ycol='inj_msini'):

        self.recoveries = recoveries

        self.xcol = xcol
        self.ycol = ycol

        self.grid = None
        self.interpolator = None

    def completeness_grid(self, xlim, ylim, resolution=20, xlogwin=0.5, ylogwin=0.5):

        xgrid = np.logspace(np.log10(xlim[0]),
                            np.log10(xlim[1]),
                            resolution)
        ygrid = np.logspace(np.log10(ylim[0]),
                            np.log10(ylim[1]),
                            resolution)

        xinj = self.recoveries[self.xcol]
        yinj = self.recoveries[self.ycol]

        good = self.recoveries['recovered']

        z = np.zeros((len(ygrid), len(xgrid)))
        last = 0
        for i,x in enumerate(xgrid):
            for j,y in enumerate(ygrid):
                xlow  = 10**(np.log10(x) - xlogwin/2)
                xhigh = 10**(np.log10(x) + xlogwin/2)
                ylow  = 10**(np.log10(y) - ylogwin/2)
                yhigh = 10**(np.log10(y) + ylogwin/2)

                xbox = yinj[np.where((xinj <= xhigh) & (xinj >= xlow))[0]]
                if len(xbox) == 0 or y > max(xbox) or y < min(xbox):
                    z[j, i] = np.nan
                    continue

                boxall = np.where((xinj <= xhigh) & (xinj >= xlow) &
                                  (yinj <= yhigh) & (yinj >= ylow))[0]
                boxgood = np.where((xinj[good] <= xhigh) &
                                   (xinj[good] >= xlow) & (yinj[good] <= yhigh) &
                                   (yinj[good] >= ylow))[0]

                if len(boxall) > 10:
                    z[j, i] = float(len(boxgood))/len(boxall)
                    last = float(len(boxgood))/len(boxall)
                else:
                    z[j, i] = np.nan

        self.grid = (xgrid, ygrid, z)

    def interpolate(self, x, y, refresh=False):

        if self.interpolator is None or refresh:
            assert self.grid is not None, "Must run Completeness.completeness_grid()."
            zi = self.grid[2].T
            self.interpolator = RegularGridInterpolator((self.grid[0], self.grid[1]), zi,
                                                        bounds_error=False, fill_value=0.001) # Maybe don't set fill

        return self.interpolator(np.array([np.atleast_1d(x), np.atleast_1d(y)]).T)


class Hierarchy(object):
    """Do hierarchical Bayesian sampling of occurrence posteriors, based on DFM et al. 2014.
    Args:
        pop (pandas DataFrame): dataframe of planet parameter chains

    """
    def __init__(self, pop, completeness, res=4, bins=np.array([[[np.log(0.02), np.log(20)],
                                                                 [np.log(2.), np.log(6000)]]]),
                                                                  nstars=None, mass_lim=[3, 7000],
                                                                  fraction=False, lenrun=1000,
                                                                  chainname='occur_chains.csv'):
        # TO-DO: Replace single-param planets with paths to posteriors.
        self.pop          = pop # Replace pairs of m & a with chains
        self.completeness = completeness # Completeness grid, defined as class object below.
        self.completeness.completeness_grid([0.01, 40], mass_lim)
        # Fill in completeness nans.
        self.completeness.grid[2][np.isnan(self.completeness.grid[2])] = 1. #0.99


        self.res = int(round(res)) # Resolution for logarithmic completeness integration.
        self.bins = bins # Logarithmic bins in msini/axis space.
        self.nbins = len(self.bins)
        self.lna_edges = np.unique(self.bins[:, 0])
        self.lnm_edges = np.unique(self.bins[:, 1])
        self.nabins = len(self.lna_edges) - 1
        self.nmbins = len(self.lnm_edges) - 1

        # Compute bin centers and widths.
        self.bin_widths  = np.diff(self.bins)
        self.bin_centers = np.mean(self.bins, axis=2)
        self.bin_areas   = self.bin_widths[:,0]*self.bin_widths[:,1]

        # Pre-compute integrated completeness for each bin.
        self.Qints = np.zeros(self.nbins)
        for n, binn in enumerate(self.bins):
            for i in np.arange(self.res):
                for j in np.arange(self.res):
                    lna_av = binn[0][0] + (i/self.res + 1/(2*self.res))*(binn[0][1] - binn[0][0])
                    lnm_av = binn[1][0] + (j/self.res + 1/(2*self.res))*(binn[1][1] - binn[1][0])
                    self.Qints[n] += (self.bin_areas[n][0]/self.res**2) * \
                                      self.completeness.interpolate(np.exp(lna_av),
                                                                    np.exp(lnm_av))

        axis  = []
        msini = []
        self.planetnames = np.unique([x[:-2] + x[-1] for x in pop.columns])
        self.starnames   = np.unique([x[:-1] for x in self.planetnames])
        self.nplanets    = len(self.planetnames)
        self.nsamples    = len(self.pop)

        if nstars is not None:
            self.nstars = nstars
        else:
            self.nstars = len(self.starnames)

        medians = pop.median() # Along chain axis, once using chains.
        for name in self.planetnames:
            axis.append(medians[[name[:-1] + 'a' + name[-1]]][0])
            msini.append(medians[[name[:-1] + 'M' + name[-1]]][0])
        self.pop_med = pd.DataFrame.from_dict({'axis':axis, 'msini':msini})

        self.fraction  = fraction
        self.lenrun    = lenrun
        self.chainname = chainname

    def max_like(self):
        ### Approximate max-likelihood occurrence values, with which to seed MCMC.
        mlvalues = np.empty((0, 2))
        for n, binn in enumerate(self.bins):
            # Integrate completeness across each individual bin.
            a1 = np.exp(binn[0][0])
            a2 = np.exp(binn[0][1])
            M1 = np.exp(binn[1][0])
            M2 = np.exp(binn[1][1])
            planets = self.pop_med.query('axis >= @a1 and axis < @a2 and \
                                         msini >= @M1 and msini < @M2')
            nplanets = len(planets)
            ml  = nplanets/self.Qints[n]
            uml = ml/np.sqrt(nplanets)
            if not np.isfinite(ml):
                ml = 0.01
            if not np.isfinite(uml):
                uml = 1.
            mlvalues = np.append(mlvalues, np.array([[ml, uml]]), axis=0)
        mlvalues[np.isnan(mlvalues)] = 0.01
        mlvalues[mlvalues == 0] = 0.01
        self.mlvalues = mlvalues
        self.ceiling = np.amax(mlvalues)

    def occurrence(self, lna, lnm, theta):
        # Select appropriate bins, given lna & lnm.
        ia = np.atleast_1d(np.digitize(lna, self.lna_edges) - 1)
        im = np.atleast_1d(np.digitize(lnm, self.lnm_edges) - 1)
        iao = np.copy(ia)
        imo = np.copy(im)
        ia[ia < 0] = 0
        im[im < 0] = 0
        ia[ia > self.nabins - 1] = self.nabins - 1
        im[im > self.nmbins - 1] = self.nmbins - 1

        occur = theta[ia + im*self.nabins]
        # Return filler value for samples outside of the bin limits.
        occur[iao < 0] = 0.01
        occur[imo < 0] = 0.01
        occur[iao > self.nabins - 1] = 0.01
        occur[imo > self.nmbins - 1] = 0.01
        return occur

    def lnlike(self, theta):
        # Implement probability hard-bound prior.
        if self.fraction:
            if np.sum(theta * self.bin_areas)/self.nstars > 1:
                return -np.inf
        if np.any((theta <= 0) + (theta > 10*self.ceiling)):
            return -np.inf
        sums = []
        for planet in self.planetnames:
            probs = []
            sample_a = np.array(self.pop[planet[:-2] + '_a' + planet[-1]])
            sample_M = np.array(self.pop[planet[:-2] + '_M' + planet[-1]])
            probs = self.completeness.interpolate(sample_a, sample_M)*self.occurrence(
                                           np.log(sample_a), np.log(sample_M), theta)
            #print(planet, probs)
            sums.append(np.sum(probs))

        # Integrate the observed occurrence over all bins.
        nexpect = 0
        for i, binn in enumerate(self.bins):
            for j in np.arange(4):
                for k in np.arange(4):
                    lna_av = binn[0][0] + (0.25*j + 0.125)*(binn[0][1] - binn[0][0])
                    lnm_av = binn[1][0] + (0.25*k + 0.125)*(binn[1][1] - binn[1][0])
                    nexpect += (self.bin_areas[i][0]/16)*self.completeness.interpolate(
                                                                        np.exp(lna_av),
                                                                        np.exp(lnm_av))*self.occurrence(
                                                                        lna_av, lnm_av, theta)
        ll = -nexpect + np.sum(np.log(np.array(sums)/self.nsamples))
        if not np.isfinite(ll):
            return -np.inf
        return ll

    def lnpost(self, theta):
        return self.lnlike(theta)

    def sample(self, gp=False, parallel=False, save=True):
        nwalkers = 4*self.nbins
        ndim = self.nbins
        pos = np.array([np.abs(self.mlvalues[:, 0] + 0.001*np.random.randn(ndim)) \
                                                 for i in np.arange(nwalkers)]) + 0.0001

        if parallel:
            with Pool(8) as pool:
                if gp:
                    self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.gppost, pool=pool)
                else:
                    self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnpost, pool=pool)
                self.sampler.run_mcmc(pos, self.lenrun, progress=True)
                self.chains = self.sampler.chain[:, 100:, :].reshape((-1, ndim))
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnpost)
            self.sampler.run_mcmc(pos, self.lenrun, progress=True)
            self.chains = self.sampler.chain[:, 100:, :].reshape((-1, ndim))

        if save:
            chaindb = pd.DataFrame()
            for n, binn in enumerate(self.bins):
                chaindb['gamma{}'.format(n)] = self.chains[:, n]
            chaindb.to_csv(self.chainname)

    def run(self):
        self.max_like()
        self.sample()


def lngrid(min_a, max_a, min_M, max_M, resa, resm):
    lna1 = np.log(min_a)
    lna2 = np.log(max_a)
    lnM1 = np.log(min_M)
    lnM2 = np.log(max_M)

    dlna = (lna2 - lna1)/resa
    dlnM = (lnM2 - lnM1)/resm

    bins = []
    for i in np.arange(int(resa)):
        for j in np.arange(int(resm)):
            bins.append([[lna1 + i*dlna, lna1 + (i+1)*dlna],
                         [lnM1 + j*dlnM, lnM1 + (j+1)*dlnM]])

    return np.array(bins)
