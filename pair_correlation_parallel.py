import pdb
import csv
from math import log10, floor

import numpy as np
import scipy
import scipy.optimize as op
import scipy.special as spec
from scipy import stats
import pandas as pd

import pathos.multiprocessing as mp
from multiprocessing import Value
import emcee

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

import occur
from occur import Completeness, Hierarchy

matplotlib.rcParams.update({'font.size': 14})

recoveries_all = pd.read_csv('../recoveries_all_earth.csv')
completey_all  = Completeness(recoveries_all)
axislims = [0.1, 20]
masslims = [30, 6000]
completey_all.completeness_grid([0.9*axislims[0], 1.1*axislims[1]],
                                [0.9*masslims[0], 1.1*masslims[1]])

#complete_correlations = []
#complete_devvys = []
real_pairs = pd.read_csv('legacy_giant_pairs.csv')
a_array = np.array(real_pairs.axis)
m_array = np.array(real_pairs.mass)
indices = np.arange(len(real_pairs))
completeyray = [completey_all.interpolate(a_array[i], m_array[i]) for i in indices]

n = int(len(real_pairs)/2)

def parallely(n):
    complete_correlations = []
    complete_devvys = []
    for i in range(int(n)):
        M1 = []
        M2 = []
        j = 0
        while j < n:
            one_pair = np.random.choice(indices, size=2, replace=False)
            a1, m1 = a_array[one_pair[0]], m_array[one_pair[0]]
            a2, m2 = a_array[one_pair[1]], m_array[one_pair[1]]
            completey1, completey2 = completeyray[one_pair[1]], completeyray[one_pair[1]]
            randy1, randy2 = np.random.uniform(size=2)
            if randy1 <= completey1 and randy2 <= completey2:
                M1.append(m1)
                M2.append(m2)
                j += 1
        complete_correlations.append(R(np.log(M1), np.log(M2)))
        complete_devvys.append(devvy(M1, M2))
        return complete_correlations, complete_devvys


def paralleloo(n, m):
    p = mp.Pool(processes=m)
    output = p.map(parallely, (n*np.ones(m)))
    # Sort output.
    all_corrs = []
    all_devys = []
    for chunk in output:
        all_corrs.append(chunk[0])
        all_devys.append(chunk[1])

    return all_corrs, all_devys

corrys, devyys = paralleloo(64, 100)

datababy = pd.DataFrame()
datababy['correlation'] = corrys
datababy['deviation'] = devyys

datababy.to_csv('correlations_deviations_completeness.csv')
