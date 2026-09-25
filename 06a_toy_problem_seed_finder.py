"""06a_toy_problem_seed_finder.py: pick the reference observations for the toy problem (Appendix C2).

Fast by default: writes the seeds used in the paper to outputs/toy_problem/seeds.npy.
With --search it repeats the search (minutes to hours on a laptop): for each target
tension (0.5, 1, ..., 3.5 sigma) find a random seed whose linear-Gaussian toy
experiments A and B give an analytical tension within 0.05 sigma of the target,
for the narrowest prior (Sigma = 0.1 I).

Needs lsbi and tensionnet (https://github.com/htjb/tension-networks) on the PYTHONPATH.

Usage:
    python 06a_toy_problem_seed_finder.py [--search]
"""

import argparse
import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 = All messages, 1 = Filter out INFO messages, 2 = Filter out WARNING, 3 = Filter out all messages (Fatal errors remain)

from lsbi.model import LinearModel
from lsbi.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensionnet.tensionnet import nre
import numpy as np
from random import shuffle
from scipy.stats import ecdf, norm
from scipy.special import gammaincc, ndtri
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import rc

mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
rc('font', family='serif')
rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

# ---------------------------------------------------------------------------
# Analytical log R, log I and suspiciousness tension of the linear-Gaussian toy experiments
# ---------------------------------------------------------------------------
def logR(A, B):
        return model_AB.evidence().logpdf(np.hstack([A, B])) - \
            model_A.evidence().logpdf(A) - model_B.evidence().logpdf(B)

def logI(A, B):
        return model_A.dkl(A) + model_B.dkl(B) - \
            model_AB.dkl(np.hstack([A, B]))

def suspiciousness_tension(logS, d):
    log_p_value = np.log(gammaincc(d / 2, (d - 2 * logS) / 2))
    log_half_p = log_p_value - np.log(2)  # log(p/2)
    return ndtri(np.exp(log_half_p)) * -1

def simulation_process(simsA, simsB):
    # generate lots of simulations 
        
    idx = np.arange(0, len(simsB), 1)
    shuffle(idx)
    mis_labeled_simsB = simsB[idx]

    data = []
    for i in range(len(simsA)):
        """
        Sigma(log(r)) = 1 results in R >> 1 i.e. data sets are consistent
        sigma(log(r)) = 0 --> R << 1 i.e. data sets are inconsistent
        """
        data.append([*simsA[i], *simsB[i], 1]) 
        data.append([*simsA[i], *mis_labeled_simsB[i], 0])
    data = np.array(data)
    idx = np.arange(0, 2*len(simsA), 1)
    shuffle(idx)
    labels = data[idx, -1]
    data = data[idx, :-1]
    
    print('Simulations built.')
    print('Splitting data and normalizing...')

    data_train, data_test, labels_train, labels_test = \
            train_test_split(data, labels, test_size=0.2)
    
    labels_test = labels_test
    labels_train = labels_train

    data_trainA = data_train[:, :len(simsA[0])]
    data_trainB = data_train[:, len(simsA[0]):]
    data_testA = data_test[:, :len(simsA[0])]
    data_testB = data_test[:, len(simsA[0]):]

    data_testA = (data_testA - data_trainA.mean(axis=0)) / \
        data_trainA.std(axis=0)
    data_testB = (data_testB - data_trainB.mean(axis=0)) / \
        data_trainB.std(axis=0)
    data_trainA = (data_trainA - data_trainA.mean(axis=0)) / \
        data_trainA.std(axis=0)
    data_trainB = (data_trainB - data_trainB.mean(axis=0)) / \
        data_trainB.std(axis=0)

    norm_data_train = np.hstack([data_trainA, data_trainB])
    norm_data_test = np.hstack([data_testA, data_testB])
    return norm_data_train, norm_data_test, data_train, \
        data_test, labels_train, labels_test

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--search', action='store_true', help='search for the seeds again instead of using the ones found for the paper')
args = parser.parse_args()

TOY_PROBLEM_DIR = config.TOY_PROBLEM_DIR
if not os.path.exists(TOY_PROBLEM_DIR):
    os.makedirs(TOY_PROBLEM_DIR)

# ---------------------------------------------------------------------------
# Search for one seed per target tension
# ---------------------------------------------------------------------------
print("Finding seeds for toy problem...")
np.random.seed(42)

eps = 0.05

targets = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
best_seeds = {t: None for t in targets}
best_diffs = {t: np.inf for t in targets}

found = {t: False for t in targets}

pbar = tqdm(total=len(targets), desc="Targets found")

found_seeds = not args.search

if not found_seeds:
    seed = 0
    while not all(found.values()):
        np.random.seed(seed)

        # Parameters & priors
        n = 3
        d = 50

        # Data B
        MB = np.random.rand(d, n)
        mB = np.random.rand(d)
        CB = 0.01

        # Data A
        MA = np.random.rand(d, n)
        mA = np.random.rand(d)
        CA = 0.01

        mu = np.random.rand(n)
        theta_true = multivariate_normal(mu, 0.01).rvs()

        # Build models
        model_A = LinearModel(M=MA, m=mA, C=CA, mu=mu, Sigma=0.1)
        model_B = LinearModel(M=MB, m=mB, C=CB, mu=mu, Sigma=0.1)

        # Combined model
        M = np.vstack([model_A.M, model_B.M])
        m = np.hstack([model_A.m, model_B.m])
        C = np.concatenate([
            model_A.C * np.ones(model_A.d),
            model_B.C * np.ones(model_B.d)
        ])

        model_AB = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=0.1)

        A_obs = model_A.likelihood(theta_true).rvs()
        B_obs = model_B.likelihood(theta_true).rvs()

        Robs = logR(A_obs, B_obs)

        # Simulate
        N_test_sim = 5000
        AB_sim = model_AB.evidence().rvs(N_test_sim)

        A_sim = AB_sim[:, :model_A.d]
        B_sim = AB_sim[:, model_A.d:]

        logr_true_dist = np.sort(logR(A_sim, B_sim))
        true_cdf = ecdf(logr_true_dist)
        true_sigmaD = norm.isf(true_cdf.cdf.evaluate(Robs) / 2)

        # Compare against all targets
        for t in targets:

            diff = abs(true_sigmaD - t)

            # If better than previous best, store it
            if diff < best_diffs[t]:
                best_diffs[t] = diff
                best_seeds[t] = seed

                # If within tolerance, mark as done
                if diff < eps:
                    if found[t] == False:
                        found[t] = True
                        pbar.update(1)
                        print(f"Target {t}σ satisfied with tension {true_sigmaD:.4f}, seed {seed}")
                    else:
                        print(f"Target {t}σ already satisfied, but found closer tension {true_sigmaD:.4f}, seed {seed}")
        
        seed += 1

    pbar.close()

    # Collect final seeds in order
    seeds = [best_seeds[t] for t in targets]
    np.save(TOY_PROBLEM_DIR + "seeds.npy", seeds)

# Seeds found for the paper
else:
    np.save(TOY_PROBLEM_DIR + "seeds.npy", [1685, 477, 1348, 1137, 467, 205, 188])
    
# Seeds found for the paper (target: seed (true tension)):
# 0.5σ: seed 1685 (0.4925)
# 1σ: seed 477 (0.9994)
# 1.5σ: seed 1348 (1.5008)
# 2σ: seed 1137 (1.9991)
# 2.5σ: seed 467 (2.5063)
# 3σ: seed 205 (3.0115)
# 3.5σ: seed 188 (3.5401)