"""06b_toy_problem.py: NRE tension vs. Bayesian suspiciousness on an analytical toy problem (Appendix C2).

*** HPC-GRADE (long) *** Trains 5 NREs for each of 7 reference observations and
4 prior widths (140 NREs, each on 1 000 000 simulation pairs for up to 1000 epochs).
It runs on a CPU but takes days; use a GPU node (see pbs/toy_problem.pbs).

For every reference observation pair (seed) and prior width Sigma it computes
    * the true tension, from the analytical in concordance log R distribution,
    * the Bayesian suspiciousness tension,
    * the NRE tension, from the in concordance log R distribution predicted by
      an NRE trained on matched/mismatched simulation pairs,
and saves the mean and error over the runs to outputs/toy_problem/overall_*.npy
(read by 14_figure9_toy_problem.py).

Needs the seeds from 06a_toy_problem_seed_finder.py, and lsbi and tensionnet
(https://github.com/htjb/tension-networks) on the PYTHONPATH.

Usage:
    python 06b_toy_problem.py
"""

import argparse
import config
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 = All messages, 1 = Filter out INFO messages, 2 = Filter out WARNING, 3 = Filter out all messages (Fatal errors remain)

from lsbi.model import LinearModel
from lsbi.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensionnet.tensionnet import nre
import tensorflow as tf
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
from random import shuffle
from scipy.stats import ecdf, norm
from scipy.special import gammaincc, ndtri
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import rc
import scipy as sp

os.environ["TF_USE_LEGACY_KERAS"] = "True"

# Tensorflow GPU configuration
print('Configuring TensorFlow...')
physical_devices = tf.config.list_physical_devices()
print("Available physical devices:", physical_devices) # Print TensorFlow device info
ngpus = len(tf.config.list_physical_devices('GPU'))
if ngpus == 1:
    print(f"{ngpus} GPU available.")
else:
    print("No GPU available - falling back to CPU.")
tf.config.set_soft_device_placement(True) # allow TensorFlow to fall back to CPU if GPU is not available
if os.uname().sysname == 'Darwin':
    print('Running on macOS - disabling XLA')
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
    os.environ["TF_DISABLE_MLIR_BRIDGE"] = "1"
else:
    print('Running on Linux - enabling XLA')
    tf.config.optimizer.set_jit(True) # enable XLA (Accelerated Linear Algebra) for performance improvements

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

# ---------------------------------------------------------------------------
# Matched (label 1) and mismatched (label 0) simulation pairs, split 80/20 and normalised
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--n-seeds', type=int, default=None, help='only use the first N reference observations (default: all 7)')
parser.add_argument('--n-runs', type=int, default=5, help='NREs per reference observation and prior (paper: 5)')
parser.add_argument('--n-sim', type=int, default=500000, help='in concordance simulations -> 2x as many matched/mismatched pairs (paper: 500000)')
parser.add_argument('--n-test-sim', type=int, default=5000, help='simulations for the in concordance distribution (paper: 5000)')
parser.add_argument('--epochs', type=int, default=1000, help='maximum epochs (paper: 1000)')
args = parser.parse_args()

TOY_PROBLEM_DIR = config.TOY_PROBLEM_DIR
n_runs = args.n_runs

# ---------------------------------------------------------------------------
# Evaluate the toy problem
# ---------------------------------------------------------------------------
# General model is 
# D = m + M theta +/- sqrt(C)
# theta = mu +/- sqrt(Sigma)

overall_meantrueTs = []
overall_errortrueTs = []
overall_meanpredictedTs = []
overall_errorpredictedTs = []
overall_mean_true_suspiciousness_tensions = []
overall_error_true_suspiciousness_tensions = []

seeds = np.load(TOY_PROBLEM_DIR + "seeds.npy")
if args.n_seeds is not None:
    seeds = seeds[:args.n_seeds]

print("Evaluating toy problem...")

for k, seed in enumerate(seeds):
    print(f'Seed {k}: {seed}')

    np.random.seed(seed)
    # Parameters & priors
    n = 3
    # Data B
    d =  50
    MB = np.random.rand(d, n)
    mB = np.random.rand(d)
    CB = 0.01
    # Data A
    MA = np.random.rand(d, n)
    mA = np.random.rand(d)
    CA = 0.01

    mu = np.random.rand(n)

    theta_true = multivariate_normal(mu, 0.01).rvs()

    Sigmas = [0.1, 1, 10, 100]

    true_distributions, predicted_distributions = [], []
    trueTs = []
    predictedTs = []
    true_suspiciousness_tensions = []
    
    for j in range(n_runs):
        print(f'Run {j}')
        truet = []
        S_tensions = []
        predt = []

        i = 0
        while i < len(Sigmas):
            Sigma = Sigmas[i]
            print(f'Sigma {i}: {Sigma}')

            # build models and generate data
            model_A = LinearModel(M=MA, m=mA, C=CA, 
                                mu=mu, Sigma=Sigma)
            model_B = LinearModel(M=MB, m=mB, C=CB, 
                                mu=mu, Sigma=Sigma)

            # Data AB
            d = model_A.d + model_B.d
            M = np.vstack([model_A.M, model_B.M])
            m = np.hstack([model_A.m, model_B.m])
            C = np.concatenate([model_A.C * np.ones(model_A.d), 
                                model_B.C * np.ones(model_B.d)])
            model_AB = LinearModel(M=M, m=m, C=C, mu=mu, Sigma=Sigma)

            if i == 0 and j == 0:
                # pull a real observation from the narrow prior
                A_obs = model_A.likelihood(theta_true).rvs()
                B_obs = model_B.likelihood(theta_true).rvs()

            Robs = logR(A_obs, B_obs)
            Iobs = logI(A_obs, B_obs)
            Sobs = Robs - Iobs
            sigma_obs = suspiciousness_tension(Sobs, n)

            N_sim = args.n_sim

            AB_sim = model_AB.evidence().rvs(N_sim)
            A_sim = AB_sim[:, :model_A.d]
            B_sim = AB_sim[:, model_A.d:]

            # build the nre
            nrei = nre(lr=1e-4)
            nrei.build_model(len(A_obs) + len(B_obs),
                                [25]*5, 'sigmoid')
            norm_data_train, norm_data_test, data_train, data_test, \
                labels_train, labels_test = \
                    simulation_process(A_sim, B_sim)
            nrei.data_test = norm_data_test
            nrei.labels_test = labels_test
            nrei.data_train = norm_data_train
            nrei.labels_train = labels_train
            nrei.simulation_func_A = None
            nrei.simulation_func_B = None

            # generate some test data to build the distribution
            N_test_sim = args.n_test_sim
            AB_sim = model_AB.evidence().rvs(N_test_sim)
            A_sim = AB_sim[:, :model_A.d]
            B_sim = AB_sim[:, model_A.d:]
            logr_true_dist = logR(A_sim, B_sim)

            # build the analytic distribution and calculate T and C
            logr_true_dist = np.sort(logr_true_dist)

            true_cdf = ecdf(logr_true_dist)
            true_sigmaD = norm.isf(true_cdf.cdf.evaluate(Robs)/2)
            print(f'True Tension for Target {(k+1)/2}: {true_sigmaD}')

            if np.isinf(true_sigmaD):
                print('[ALERT] True tension is infinite - repeating...')
                continue

            sigmaD_fname = TOY_PROBLEM_DIR + f'toy_problem_seed{k}_run{j}_sigma{i}.npy'
            
            if not os.path.exists(sigmaD_fname):
                print('sigmaD files not found - training model...')

                # train the model
                model, data_test, labels_test = nrei.training(epochs=args.epochs,
                                                            batch_size=1000)

                # normalise the test data
                A_sim = (A_sim - data_train[:, :len(A_obs)].mean(axis=0)) / \
                    data_train[:, :len(A_obs)].std(axis=0)
                B_sim = (B_sim - data_train[:, len(A_obs):].mean(axis=0)) / \
                    data_train[:, len(A_obs):].std(axis=0)

                data_test = np.hstack([A_sim, B_sim])

                # evalute the predicted distribution
                nrei.__call__(iters=data_test)
                predicted_r_dist = nrei.r_values

                mask = np.isfinite(predicted_r_dist)

                # calcualte the predicted T and C
                predicted_r_dist  = np.sort(predicted_r_dist[mask])
                c = ecdf(predicted_r_dist)

                sigmaD = norm.isf(c.cdf.evaluate(Robs)/2)

                if np.isinf(sigmaD):
                    print('[ALERT] Newly predicted tension is infinite - using Pearson 3 fit instead...')
                    
                    class PearsonCDF:
                        def __init__(self, mu, sigma, skew):
                            self.dist = sp.stats.pearson3(skew, loc=mu, scale=sigma)
                            self.cdf = self  # for "pearson.cdf.evaluate(x)" compatibility
                        
                        def evaluate(self, x):
                            return self.dist.cdf(x)

                    mu = np.mean(predicted_r_dist)
                    std = np.std(predicted_r_dist)
                    skew = sp.stats.skew(predicted_r_dist)
                    c = PearsonCDF(mu, std, skew)
                    sigmaD = norm.isf(c.cdf.evaluate(Robs)/2)

                    # Mark this as a Pearson fit
                    pearson_label_fname = sigmaD_fname.replace('.npy', '_pearson.txt')
                    with open(pearson_label_fname, 'w') as f:
                        f.write('')

                print(f'Saving sigmaD to {sigmaD_fname}')
                np.save(sigmaD_fname, sigmaD)

            else:
                print('Loading sigmaD from file...')
                sigmaD = np.load(sigmaD_fname)
                if np.isinf(sigmaD):
                    print('[ALERT] Predicted loaded tension is infinite - deleting file and repeating...')
                    os.remove(sigmaD_fname)
                    continue

            S_tensions.append(sigma_obs)
            truet.append(true_sigmaD)
            predt.append(sigmaD)

            i += 1
        
        trueTs.append(truet)
        predictedTs.append(predt)
        true_suspiciousness_tensions.append(S_tensions)

    trueTs = np.array(trueTs)
    predictedTs = np.array(predictedTs)

    meantrueTs = np.mean(trueTs, axis=0)
    meanpredictedTs = np.mean(predictedTs, axis=0)
    mean_true_suspiciousness_tensions = np.mean(true_suspiciousness_tensions, axis=0)

    errortrueTs = np.std(trueTs, axis=0)/np.sqrt(n_runs)
    errorpredictedTs = np.std(predictedTs, axis=0)/np.sqrt(n_runs)
    error_true_suspiciousness_tensions = np.std(true_suspiciousness_tensions, axis=0)/np.sqrt(n_runs)

    overall_meantrueTs.append(meantrueTs)
    overall_errortrueTs.append(errortrueTs)
    overall_meanpredictedTs.append(meanpredictedTs)
    overall_errorpredictedTs.append(errorpredictedTs)
    overall_mean_true_suspiciousness_tensions.append(mean_true_suspiciousness_tensions)
    overall_error_true_suspiciousness_tensions.append(error_true_suspiciousness_tensions)

overall_meantrueTs = np.array(overall_meantrueTs)
overall_errortrueTs = np.array(overall_errortrueTs)
overall_meanpredictedTs = np.array(overall_meanpredictedTs)
overall_errorpredictedTs = np.array(overall_errorpredictedTs)
overall_mean_true_suspiciousness_tensions = np.array(overall_mean_true_suspiciousness_tensions)
overall_error_true_suspiciousness_tensions = np.array(overall_error_true_suspiciousness_tensions)

np.save(TOY_PROBLEM_DIR + 'overall_meantrueTs.npy', overall_meantrueTs)
np.save(TOY_PROBLEM_DIR + 'overall_errortrueTs.npy', overall_errortrueTs)
np.save(TOY_PROBLEM_DIR + 'overall_meanpredictedTs.npy', overall_meanpredictedTs)
np.save(TOY_PROBLEM_DIR + 'overall_errorpredictedTs.npy', overall_errorpredictedTs)
np.save(TOY_PROBLEM_DIR + 'overall_mean_true_suspiciousness_tensions.npy', overall_mean_true_suspiciousness_tensions)
np.save(TOY_PROBLEM_DIR + 'overall_error_true_suspiciousness_tensions.npy', overall_error_true_suspiciousness_tensions)