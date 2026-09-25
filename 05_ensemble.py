"""05_ensemble.py: combine the trained NREs of one data set combination into an ensemble.

Fast (seconds). Needs the predictions/ and run*_loss_test.npy files written by
04_train_nre.py for this combination.

Steps:
    1. Ensemble in concordance log R distribution: for every in concordance
       simulation pair, pick one of the NREs at random with probability
       p_i = exp(-loss_i) / sum_j exp(-loss_j), where loss_i is its validation
       loss, and keep that NRE's prediction.
    2. Tension T and concordance C of the observed log R against the ensemble
       distribution -> stats/ensemble_stats.npy. For Planck-CatWISE and
       Planck-CatSIM the observed log R is far in the tail, so a fitted Pearson
       type III distribution is used instead of the empirical CDF.
    3. For Planck-CatWISE and Planck-CatSIM, also recompute each run's T with
       the Pearson fit -> stats/{runs}runs_stats_pearson.npy.
    4. Diagnostic plots of every run's distribution and the ensemble.

Usage:
    python 05_ensemble.py --first planck --second racs
"""

# Import packages
import argparse
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import ecdf
from scipy.stats import norm
from tensionnet.utils import calcualte_stats
import config

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--first', required=True, choices=['planck', 'racs', 'nvss'])
parser.add_argument('--second', required=True, choices=['racs', 'nvss', 'catwise', 'catsim'])
parser.add_argument('--runs', type=int, default=config.N_RUNS)
parser.add_argument('--force', action='store_true', help='rebuild the ensemble even if it already exists')
args = parser.parse_args()

FIRST_dataset = args.first
SECOND_dataset = args.second
runs = args.runs

# Planck-CatWISE and Planck-CatSIM use a Pearson type III fit (observed log R is in the far tail)
use_pearson = FIRST_dataset == 'planck' and (SECOND_dataset == 'catwise' or SECOND_dataset == 'catsim')

# Set directories
BASE_DIR = config.nre_dir(FIRST_dataset, SECOND_dataset)
if not os.path.exists(BASE_DIR + 'evaluations/'):
    os.makedirs(BASE_DIR + 'evaluations/')
if not os.path.exists(BASE_DIR + 'stats/'):
    os.makedirs(BASE_DIR + 'stats/')

# Get the R-statistic of the real datasets
R_DIR = config.R_DIR
R = np.load(f'{R_DIR}/R_{FIRST_dataset}_{SECOND_dataset}.npy')
errorR = np.load(f'{R_DIR}/errorR_{FIRST_dataset}_{SECOND_dataset}.npy')/2 # convert to 1 sigma error
print('Using R = ', R, ' +/- ', errorR)

# Pearson type III CDF with the same interface as scipy's ecdf ("c.cdf.evaluate(x)")
class PearsonCDF:
    def __init__(self, mu, sigma, skew):
        self.dist = sp.stats.pearson3(skew, loc=mu, scale=sigma)
        self.cdf = self  # for "pearson.cdf.evaluate(x)" compatibility

    def evaluate(self, x):
        return self.dist.cdf(x)

# Load one run's in concordance predictions (one file per concordant chunk)
def load_run_predictions(run):
    predictions_part = []
    i = 1
    while os.path.exists(f'{BASE_DIR}predictions/run{run+1}_concordance{i}.npy'):
        predictions_part.append(np.load(f'{BASE_DIR}predictions/run{run+1}_concordance{i}.npy'))
        i += 1
    return np.concatenate(predictions_part, axis=0)

# ---------------------------------------------------------------------------
# 1. Build the ensemble in concordance distribution
# ---------------------------------------------------------------------------
if args.force or (not os.path.exists(f'{BASE_DIR}predictions/ensemble_concordance.npy')):
    # Load all predictions
    predictions = []
    for run in range(runs):
        predictions.append(load_run_predictions(run))
    predictions = np.array(predictions)

    # Load all loss scores
    loss_scores_test = []
    for run in range(runs):
        loss_scores_test.append(np.load(f'{BASE_DIR}run{run+1}_loss_test.npy'))
    loss_scores_test = np.array(loss_scores_test)
    print('Using loss scores: ', loss_scores_test)

    # Compute sampling probabilities
    beta = 1
    weights = np.exp(-beta * loss_scores_test)
    probabilities = weights / weights.sum()
    print('Probabilities: ', probabilities)

    # Sample one run index for each sample, with probability p_i = exp(-loss_i)/Z
    samples = predictions.shape[1]
    np.random.seed(42)
    chosen_runs = np.random.choice(runs, size=samples, p=probabilities)

    # Build indexer to select entire prediction vectors for each sample
    indexer = (chosen_runs, np.arange(samples)) + tuple([slice(None)] * (predictions.ndim - 2))
    final_predictions = predictions[indexer]  # shape: (n_samples, ...) — the ensemble-built logR predictions

    r_predicted = np.asarray(final_predictions).flatten()
    r = r_predicted
    np.save(f'{BASE_DIR}predictions/ensemble_concordance.npy', r)
else:
    r = np.load(f'{BASE_DIR}predictions/ensemble_concordance.npy')
mask = np.isfinite(r)

# Plot in concordance distribution and cumulative distribution
fig, axes = plt.subplots(1, 2, figsize=(4.5, 3))
axes[0].hist(r[mask], bins=25, density=True)
axes[0].set_xlabel(r'$\log R$')
axes[0].set_ylabel('Density')
axes[0].axvline(R, ls='--', c='r')
axes[0].axvspan(R - errorR, R + errorR, alpha=0.1, color='r')
axes[0].set_xlim(-20, 20)
rsort  = np.sort(r[mask])
if use_pearson:
    print('USING PEARSON FIT')
    mu = np.mean(r[mask])
    std = np.std(r[mask])
    skew = sp.stats.skew(r[mask])
    c = PearsonCDF(mu, std, skew)
else:
    print('USING EMPIRICAL CDF')
    c = ecdf(rsort)
axes[1].plot(rsort, c.cdf.evaluate(rsort))
axes[1].axhline(c.cdf.evaluate(R), ls='--', color='r')
axes[1].axhspan(c.cdf.evaluate(R - errorR), c.cdf.evaluate(R + errorR), alpha=0.1, color='r')
axes[1].set_xlabel(r'$\log R$')
axes[1].set_ylabel(r'$P(\log R < \log R^\prime)$')
axes[1].set_xlim(-20, 20)
plt.tight_layout()
plt.savefig(BASE_DIR + 'evaluations/' + f'ensemble_distribution.pdf', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 2. Ensemble tension and concordance
# ---------------------------------------------------------------------------
sigmaD = []
sigmaA = []
widths = []
mu, width = norm.fit(r[mask]) # width is taken as the standard deviation of the fitted Gaussian
widths.append(width)

stats = calcualte_stats(R, errorR, c)
sigmaD.append(stats[:3])
sigmaA.append(stats[3:6])
sigmaA = np.array(sigmaA)
sigmaD = np.array(sigmaD)
mean_sigmaA = sigmaA[:, 0].mean()
mean_sigmaD = sigmaD[:, 0].mean()
sigmaA_lower = sigmaA[:, 0] - sigmaA[:, 2]
sigmaA_upper = sigmaA[:, 1] - sigmaA[:, 0]
sigmaD_lower = sigmaD[:, 0] - sigmaD[:, 1]
sigmaD_upper = sigmaD[:, 2] - sigmaD[:, 0]
lower_mean_sigmaD_error = 1/np.sqrt(1) * np.sqrt(np.sum((sigmaD_lower)**2))
upper_mean_sigmaD_error = 1/np.sqrt(1) * np.sqrt(np.sum((sigmaD_upper)**2))
lower_mean_sigmaA_error = 1/np.sqrt(1) * np.sqrt(np.sum((sigmaA_lower)**2))
upper_mean_sigmaA_error = 1/np.sqrt(1) * np.sqrt(np.sum((sigmaA_upper)**2))
print(f"Tension: {mean_sigmaD:.3f} (+{upper_mean_sigmaD_error:.3f}/-{lower_mean_sigmaD_error:.3f})")
print(f"Concordance: {mean_sigmaA:.3f} (+{upper_mean_sigmaA_error:.3f}/-{lower_mean_sigmaA_error:.3f})")
np.save(f'{BASE_DIR}stats/ensemble_stats.npy', np.array([sigmaD, sigmaA], dtype=object))

# ---------------------------------------------------------------------------
# 3. Per-run tension with the Pearson fit (Planck-CatWISE and Planck-CatSIM only)
# ---------------------------------------------------------------------------
if use_pearson:
    print('Calculating stats using Pearson distribution fit')
    STATS_FILE_PEARSON = BASE_DIR + 'stats/' + f'{runs}runs_stats_pearson.npy'
    sigmaD = []
    sigmaA = []
    for run in range(runs):
        r_run = load_run_predictions(run).flatten()
        mask_run = np.isfinite(r_run)

        mu = np.mean(r_run[mask_run])
        std = np.std(r_run[mask_run])
        skew = sp.stats.skew(r_run[mask_run])
        c = PearsonCDF(mu, std, skew)

        stats = calcualte_stats(R, errorR, c)
        sigmaD.append(stats[:3])
        sigmaA.append(stats[3:6])
    sigmaD = np.array(sigmaD)
    sigmaA = np.array(sigmaA)

    np.save(STATS_FILE_PEARSON, np.array([sigmaD, sigmaA], dtype=object))
    print('Saved Pearson stats to', STATS_FILE_PEARSON)

# ---------------------------------------------------------------------------
# 4. Plot all distributions (every run and the ensemble)
# ---------------------------------------------------------------------------
predictions = []
for run in range(runs):
    predictions.append(load_run_predictions(run).flatten())
predictions.append(np.load(f'{BASE_DIR}predictions/ensemble_concordance.npy'))
rmax = 20
rmin = -20
xgrid = np.linspace(rmin, rmax, 1000)
plt.figure(figsize=(6, 4))
for run_idx, r_predicted in enumerate(predictions):
    r = r_predicted
    mask = np.isfinite(r)
    kde_actual = sp.stats.gaussian_kde(r[mask])
    if run_idx < runs:
        plt.plot(xgrid, kde_actual.evaluate(xgrid), label=f'Run {run_idx+1}', alpha=0.5)
    else:
        plt.plot(xgrid, kde_actual.evaluate(xgrid), label='Ensemble', color='black', linewidth=2)
plt.xlabel(r'$\log R$')
plt.ylabel('Density')
plt.axvline(R, ls='--', color='r', label='R')
plt.axvspan(R - errorR, R + errorR, alpha=0.1, color='r')
plt.legend()
plt.tight_layout()
plt.savefig(f'{BASE_DIR}evaluations/' + 'all_distributions.pdf', bbox_inches='tight')
plt.close()

if use_pearson:
    plt.figure(figsize=(6, 4))
    for run_idx, r_predicted in enumerate(predictions):
        r = r_predicted
        mask = np.isfinite(r)
        mu = np.mean(r[mask])
        std = np.std(r[mask])
        skew = sp.stats.skew(r[mask])
        kde_pearson = sp.stats.pearson3(skew, loc=mu, scale=std)
        if run_idx < runs:
            plt.plot(xgrid, kde_pearson.pdf(xgrid), label=f'Run {run_idx+1}', alpha=0.5)
        else:
            plt.plot(xgrid, kde_pearson.pdf(xgrid), label='Ensemble', color='black', linewidth=2)
    plt.xlabel(r'$\log R$')
    plt.ylabel('Density')
    plt.axvline(R, ls='--', color='r', label='R')
    plt.axvspan(R - errorR, R + errorR, alpha=0.1, color='r')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{BASE_DIR}evaluations/' + 'all_distributions_pearson.pdf', bbox_inches='tight')
    plt.close()
