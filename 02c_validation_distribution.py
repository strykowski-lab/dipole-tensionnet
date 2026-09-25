"""02c_validation_distribution.py: the ground-truth in concordance log R distribution and T_NS.

Validation: only needed to make Figure 2 (and the ground-truth tension in
Figure 3) and to run the optional Optuna searches.

Fast (seconds). Needs the logR_{i}.npy files from 02b_validation_logR.py.

Steps:
    1. Collect log R_i and its 1 sigma errors for every nested-sampled pair
       -> r_actual.npy, rerr_actual_upper.npy, rerr_actual_lower.npy
    2. Account for the nested sampling error on each log R_i by redrawing every
       value five times from N(log R_i, err_i) -> r_actual_redraws.npy
       (this is P_NS(log R) in Figure 2)
    3. The ground-truth tension T_NS of the observed Planck-RACS log R
       -> outputs/nre/planck_racs/stats/ns_stats.npy (used in Figure 3)

Usage:
    python 02c_validation_distribution.py [--n-pairs 10000]
"""

# Import necessary packages and functions
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import ecdf
from tqdm import tqdm
from tensionnet.utils import calcualte_stats
import config

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--n-pairs', type=int, default=10000, help='number of nested-sampled pairs (paper: 10000)')
args = parser.parse_args()

# Set dirs
R_VALUES_DIR = config.VALIDATION_LOGR_DIR
NS_STATS_DIR = config.VALIDATION_DIR
if not os.path.exists(NS_STATS_DIR):
    os.makedirs(NS_STATS_DIR)

# ---------------------------------------------------------------------------
# 1. Collect log R and its errors for every pair
# ---------------------------------------------------------------------------
runs = np.arange(args.n_pairs)
r_actual = []
rerr_actual_upper = []
rerr_actual_lower = []
for run in tqdm(runs):
    r_actual.append(np.load(os.path.join(R_VALUES_DIR, f'logR_{run}.npy')))
    rerr = np.load(os.path.join(R_VALUES_DIR, f'logRerr_{run}.npy'))
    rerr_actual_upper.append(rerr[0])
    rerr_actual_lower.append(rerr[1])

np.save(os.path.join(NS_STATS_DIR, 'r_actual.npy'), r_actual)
np.save(os.path.join(NS_STATS_DIR, 'rerr_actual_upper.npy'), rerr_actual_upper)
np.save(os.path.join(NS_STATS_DIR, 'rerr_actual_lower.npy'), rerr_actual_lower)

# ---------------------------------------------------------------------------
# 2. Broaden the distribution by the nested sampling errors (5 redraws per pair)
# ---------------------------------------------------------------------------
r_actual = np.load(NS_STATS_DIR + 'r_actual.npy')
rerr_actual_lower = np.load(NS_STATS_DIR + 'rerr_actual_lower.npy')
rerr_actual_upper = np.load(NS_STATS_DIR + 'rerr_actual_upper.npy')
rerrs = np.mean([rerr_actual_lower, rerr_actual_upper], axis=0) / 2  # convert to 1sigma

np.random.seed(42)

redraws = 5
r_actual_redraws = np.array([[np.random.normal(r_actual[i], rerrs[i]) for i in range(len(r_actual))] for _ in range(redraws)])
np.save(NS_STATS_DIR + 'r_actual_redraws.npy', r_actual_redraws)
r_actual_redraws_concatenated = np.concatenate(r_actual_redraws)

# Define KDE grid range
kde_min = min(r_actual.min(), r_actual_redraws_concatenated.min())
kde_max = max(r_actual.max(), r_actual_redraws_concatenated.max())
kde_grid = np.linspace(kde_min, kde_max, 1000)

# Diagnostic plot: raw distribution, each redraw, and all redraws together
plt.figure(figsize=(12, 6))
kde_r_actual = gaussian_kde(r_actual)
plt.plot(kde_grid, kde_r_actual(kde_grid), label='r_actual KDE', linewidth=2)
for j in range(redraws):
    kde_redraw = gaussian_kde(r_actual_redraws[j])
    plt.plot(kde_grid, kde_redraw(kde_grid), label=f'redraw {j+1} KDE', linewidth=2, alpha=0.7)
kde_all = gaussian_kde(r_actual_redraws_concatenated)
plt.plot(kde_grid, kde_all(kde_grid), label='All redraws KDE', color='gray', alpha=0.7, linewidth=2)
plt.xlabel('r_actual')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(NS_STATS_DIR + 'broaden_ns_distribution.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Ground-truth tension T_NS for the observed Planck-RACS log R
# ---------------------------------------------------------------------------
FIRST_dataset = 'planck'
SECOND_dataset = 'racs'

# Get the R-statistic of the real datasets
R_DIR = config.R_DIR
R = np.load(f'{R_DIR}/R_{FIRST_dataset}_{SECOND_dataset}.npy')
errorR = np.load(f'{R_DIR}/errorR_{FIRST_dataset}_{SECOND_dataset}.npy')/2 # convert to 1 sigma error
print('Using R = ', R, ' +/- ', errorR)

# T_NS using the raw NS log R values (for reference only)
sigmaD = []
sigmaA = []
c = ecdf(np.sort(r_actual))
stats = calcualte_stats(R, errorR, c)
sigmaD.append(stats[:3])
sigmaA.append(stats[3:6])
sigmaD = np.array(sigmaD)
T_ns_base = sigmaD[:, 0].mean()
T_error_lower_ns_base = sigmaD[:, 0] - sigmaD[:, 1]
T_error_upper_ns_base = sigmaD[:, 2] - sigmaD[:, 0]
print('--------')
print('Using R distribution from raw NS R values')
print('T_NS statistic:', T_ns_base, '+', T_error_upper_ns_base, '-', T_error_lower_ns_base)

# T_NS using the 5 redrawn NS log R values (this is the one used in the paper)
sigmaD = []
sigmaA = []
c = ecdf(np.sort(r_actual_redraws_concatenated))
stats = calcualte_stats(R, errorR, c)
sigmaD.append(stats[:3])
sigmaA.append(stats[3:6])
sigmaD = np.array(sigmaD)
sigmaA = np.array(sigmaA)
T_ns_base = sigmaD[:, 0].mean()
T_error_lower_ns_base = sigmaD[:, 0] - sigmaD[:, 1]
T_error_upper_ns_base = sigmaD[:, 2] - sigmaD[:, 0]
BASE_DIR = config.nre_dir(FIRST_dataset, SECOND_dataset)
if not os.path.exists(BASE_DIR + 'stats/'):
    os.makedirs(BASE_DIR + 'stats/')
np.save(BASE_DIR + 'stats/ns_stats.npy', np.array([sigmaD, sigmaA], dtype=object))

print('--------')
print('Using R distribution from 5 redrawn NS R values')
print('T_NS statistic:', T_ns_base, '+', T_error_upper_ns_base, '-', T_error_lower_ns_base)
