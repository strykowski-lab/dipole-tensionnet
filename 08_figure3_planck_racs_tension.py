"""08_figure3_planck_racs_tension.py: Figure 3 - Planck-RACS-low tension from each NRE, the ensemble,
nested sampling and Bayesian suspiciousness.

Fast (seconds). Reads, for outputs/nre/planck_racs/stats/:
    5runs_stats.npy (04_train_nre.py), ensemble_stats.npy (05_ensemble.py),
    ns_stats.npy (02c_validation_distribution.py)
and data/suspiciousness/planck_racs_suspiciousness_stats.npy.
Writes outputs/figures/figure3.pdf and prints the ensemble and ground-truth tensions.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scienceplots
import matplotlib as mpl
from scipy.stats import ecdf
from tensionnet.utils import calcualte_stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import config
if not os.path.exists(config.FIGURES):
    os.makedirs(config.FIGURES)

np.random.seed(42)
plt.style.use('science')
mpl.rcParams.update({
    'font.size': 8,                # match MNRAS caption font
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# Setup
FIRST_dataset = 'planck'
SECOND_dataset = 'racs'

runs = config.N_RUNS

# Set directories
BASE_DIR = config.nre_dir(FIRST_dataset, SECOND_dataset)
STATS_FILE = BASE_DIR + 'stats/' + f'{runs}runs_stats.npy'

# Get the R-statistic of the real datasets
R_DIR = config.R_DIR
R = np.load(f'{R_DIR}/R_{FIRST_dataset}_{SECOND_dataset}.npy')
errorR = np.load(f'{R_DIR}/errorR_{FIRST_dataset}_{SECOND_dataset}.npy')/2 # convert to 1 sigma error
print('Using R = ', R, ' +/- ', errorR)

# --- Tension from each NRE training run, and their average ---
runs_stats = np.load(STATS_FILE, allow_pickle=True)
sigmaD = runs_stats[0]

sigmaD = np.array(sigmaD)
mean_sigmaD = sigmaD[:, 0].mean()
sigmaDs = sigmaD[:, 0]
sigmaD_lower = sigmaD[:, 0] - sigmaD[:, 1]
sigmaD_upper = sigmaD[:, 2] - sigmaD[:, 0]
norm_sigmaD = sigmaDs / mean_sigmaD
lower_mean_sigmaD_error = 1/np.sqrt(runs) * np.sqrt(np.sum((sigmaD_lower)**2))
upper_mean_sigmaD_error = 1/np.sqrt(runs) * np.sqrt(np.sum((sigmaD_upper)**2))

# Tension figure
fig, axes = plt.subplots(1, 1, figsize=(3.5, 2.8))
axes.errorbar(np.arange(len(sigmaDs)) + 1, sigmaDs, yerr=[sigmaD_lower, sigmaD_upper], fmt='o')
axes.set_xticks(np.arange(len(sigmaDs)) + 1)
axes.set_ylabel(r'Tension ($N\sigma$)')
axes.set_xlabel('NRE Training Run')
axes.axhline(mean_sigmaD, ls='--', c='r', label='Predicted Tension (average across runs)')
axes.axhspan(mean_sigmaD - lower_mean_sigmaD_error, mean_sigmaD + upper_mean_sigmaD_error, alpha=0.1, color='r')
plt.tight_layout()

# --- Ensemble tension ---
ensemble_stats = np.load(f'{BASE_DIR}stats/ensemble_stats.npy', allow_pickle=True)
ensemble_sigmaD = ensemble_stats[0]
ensemble_sigmaD = np.array(ensemble_sigmaD)
ensemble_mean_sigmaD = ensemble_sigmaD[:, 0].mean()
ensemble_sigmaDs = ensemble_sigmaD[:, 0]
ensemble_sigmaD_lower = ensemble_sigmaD[:, 0] - ensemble_sigmaD[:, 1]
ensemble_sigmaD_upper = ensemble_sigmaD[:, 2] - ensemble_sigmaD[:, 0]
ensemble_norm_sigmaD = ensemble_sigmaDs / ensemble_mean_sigmaD
ensemble_lower_mean_sigmaD_error = 1/np.sqrt(1) * np.sqrt(np.sum((ensemble_sigmaD_lower)**2))
ensemble_upper_mean_sigmaD_error = 1/np.sqrt(1) * np.sqrt(np.sum((ensemble_sigmaD_upper)**2))
axes.axhline(ensemble_mean_sigmaD, ls='--', c='C1', label=r'Predicted Tension ($T_\text{NRE}$, ensemble)')
axes.axhspan(ensemble_mean_sigmaD - ensemble_lower_mean_sigmaD_error, ensemble_mean_sigmaD + ensemble_upper_mean_sigmaD_error, alpha=0.15, color='C1')

if FIRST_dataset == 'planck' and SECOND_dataset == 'racs':
    # --- Nested sampling (ground-truth) tension ---
    ns_stats = np.load(f'{BASE_DIR}stats/ns_stats.npy', allow_pickle=True)
    ns_sigmaD = ns_stats[0]
    T_ns_base = ns_sigmaD[:, 0].mean()
    T_error_lower_ns_base = ns_sigmaD[:, 0] - ns_sigmaD[:, 1]
    T_error_upper_ns_base = ns_sigmaD[:, 2] - ns_sigmaD[:, 0]
    axes.axhline(T_ns_base, ls='--', c='C2', label=r'True Tension ($T_\text{NS}$, nested sampling)')
    axes.axhspan(T_ns_base - T_error_lower_ns_base[0], T_ns_base + T_error_upper_ns_base[0], alpha=0.15, color='C2')

if SECOND_dataset != 'catsim':
    # --- Bayesian suspiciousness tension ---
    suspiciousness_stats = np.load(f'{config.SUSPICIOUSNESS_DIR}/{FIRST_dataset}_{SECOND_dataset}_suspiciousness_stats.npy', allow_pickle=True) # [T, +err, -err]
    suspiciousness = suspiciousness_stats[0]
    suspiciousness_upper = suspiciousness_stats[1]
    suspiciousness_lower = suspiciousness_stats[2]
    axes.axhline(suspiciousness, ls='--', c='C0', label='Bayesian Suspiciousness')
    axes.axhspan(suspiciousness - suspiciousness_lower, suspiciousness + suspiciousness_upper, alpha=0.15, color='C0')

labels = {'planck': 'Planck', 'nvss': 'NVSS', 'racs': 'RACS', 'catwise': 'CatWISE', 'catsim': 'CatSIM'}
# plt.title(f'{labels[FIRST_dataset]} -- {labels[SECOND_dataset]}')
axes.set_ylim(3.16,3.85)
axes.set_xlim(0.5,5.5)
axes.set_xticks(np.arange(len(sigmaDs)) + 1)
# axes.set_yticks([2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7])
axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
# axes.legend(framealpha=0.6, facecolor='white', frameon=True, bbox_to_anchor=(0.02, 0.025), loc='lower left')
axes.legend(loc='upper left')

# Add label in bottom right
axes.text(0.96, 0.124, rf'\textit{{Planck}}--RACS-low', 
        transform=axes.transAxes, ha='right', va='top')

plt.savefig(config.FIGURES + '/figure3.pdf', bbox_inches='tight')

if SECOND_dataset != 'catsim':
    # Compute the approximate sigma (error) value where the errorbars of 'average tension' and suspiciousness overlap, i.e. where they agree to
    avg_T = mean_sigmaD
    avg_T_err_lower = lower_mean_sigmaD_error
    avg_T_lower_bound = avg_T - avg_T_err_lower
    difference = avg_T_lower_bound - suspiciousness
    sigmas_difference = difference / suspiciousness_upper
    # sigmas_difference = max(0, difference / suspiciousness_upper)
    print(f'Approximate sigma agreement between Average Tension and Suspiciousness: {sigmas_difference} sigma')

print('--- Summary of Tension Values ---')
print('Ensemble Tension (T_NRE):', f'{ensemble_mean_sigmaD:.4f} +{ensemble_upper_mean_sigmaD_error:.4f} -{ensemble_lower_mean_sigmaD_error:.4f}')
print('True Tension (T_NS):', f'{T_ns_base:.4f} +{T_error_upper_ns_base[0]:.4f} -{T_error_lower_ns_base[0]:.4f}')