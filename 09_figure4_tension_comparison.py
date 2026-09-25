"""09_figure4_tension_comparison.py: Figure 4 - tension between Planck, NVSS, RACS-low and CatWISE from each NRE, the ensemble and Bayesian suspiciousness.

Fast (seconds). Reads, for planck_catwise, racs_catwise, planck_nvss, racs_nvss and nvss_catwise:
    outputs/nre/{first}_{second}/stats/5runs_stats.npy (04_train_nre.py;
        5runs_stats_pearson.npy from 05_ensemble.py for Planck-CatWISE/CatSIM)
    outputs/nre/{first}_{second}/stats/ensemble_stats.npy (05_ensemble.py)
    data/suspiciousness/{first}_{second}_suspiciousness_stats.npy
Writes outputs/figures/figure4.pdf.
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
    'axes.linewidth': 0.8,         # thinner axes lines for print
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.0,
    'pdf.fonttype': 42,            # ensures editable text in PDF
    'ps.fonttype': 42,
})

scale = 3.65/4.1
scale_y = 0.88
hspace = 0.10
fig, axes_list = plt.subplots(5, 1, figsize=(scale * 4.1, scale * 1.75 * 7 * scale_y))
fig.subplots_adjust(hspace=hspace)
for axes, FIRST_dataset, SECOND_dataset in zip(axes_list, ['planck', 'racs', 'planck', 'racs', 'nvss'], ['catwise', 'catwise', 'nvss', 'nvss', 'catwise']):

    runs = config.N_RUNS

    # Set directories
    BASE_DIR = config.nre_dir(FIRST_dataset, SECOND_dataset)
    if FIRST_dataset == 'planck' and (SECOND_dataset == 'catwise' or SECOND_dataset == 'catsim'):
        STATS_FILE = BASE_DIR + 'stats/' + f'{runs}runs_stats_pearson.npy'
    else:
        STATS_FILE = BASE_DIR + 'stats/' + f'{runs}runs_stats.npy'

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

    # --- Tension from each NRE training run, and their average ---
    axes.errorbar(np.arange(len(sigmaDs)) + 1, sigmaDs, yerr=[sigmaD_lower, sigmaD_upper], fmt='o')
    axes.set_xticks(np.arange(len(sigmaDs)) + 1)
    axes.set_xlabel('NRE Training Run')
    axes.axhline(mean_sigmaD, ls='--', c='r', label='Predicted Tension (average across runs)')
    axes.axhspan(mean_sigmaD - lower_mean_sigmaD_error, mean_sigmaD + upper_mean_sigmaD_error, alpha=0.1, color='r')

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
        # --- Nested sampling (ground-truth) tension (not drawn: Planck-RACS-low is Figure 3) ---
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

    if FIRST_dataset == 'planck' and SECOND_dataset == 'catwise':
        axes.set_xlim(0.5,5.5)
        axes.set_ylim(4.75,7.325)
        axes.set_ylabel(None)
        # axes.set_yticks([5, 5.8, 6.6])
        axes.set_xlabel(None)
        axes.set_xticklabels([])
        axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
        axes.legend(loc='upper left')#, bbox_to_anchor=(0, 0.6))
        axes.text(0.965, 0.28, rf'\textit{{Planck}}--CatWISE', 
                transform=axes.transAxes, ha='right', va='top')

    if FIRST_dataset == 'racs' and SECOND_dataset == 'catwise':
        axes.set_xlim(0.5,5.5)
        axes.set_xticks(np.arange(len(sigmaDs)) + 1)
        axes.set_xlabel(None)
        axes.set_ylabel(None)
        axes.set_xticklabels([])
        axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
        axes.text(0.965, 0.54, rf'RACS-low--CatWISE', 
                transform=axes.transAxes, ha='right', va='top')

    if FIRST_dataset == 'planck' and SECOND_dataset == 'nvss':
        axes.set_xlim(0.5,5.5)
        axes.set_xticks(np.arange(len(sigmaDs)) + 1)
        # axes.set_yticks([1.8, 2.2, 2.6])
        axes.set_yticks([1.8, 2.1, 2.4, 2.7])
        axes.set_xlabel(None)
        axes.set_ylabel(r'Tension ($N\sigma$)')
        axes.set_xticklabels([])
        axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
        axes.text(0.965, 0.14, rf'\textit{{Planck}}--NVSS', 
                transform=axes.transAxes, ha='right', va='top')

    if FIRST_dataset == 'racs' and SECOND_dataset == 'nvss':
        axes.set_xlim(0.5,5.5)
        axes.set_xticks(np.arange(len(sigmaDs)) + 1)
        axes.set_xlabel(None)
        axes.set_ylabel(None)
        axes.set_xticklabels([])
        axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
        axes.text(0.965, 0.14, rf'NVSS--RACS-low', 
                transform=axes.transAxes, ha='right', va='top')

    if FIRST_dataset == 'nvss' and SECOND_dataset == 'catwise':
        axes.set_ylim(0.03,1.43)
        axes.set_xlim(0.5,5.5)
        axes.set_xticks(np.arange(len(sigmaDs)) + 1)
        axes.set_xticklabels(np.arange(len(sigmaDs)) + 1)
        # axes.set_yticks([0.4, 0.8, 1.2])
        axes.set_yticks([0.3, 0.6, 0.9, 1.2])
        axes.set_xlabel('NRE Training Run')
        axes.set_ylabel(None)
        axes.set_xticklabels([1, 2, 3, 4, 5])
        axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
        axes.text(0.965, 0.14, rf'NVSS--CatWISE', 
                transform=axes.transAxes, ha='right', va='top')

fig.subplots_adjust(hspace=hspace)
plt.savefig(config.FIGURES + '/figure4.pdf', bbox_inches='tight')