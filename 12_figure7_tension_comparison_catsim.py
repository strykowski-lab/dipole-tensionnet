"""12_figure7_tension_comparison_catsim.py: Figure 7 - tension between CatSIM (Eddington bias) and Planck, RACS-low and NVSS from each NRE and the ensemble.

Fast (seconds). Reads, for planck_catsim, racs_catsim and nvss_catsim:
    outputs/nre/{first}_{second}/stats/5runs_stats.npy (04_train_nre.py;
        5runs_stats_pearson.npy from 05_ensemble.py for Planck-CatWISE/CatSIM)
    outputs/nre/{first}_{second}/stats/ensemble_stats.npy (05_ensemble.py)
Writes outputs/figures/figure7.pdf.
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
fig, axes_list = plt.subplots(3, 1, figsize=(scale * 4.1, scale * 1.75 * 7 * scale_y * 3/5))
fig.subplots_adjust(hspace=hspace)
SECOND_dataset = 'catsim'
for axes, FIRST_dataset,  in zip(axes_list, ['planck', 'racs', 'nvss']):

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
    axes.set_ylabel(r'Tension ($N\sigma$)')
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

    if FIRST_dataset == 'planck':
        axes.set_xlim(0.5,5.5)
        axes.set_ylim(4.1,10.4)
        axes.set_yticks([5, 6, 7, 8, 9, 10])
        axes.set_yticklabels(['5', '6', '7', '8', '9', '10'])
        axes.set_xlabel(None)
        axes.set_xticklabels([])
        axes.set_ylabel(None)
        axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
        axes.text(0.965, 0.13, rf'\textit{{Planck}}--CatSIM', 
                transform=axes.transAxes, ha='right', va='top')
        axes.legend(loc='upper left')

    if FIRST_dataset == 'racs':
        axes.set_xlim(0.5,5.5)
        axes.set_ylim(2.9,3.95)
        axes.set_xlabel(None)
        axes.set_ylabel(r'Tension ($N\sigma$)')
        axes.set_xticklabels([])
        axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
        axes.text(0.965, 0.14, rf'RACS-low--CatSIM', 
                transform=axes.transAxes, ha='right', va='top')

    if FIRST_dataset == 'nvss':
        axes.set_ylim(0.38,2.01)
        axes.set_xlim(0.5,5.5)
        axes.set_xticks(np.arange(len(sigmaDs)) + 1)
        axes.set_xlabel('NRE Training Run')
        axes.set_xticklabels([])
        axes.set_ylabel(None)
        axes.set_yticks([0.6, 1.0, 1.4, 1.8])
        # axes.set_yticks([0.4, 0.8, 1.2, 1.6])
        # axes.set_yticklabels(['0.4', '0.8', '1.2', '1.6'])
        axes.set_xticks(np.arange(len(sigmaDs)) + 1)
        axes.set_xticklabels(np.arange(len(sigmaDs)) + 1)
        axes.tick_params(axis='both', which='minor', bottom=False, left=True, right=True, top=False)
        axes.text(0.965, 0.14, rf'NVSS--CatSIM', 
                transform=axes.transAxes, ha='right', va='top')

fig.subplots_adjust(hspace=hspace)
plt.savefig(config.FIGURES + '/figure7.pdf', bbox_inches='tight')