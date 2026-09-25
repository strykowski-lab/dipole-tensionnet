"""07_figure2_nre_vs_ns.py: Figure 2 - NRE vs. nested sampling in concordance log R (Planck-RACS-low).

Fast (a minute: 2D KDE of 50 000 points). Reads:
    outputs/nre/planck_racs/predictions/ensemble_concordance.npy (05_ensemble.py)
    outputs/validation/distribution/r_actual_redraws.npy (02c_validation_distribution.py)
Prints the KL divergence, Jensen-Shannon divergence and KS statistic between the
two distributions, and writes outputs/figures/figure2.pdf: a corner plot of the
true (nested sampling) and predicted (NRE ensemble) log R of the same pairs.
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
from sklearn.metrics import r2_score
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

# Setup
FIRST_dataset = 'planck'
SECOND_dataset = 'racs'

# Set directories
BASE_DIR = config.nre_dir(FIRST_dataset, SECOND_dataset)

# Get R statistic
CHAIN_DIR = config.R_DIR
R = np.load(f'{CHAIN_DIR}/R_{FIRST_dataset}_{SECOND_dataset}.npy')
Rs = np.array([R])
errorR = np.load(f'{CHAIN_DIR}/errorR_{FIRST_dataset}_{SECOND_dataset}.npy') / 2  # 1 sigma error
print('Using R = ', R, ' +/- ', errorR)

# Nested sampling ground truth: 5 redraws of each of the 10 000 nested-sampled pairs
r_actual_redraws = np.load(config.VALIDATION_DIR + 'r_actual_redraws.npy')
# NRE ensemble prediction for the same pairs (the first pairs of the first concordant chunk)
r_predicted_ensemble = np.asarray(np.load(f'{BASE_DIR}predictions/ensemble_concordance.npy')).flatten()
r_predicted = r_predicted_ensemble[:r_actual_redraws.shape[1]]
r_actual = np.concatenate(r_actual_redraws)

# --- Limits ---
minR_true = np.nanmin(r_actual)
maxR = max(np.nanmax(r_actual), np.nanmax(r_predicted))
pad = 0.05 * (maxR - minR_true)
xmin, xmax = minR_true - pad, maxR + pad
xmin = -6.5

# --- Stats ---
grid = np.linspace(xmin, xmax, 1000)
kde_true = sp.stats.gaussian_kde(r_actual)
kde_pred = sp.stats.gaussian_kde(r_predicted)
p = kde_true(grid)
q = kde_pred(grid)
p /= np.trapz(p, grid)
q /= np.trapz(q, grid)
eps = 1e-12
kl = np.trapz(p * np.log((p + eps) / (q + eps)), grid)
ks_stat, ks_pvalue = sp.stats.ks_2samp(r_actual, r_predicted)

# Compute Jensen-Shannon divergence
m = 0.5 * (p + q)
jsd = 0.5 * np.trapz(p * np.log((p + eps) / (m + eps)), grid) + 0.5 * np.trapz(q * np.log((q + eps) / (m + eps)), grid)

r2 = r2_score(r_actual, np.tile(r_predicted, 5))
print('R2 score of data:', r2)
r2 = r2_score(p, q)
print('R2 score of KDEs:', r2)

print('log KL divergence:', np.log(kl))
print('Jensen-Shannon divergence:', jsd)
print('log KS statistic:', np.log(ks_stat))
print('KS p-value:', ks_pvalue)

# --- Corner plot ---
make_corner = True
if make_corner:
    print('\nMaking corner plot...')
    # --- Figure layout: 2x2 equal boxes that touch ---
    fig, ax = plt.subplots(2, 2, figsize=(4.3, 4.3),
                        gridspec_kw=dict(width_ratios=[1, 1], height_ratios=[1, 1],
                                            wspace=0.0, hspace=0.0))

    # Force each axes box to be a square (same width & height)
    for axi in ax.flat:
        axi.set_box_aspect(1.0)

    # --- Bottom-left: 2D KDE (Pred vs True) with 1σ and 2σ contours ---
    xy = np.vstack([r_actual, np.tile(r_predicted, 5)])
    kde2d = sp.stats.gaussian_kde(xy)
    xx, yy = np.mgrid[xmin:xmax:200j, xmin:xmax:200j]
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde2d(grid_points).reshape(xx.shape)
    # Compute probability mass for contour levels
    zz_flat = zz.flatten()
    zz_sorted = np.sort(zz_flat)[::-1]
    cumsum = np.cumsum(zz_sorted)
    cumsum /= cumsum[-1]
    # Find density thresholds for sigma contours
    sigmas = [0.5, 1, 1.5, 2]
    def sigma_to_cumulative_frac(sigma):
        """Convert sigma level to cumulative probability fraction for 2D space"""
        return 1 - np.exp(-0.5 * sigma**2)
    def find_level(cumsum, zz_sorted, frac):
        idx = np.searchsorted(cumsum, frac)
        return zz_sorted[idx]
    cumulative_fracs = np.round([sigma_to_cumulative_frac(s) for s in sigmas], 3)
    print('Sigma input:', sigmas)
    print('Probability fractions for contours:', cumulative_fracs)
    levels = [find_level(cumsum, zz_sorted, frac) for frac in cumulative_fracs]
    # Plot contours
    ax[1, 0].contour(xx, yy, zz, levels=np.sort(levels), colors='C0', linewidths=1.5)
    ax[1, 0].plot([xmin, xmax], [xmin, xmax], '--', color='gray', lw=1, alpha=0.7)
    ax[1, 0].set_xlim(xmin, xmax)
    ax[1, 0].set_ylim(xmin, xmax)
    ax[1, 0].set_aspect('equal', adjustable='box')
    ax[1, 0].set_xlabel(r'True $\log R_\text{NS}$')
    ax[1, 0].set_ylabel(r'Predicted $\log R_\text{NRE}$')
    ax[1, 0].set_yticks([-5, 0, 5, 10])
    ax[1, 0].set_xticks([-5, 0, 5, 10])
    # Only internal ticks
    ax[1, 0].tick_params(axis='both', which='both', direction='in', top=True, right=True, bottom=True, left=True)

    # --- Top-right: Overlap comparison (no ticks) ---
    ax[0, 1].plot(grid, p, color='C0', lw=1.5)
    ax[0, 1].plot(grid, q, label=r'Predicted $\mathcal{P}_\text{NRE}(\log R)$', color='#ff7f0e', lw=1.5)
    ax[0, 1].fill_between(grid, 0, p, color='C0', alpha=0.2)
    ax[0, 1].fill_between(grid, 0, q, color='#ff7f0e', alpha=0.2)
    ax[0, 1].set_xlim(xmin, xmax)
    ax[0, 1].set_ylim(0, 0.22)
    ax[0, 1].set_xlabel(r'$\log R$')
    ax[0, 1].set_yticks([0.05, 0.10, 0.15, 0.20])
    ax[0, 1].tick_params(labelleft=False)
    ax[0, 1].set_xticks([-5, 0, 5, 10])
    ax[0, 1].set_xticklabels([-5, 0, 5, 10])
    # ax[0, 1].legend(loc='upper left')
    # Only internal ticks
    ax[0, 1].tick_params(axis='both', which='both', direction='in', top=False, right=True, bottom=True, left=True)

    # --- Top-left: KDE of true log R (show y ticks only, no x ticks) ---
    ax[0, 0].plot(grid, p, label=r'True $\mathcal{P}_\text{NS}(\log R)$', color='C0', lw=1.5)
    # ax[0, 0].plot([], [], label=r'Predicted $\mathcal{P}_\text{NRE}(\log R)$', color='#ff7f0e', lw=1.5)
    ax[0, 0].fill_between(grid, 0, p, color='C0', alpha=0.2)
    ax[0, 0].set_xlim(xmin, xmax)
    ax[0, 0].set_ylim(0, 0.22)
    ax[0, 0].set_xlabel(r'$\log R$')
    ax[0, 0].set_xticks([-5, 0, 5, 10])
    ax[0, 0].set_ylabel('PDF')
    ax[0, 0].set_yticks([0.05, 0.10, 0.15, 0.20])
    # ax[0, 0].legend(loc='upper left')
    # Only internal ticks
    ax[0, 0].tick_params(axis='both', which='both', direction='in', top=False, right=True, bottom=True, left=True)

    # --- Bottom-right: KDE of predicted log R (show x ticks only, no y ticks) ---
    ax[1, 1].plot(q, grid, color='#ff7f0e', lw=1.5)
    ax[1, 1].fill_betweenx(grid, 0, q, color='#ff7f0e', alpha=0.2)
    ax[1, 1].set_ylim(xmin, xmax)
    ax[1, 1].set_xlim(0, 0.22)
    ax[1, 1].set_xticks([0.05, 0.10, 0.15, 0.20])
    ax[1, 1].tick_params(labelleft=False)
    ax[1, 1].set_yticks([-5, 0, 5, 10])
    ax[1, 1].set_xlabel('PDF')
    ax[1, 1].plot([], [], label=r'True $\mathcal{P}_\text{NS}(\log R)$', color='C0', lw=1.5)
    ax[1, 1].plot([], [], label=r'Predicted $\mathcal{P}_\text{NRE}(\log R)$', color='#ff7f0e', lw=1.5)
    ax[1, 1].legend(
        loc='lower right',
        bbox_to_anchor=(1+0.03, 0+0.03)  # tweak these
    )
    # Only internal ticks
    ax[1, 1].tick_params(axis='both', which='both', direction='in', top=False, right=False, bottom=True, left=True)

    # --- Move right subplots 1 pixel to the left ---
    pixels = 1
    fig.canvas.draw()  # Needed to ensure positions are computed
    for i in [0, 1]:
        pos = ax[i, 1].get_position()
        ax[i, 1].set_position([pos.x0 - pixels/fig.dpi, pos.y0, pos.width, pos.height])
        
    # Tight layout and save
    plt.savefig(config.FIGURES + '/figure2.pdf', bbox_inches='tight', dpi=300)