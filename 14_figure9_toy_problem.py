"""14_figure9_toy_problem.py: Figure 9 (Figure C1) - NRE tension and Bayesian suspiciousness vs. the true
tension of the analytical toy problem.

Fast (seconds). Reads outputs/toy_problem/overall_*.npy (06b_toy_problem.py) and
writes outputs/figures/figure9.pdf. One panel per prior width Sigma = 0.1 I, 1 I and 100 I,
using the first six reference observations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import matplotlib as mpl
import config
if not os.path.exists(config.FIGURES):
    os.makedirs(config.FIGURES)

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

# Each row is a different seed (sigma), each column is a different Sigma
overall_meantrueTs = np.load(config.TOY_PROBLEM_DIR + 'overall_meantrueTs.npy')
overall_errortrueTs = np.load(config.TOY_PROBLEM_DIR + 'overall_errortrueTs.npy')
overall_meanpredictedTs = np.load(config.TOY_PROBLEM_DIR + 'overall_meanpredictedTs.npy')
overall_errorpredictedTs = np.load(config.TOY_PROBLEM_DIR + 'overall_errorpredictedTs.npy')
overall_mean_true_suspiciousness_tensions = np.load(config.TOY_PROBLEM_DIR + 'overall_mean_true_suspiciousness_tensions.npy')
overall_error_true_suspiciousness_tensions = np.load(config.TOY_PROBLEM_DIR + 'overall_error_true_suspiciousness_tensions.npy')

Sigmas = [0.1, 1, 10, 100]

# Make each row a different Sigma, and each column a different seed (sigma)
overall_meantrueTs = overall_meantrueTs.T
overall_errortrueTs = overall_errortrueTs.T
overall_meanpredictedTs = overall_meanpredictedTs.T
overall_errorpredictedTs = overall_errorpredictedTs.T
overall_mean_true_suspiciousness_tensions = overall_mean_true_suspiciousness_tensions.T
overall_error_true_suspiciousness_tensions = overall_error_true_suspiciousness_tensions.T

# Only keep first 6 columns (seeds) for plotting
overall_meantrueTs = overall_meantrueTs[:, :6]
overall_errortrueTs = overall_errortrueTs[:, :6]
overall_meanpredictedTs = overall_meanpredictedTs[:, :6]
overall_errorpredictedTs = overall_errorpredictedTs[:, :6]
overall_mean_true_suspiciousness_tensions = overall_mean_true_suspiciousness_tensions[:, :6]
overall_error_true_suspiciousness_tensions = overall_error_true_suspiciousness_tensions[:, :6]

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(3.5, 6),
                        gridspec_kw=dict(
                                            wspace=0, hspace=0.0))
axes = axes.flatten()

for idx in range(4):
    if idx == 2:
        continue
    
    ax = axes[idx if idx < 2 else idx - 1]
    
    # Get data for this Sigma
    x_data = overall_meantrueTs[idx, :]
    x_err = overall_errortrueTs[idx, :]
    
    # Series 1: |true - predicted|
    y1_data = overall_meanpredictedTs[idx, :] - overall_meantrueTs[idx, :]
    y1_err = np.sqrt(overall_errortrueTs[idx, :]**2 + overall_errorpredictedTs[idx, :]**2)
    
    # Series 2: |true - suspiciousness|
    y2_data = overall_mean_true_suspiciousness_tensions[idx, :] - overall_meantrueTs[idx, :]
    y2_err = np.sqrt(overall_errortrueTs[idx, :]**2 + overall_error_true_suspiciousness_tensions[idx, :]**2)
    
    # Plot series
    ax.errorbar(x_data, y1_data, xerr=x_err, yerr=y1_err, fmt='o', label='NRE Predicted Tension', capsize=3, color='C0', markersize=5)
    ax.errorbar(x_data, y2_data, xerr=x_err, yerr=y2_err, fmt='o', label='Bayesian Suspiciousness', capsize=3, color='#ff7f0e', markersize=5)
    
    # Add dashed gray line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Add Sigma label in top right
    ax.text(0.95, 0.95, rf'$\Sigma = {Sigmas[idx]}\mathcal{{I}}$', 
            transform=ax.transAxes, ha='right', va='top')
    
    # Add legend only to first subplot
    if idx == 0:
        ax.legend(loc='upper left')
        ax.set_ylim(-0.75, 0.44)
    if idx == 1:
        ax.set_ylim(-0.98, 0.64)
        ax.set_ylabel(r'Observed $-$ True Tension ($N\sigma$)')
    if idx == 3:
        ax.set_ylim(-1.6, 0.24)
        ax.set_xlabel(r'True Tension ($N\sigma$)')
    
    ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xticks([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25], minor=True)
    if idx != 0:
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=False, bottom=True, left=True)
    else:
        ax.tick_params(axis='both', which='both', direction='in', top=False, right=False, bottom=True, left=True)
    ax.set_xlim(0.3,3.4)

plt.tight_layout()
plt.savefig(config.FIGURES + '/figure9.pdf', dpi=300, bbox_inches='tight')
