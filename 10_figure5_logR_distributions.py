"""10_figure5_logR_distributions.py: Figure 5 - ensemble in concordance log R distributions for the combinations of Planck, NVSS, RACS-low and CatWISE.

Fast (a minute: KDEs of 90 000 points per panel). Reads, for each combination,
    outputs/nre/{first}_{second}/predictions/ensemble_concordance.npy (05_ensemble.py)
    data/observed_logR/R_{first}_{second}.npy and errorR_{first}_{second}.npy (observed log R)
prints the tension T and concordance C of each, and writes outputs/figures/figure5.pdf.
Planck-CatWISE uses a fitted Pearson type III distribution for T (observed log R is in the far tail).
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import scipy as sp
from tensionnet.utils import calcualte_stats
from scipy.stats import ecdf
import matplotlib as mpl
import os
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

combinations = [['nvss','catwise'], ['planck','catwise'], ['planck','nvss'],
                ['planck', 'racs'], ['racs','catwise'], ['racs','nvss']]

Rs = []
errorRs = []
Ts = []
T_lower_errors = []
T_upper_errors = []
rs = []
for combination in combinations:
    FIRST_dataset = combination[0]
    SECOND_dataset = combination[1]

    BASE_DIR = config.nre_dir(FIRST_dataset, SECOND_dataset)

    R_DIR = config.R_DIR
    R = np.load(f'{R_DIR}/R_{FIRST_dataset}_{SECOND_dataset}.npy')
    errorR = np.load(f'{R_DIR}/errorR_{FIRST_dataset}_{SECOND_dataset}.npy')/2 # convert to 1 sigma error
    print('Using R = ', R, ' +/- ', errorR)

    r = np.load(f'{BASE_DIR}predictions/ensemble_concordance.npy')
    mask = np.isfinite(r)

    Rs.append(R)
    errorRs.append(errorR)
    rs.append(r[mask])

    # --- Tension T and concordance C of the observed log R (Pearson type III fit for the far-tail Planck case) ---
    if combination == combinations[1]:
        class PearsonCDF:
            def __init__(self, mu, sigma, skew):
                self.dist = sp.stats.pearson3(skew, loc=mu, scale=sigma)
                self.cdf = self  # for "pearson.cdf.evaluate(x)" compatibility
            
            def evaluate(self, x):
                return self.dist.cdf(x)

        mu = np.mean(r[mask])
        std = np.std(r[mask])
        skew = sp.stats.skew(r[mask])
        kurt = sp.stats.kurtosis(r[mask])

        c = PearsonCDF(mu, std, skew)
        kde_pearson = sp.stats.pearson3(skew, loc=mu, scale=std)
    else:
        rsort  = np.sort(r[mask])
        c = ecdf(rsort)

    sigmaD = []
    sigmaA = []
    stats = calcualte_stats(R, errorR, c)
    sigmaD.append(stats[:3])
    sigmaA.append(stats[3:6])
    sigmaA = np.array(sigmaA)
    sigmaD = np.array(sigmaD)
    mean_sigmaA = sigmaA[:, 0].mean()
    mean_sigmaD = sigmaD[:, 0].mean()
    sigmaAs = sigmaA[:, 0]
    sigmaA_lower = sigmaA[:, 0] - sigmaA[:, 2]
    sigmaA_upper = sigmaA[:, 1] - sigmaA[:, 0]
    sigmaDs = sigmaD[:, 0]
    sigmaD_lower = sigmaD[:, 0] - sigmaD[:, 1]
    sigmaD_upper = sigmaD[:, 2] - sigmaD[:, 0]
    norm_sigmaA = sigmaAs / mean_sigmaA
    norm_sigmaD = sigmaDs / mean_sigmaD
    lower_mean_sigmaD_error = 1/np.sqrt(1) * np.sqrt(np.sum((sigmaD_lower)**2))
    upper_mean_sigmaD_error = 1/np.sqrt(1) * np.sqrt(np.sum((sigmaD_upper)**2))
    lower_mean_sigmaA_error = 1/np.sqrt(1) * np.sqrt(np.sum((sigmaA_lower)**2))
    upper_mean_sigmaA_error = 1/np.sqrt(1) * np.sqrt(np.sum((sigmaA_upper)**2))
    print(f'Dataset combination: {FIRST_dataset}--{SECOND_dataset}')
    print(f"Tension: {mean_sigmaD:.3f} (+{upper_mean_sigmaD_error:.3f}/-{lower_mean_sigmaD_error:.3f})")
    print(f"Concordance: {mean_sigmaA:.3f} (+{upper_mean_sigmaA_error:.3f}/-{lower_mean_sigmaA_error:.3f})")
    print('---------------------------------')
    Ts.append(mean_sigmaD)
    T_lower_errors.append(lower_mean_sigmaD_error)
    T_upper_errors.append(upper_mean_sigmaD_error)

combination_labels = [r'NVSS--CatWISE', r'\textit{Planck}--CatWISE', r'\textit{Planck}--NVSS',
                      r'\textit{Planck}--RACS-low', r'RACS-low--CatWISE', r'NVSS--RACS-low']

colors = ['black', 'tomato', 'cornflowerblue', 'limegreen', 'gold', 'orchid']

lower_bound = np.nanmin(rs)
upper_bound = np.nanmax(rs)

indices = [1, 4, 3, 2, 5, 0]
combination_labels = [combination_labels[i] for i in indices]
rs = [rs[i] for i in indices]
Rs = [Rs[i] for i in indices]
errorRs = [errorRs[i] for i in indices]
Ts = [Ts[i] for i in indices]
T_lower_errors = [T_lower_errors[i] for i in indices]
T_upper_errors = [T_upper_errors[i] for i in indices]

# --- Plot one panel (KDE of the in concordance distribution, observed log R and T) per combination ---
scale = 3.6/4.1
scale_y = 1.101
hspace = 0.1
fig, axes = plt.subplots(len(combination_labels), 1, figsize=(scale * 4.1, scale * 1.75 * len(combination_labels) * scale_y), sharex=True)
fig.subplots_adjust(hspace=hspace)
for i, (label, ax, color, T, T_lower_error, T_upper_error) in enumerate(zip(combination_labels, axes, colors, Ts, T_lower_errors, T_upper_errors)):
    kde = sp.stats.gaussian_kde(rs[i])
    xgrid = np.linspace(lower_bound, upper_bound, 1000)
    pdf_vals = kde.evaluate(xgrid)
    ax.plot(xgrid, pdf_vals, color=color, label=label, linewidth=1.5)
    ax.fill_between(xgrid, pdf_vals, color=color, alpha=0.2)
    ax.axvline(Rs[i], ls='--', c=color, linewidth=1.5)
    ax.axvspan(Rs[i] - errorRs[i], Rs[i] + errorRs[i], alpha=0.1, color=color)
    ax.set_xlim(-9.9, 18)
    ax.set_ylim(-0.0003, None)
    if label == r'\textit{Planck}--CatWISE':
        ax.plot(xgrid, kde_pearson.pdf(xgrid), label='Pearson Fit', color=color, ls=':', alpha=0.7, linewidth=1.5)
    if np.round(float(T_upper_error),2) == np.round(float(T_lower_error),2):
        ax.plot([], color='white', alpha=0.0, label=fr'$T = {float(T):.2f} \pm {float(T_upper_error):.2f}$')
    else:
        ax.plot([], color='white', alpha=0.0, label=fr'$T = {float(T):.2f} ^{{+{float(T_upper_error):.2f}}}_{{-{float(T_lower_error):.2f}}}$')
    if label == r'\textit{Planck}--NVSS':
        ax.set_ylabel('Probability Density Function (PDF)')
    else:
        ax.set_ylabel('')
    ax.yaxis.set_label_coords(-0.11, 1.05)
    # Add legend with white box
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='white', framealpha=0.8)

fig.subplots_adjust(hspace=hspace)
axes[-1].set_xlabel(r'$\log R$')
plt.savefig(config.FIGURES + '/figure5.pdf', bbox_inches='tight')
plt.close()