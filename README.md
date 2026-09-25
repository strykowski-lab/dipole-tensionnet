# dipole-tensionnet

Companion code for `Simulation-based tension quantification of the cosmic
dipole` ([Land-Strykowski et al. 2026, MNRAS, 551, stag1455](https://doi.org/10.1093/mnras/stag1455)). The repository
contains the code to run the analysis and make Figures 2–9 of the paper.
Figure 1 (the schematic of the NRE) was drawn in TikZ and is not made here.

The analysis trains neural ratio estimators (NREs, "tensionnets") on simulated
Healpix skies to learn the in concordance log R distribution between
*Planck*, NVSS, RACS-low and CatWISE (with the ecliptic bias correction, or
forward-modelled with CatSIM under the Eddington bias interpretation), and uses
it to turn the observed log R into a tension T.

## Pipeline

| Step | Scripts | Needed for |
|---|---|---|
| 1. Simulations | `01a_make_simulations.py`, `01b_make_catsim_simulations.py` | Everything |
| 2. Validation (nested sampling ground truth) | `02a_validation_nested_sampling.py`, `02b_validation_logR.py`, `02c_validation_distribution.py` | Figure 2 (and the ground-truth tension in Figure 3), and the Optuna searches |
| 3. Optuna hyperparameter searches (optional) | `03a_optuna_search1.py`, `03b_optuna_search2.py` | Choosing the NRE hyperparameters |
| 4. NREs | `04_train_nre.py` | Figures 2–8 |
| 5. Ensemble | `05_ensemble.py` | Figures 2–8 |
| 6. Toy problem | `06a_toy_problem_seed_finder.py`, `06b_toy_problem.py` | Figure 9 |
| 7. Figures | `07_figure2_nre_vs_ns.py` … `14_figure9_toy_problem.py` | |

The validation (step 2) only needs to be run to make Figure 2 (and the
ground-truth tension in Figure 3) and to run the Optuna searches, which score
each trial against the ground truth.

The Optuna searches (step 3) are optional: they are not needed to train the
NREs or to run anything else in this repository. `04_train_nre.py` already
defaults to the hyperparameters chosen from these searches for the paper. The
searches are included for completeness, so the whole process can be repeated
from scratch if you wish: run them, review their outputs, and then pass your
chosen hyperparameters to `04_train_nre.py`.

## HPC-grade scripts

Most of the pipeline cannot be run on a laptop at the size used in the paper.
These scripts are marked **HPC-grade** at the top of the file, and each has a
PBS job script in `pbs/` (written for NCI Gadi):

| Script | Work at paper size | PBS script |
|---|---|---|
| `01a_make_simulations.py` | 900 000 skies per data set (~73 GB each) | `make_simulations.pbs` |
| `01b_make_catsim_simulations.py` | 900 000 CatSIM skies, 10 jobs × 48 CPUs | `make_catsim_simulations.pbs` |
| `02a_validation_nested_sampling.py` | 30 000 UltraNest runs, 20 jobs | `validation_nested_sampling.pbs` |
| `02b_validation_logR.py` | reading 30 000 runs, 18 CPUs / 192 GB | `validation_logR.pbs` |
| `03a_optuna_search1.py`, `03b_optuna_search2.py` | 43 and 369 NREs, 4 GPUs | `optuna_search.pbs` |
| `04_train_nre.py` | 5+ NREs per combination, 4 GPUs, up to 48 h each | `train_nre.pbs` |
| `06b_toy_problem.py` | 140 small NREs on 10⁶ pairs each (days on a CPU) | `toy_problem.pbs` |

The remaining scripts (the ensemble, the ground-truth distribution and every
figure) take seconds to minutes. The HPC-grade scripts also take size arguments
(`--nsamples`, `--epochs`, `--runs-per-script`, ...; see `--help`), so they can
be run at a small size on a laptop.

## Layout

```
data/
  observed_logR/              Observed log R between the real data sets (in git)
  suspiciousness/             Bayesian suspiciousness tensions from L25 (in git)
outputs/
  simulations/                Simulated skies (01a, 01b)
  validation/                 Nested sampling ground truth (02a-c)
    chains/                     UltraNest runs of each simulation pair
    logR/                       log R of each simulation pair
    distribution/               Ground-truth in concordance log R distribution
  optuna/                     Optuna hyperparameter searches (03a, 03b)
    search1/                    One directory per trial
    search2/                    One directory per trial, plus the Optuna study
  nre/
    {first}_{second}/         Trained NREs, predictions and statistics (04, 05)
  toy_problem/                Toy problem (06a, 06b)
  figures/                    figure2.pdf ... figure9.pdf
pbs/                          PBS job scripts for the HPC-grade steps
config.py                     Paths and settings shared by every script
utils.py                      Shared helper functions
catwise_utils.py              Helpers from Secrest et al. (2022) for the CatWISE selection
nnhealpix_patch/              NNhealpix layers with the masked average pooling layer
xmatch/                       Cross-matching package (removes local sources)
```

All paths come from `config.py`. To keep the (large) outputs somewhere else,
e.g. on HPC scratch, set `DIPOLE_TENSIONNET_OUTPUTS` (and
`DIPOLE_TENSIONNET_DATA` for the inputs) before running a script.

## Setup

```
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# NNhealpix (Krachmalnicoff & Tomasi 2019), plus the masked average pooling layer used by the tensionnet
.venv/bin/pip install --no-build-isolation "git+https://github.com/ai4cmb/NNhealpix.git@24b99db0c223e325e33a4edb1e6f282e60f6d9fe"
.venv/bin/python 00_install_nnhealpix_patch.py

# tensionnet (Bevins et al. 2025), for calcualte_stats and the toy problem NRE
git clone https://github.com/htjb/tension-networks.git ../tension-networks
echo "$(cd ../tension-networks && pwd)" > .venv/lib/python3.11/site-packages/tension-networks.pth
```

`tension-networks/tensionnet/utils.py` imports `pypolychord` on its first
lines, which is only used by functions this repository does not call. Either
install [PolyChordLite](https://github.com/PolyChord/PolyChordLite) or comment
out that import.

On a GPU node, install `tensorflow[and-cuda]==2.16.2` instead of
`tensorflow==2.16.2` (the paper used CUDA 12.3 and cuDNN 8.9).

**CatSIM** (only needed for `01b_make_catsim_simulations.py`) requires
[catsim](https://github.com/o-oayda/catsim) (Oayda et al. 2026), which needs
numpy ≥ 2 and so lives in its own environment:

```
python3.12 -m venv .venv-catsim
.venv-catsim/bin/pip install "git+https://github.com/o-oayda/catsim.git"
```

## Data

All the inputs are the same as for
[dipole-tensions](https://github.com/strykowski-lab/dipole-tensions) (Land-Strykowski
et al. 2025, hereafter L25), and can be downloaded from
[this Dropbox link](https://www.dropbox.com/scl/fo/1w4cxux8glbyvxsnvmd5m/AM-6RFPFJO5woLqGHg-xGRM?rlkey=adlpn128qthddsh26srvg5ocu&st=k5uo4277&dl=0).
Place these files in `data/`:

```
BP_030_IQU_n0512_v2.fits                        BeyondPlanck 30 GHz map
LFI_CompMap_Foregrounds-smica-030_R3.00.fits    Planck SMICA 30 GHz foregrounds
AS110_Derived_Catalogue_racs_dr1_sources_galacticcut_v2021_08_v02_5725.csv   RACS-low (CASDA, project AS110)
full_NVSS_combined_named.dat                    NVSS
local_sources_ned_2mrs.csv                      2MRS + NED local sources (removed from the radio samples)
racs_mask.npy, nvss_mask.npy                    RACS-low and NVSS footprint masks
catwise_agns.fits                               CatWISE AGN catalogue (Secrest et al. 2022)
exclude_master_revised.fits                     CatWISE flagged regions (Secrest et al. 2022)
```

The samples are built from these exactly as in L25: RACS-low and NVSS with
15 mJy < S < 1000 mJy and local sources (z < 0.1) removed, and CatWISE with the
Secrest et al. (2022) selection.

These small inputs are included in the repository:

```
observed_logR/R_{first}_{second}.npy            Observed log R (errorR_*.npy hold 2 sigma errors)
suspiciousness/{first}_{second}_suspiciousness_stats.npy
                                                Bayesian suspiciousness tension [T, +err, -err]
```

The observed log R and suspiciousness values for *Planck*, NVSS, RACS-low and
CatWISE come from the nested sampling analysis in L25
([dipole-tensions](https://github.com/strykowski-lab/dipole-tensions)), so they do
not need to be recomputed here. For CatSIM, the evidence of CatWISE under the
Eddington bias interpretation (and jointly with each other data set) comes from
the neural likelihood estimator of Oayda et al. (2026,
[arXiv:2602.05070](https://arxiv.org/abs/2602.05070)), applied here to the S22
CatWISE sample (rather than S21 as in that paper), combined with the L25
evidences of *Planck*, NVSS and RACS-low.

## Run order

Each script reads what the steps above it wrote. The HPC-grade scripts skip
work that is already on disk, so they can be resubmitted if a job runs out of
walltime.

```
# 0. Once, after setup
python 00_install_nnhealpix_patch.py

# 1. Simulations (HPC-grade). The theta parameters are shared by every data set and
#    are made by the first job, and the concordant skies are shared by every combination,
#    so run Planck first and the roles in this order:
python 01a_make_simulations.py --dataset planck  --role first
python 01a_make_simulations.py --dataset racs    --role second
python 01a_make_simulations.py --dataset nvss    --role second
python 01a_make_simulations.py --dataset catwise --role second
python 01a_make_simulations.py --dataset racs    --role first
python 01a_make_simulations.py --dataset nvss    --role first
SCRIPT_NUMBER=0 python 01b_make_catsim_simulations.py   # ... up to SCRIPT_NUMBER=9

# 2. Validation: nested sampling ground truth for Planck-RACS-low (HPC-grade, then fast).
#    Only needed for Figure 2 (and the ground-truth tension in Figure 3) and the Optuna searches.
SCRIPT_NUMBER=0 python 02a_validation_nested_sampling.py      # ... up to SCRIPT_NUMBER=19
python 02b_validation_logR.py
python 02c_validation_distribution.py

# 3. Optional: Optuna hyperparameter searches, Appendix B (HPC-grade; need step 2).
#    Review their outputs in outputs/optuna/search1/ and outputs/optuna/search2/
#    and choose the NRE hyperparameters for step 4.
python 03a_optuna_search1.py
python 03b_optuna_search2.py

# 4. Train the NREs (HPC-grade), for each of the nine combinations
#    planck-racs, planck-nvss, planck-catwise, racs-nvss, racs-catwise, nvss-catwise,
#    planck-catsim, racs-catsim, nvss-catsim. The hyperparameters default to the ones
#    chosen from the Optuna searches for the paper, and can be changed, e.g.
#    --layers 4 --neurons 256 --dropout 0.1 --activation gelu --learning-rate 5e-4
python 04_train_nre.py --first planck --second racs

# 5. Ensemble of the NREs (fast), for each combination
python 05_ensemble.py --first planck --second racs

# 6. Toy problem, Appendix C (06b is HPC-grade)
python 06a_toy_problem_seed_finder.py
python 06b_toy_problem.py

# 7. Figures (fast)
python 07_figure2_nre_vs_ns.py
python 08_figure3_planck_racs_tension.py
python 09_figure4_tension_comparison.py
python 10_figure5_logR_distributions.py
python 11_figure6_tension_network.py
python 12_figure7_tension_comparison_catsim.py
python 13_figure8_logR_distributions_catsim.py
python 14_figure9_toy_problem.py
```

### Training the NREs (`04_train_nre.py`)

The NRE hyperparameters are command-line arguments (`--nside-out`, `--filters`,
`--layers`, `--neurons`, `--dropout`, `--activation`, `--learning-rate`,
`--decay-steps`, `--decay-rate`). Their defaults are the architecture chosen from
the Optuna searches and used in the paper (Section 3 and Appendix B), so the
searches do not need to be rerun. To repeat the process from scratch, run the
searches, review their outputs, and set these arguments accordingly.

Five NREs are trained per combination, each from fresh random weights. After
each NRE is trained, its in concordance log R distribution is checked, and the
NRE is discarded and replaced by a newly trained one if the distribution is
unphysical:

* peak probability density < 0.13: too broad (likely catastrophic initial weights),
* peak probability density > 0.25: too narrow (likely mode collapse during training),
* median log R < 0: mostly negative, which an in concordance distribution cannot be.

Training continues until five NREs pass. Each discarded NRE is recorded in
`discarded_runs.txt` in the combination's output directory.

## Citation

If you use any code from this repository, please cite both this paper and the
tensionnet formalism paper (Bevins et al. 2025) that it builds on:

```bibtex
@article{10.1093/mnras/stag1455,
    author = {Land-Strykowski, Mali and Bevins, Harry T J and Oayda, Oliver T and Lewis, Geraint F},
    title = {Simulation-based tension quantification of the cosmic dipole},
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {551},
    number = {1},
    pages = {stag1455},
    year = {2026},
    month = {09},
    doi = {10.1093/mnras/stag1455},
    url = {https://doi.org/10.1093/mnras/stag1455},
}

@article{10.1103/47hn-h8kc,
  author = {Bevins, Harry T. J. and Handley, William J. and Gessey-Jones, Thomas},
  title = {Calibrating Bayesian tension statistics using neural ratio estimators},
  journal = {Phys. Rev. D},
  volume = {112},
  issue = {4},
  pages = {043520},
  numpages = {15},
  year = {2025},
  month = {Aug},
  doi = {10.1103/47hn-h8kc},
  url = {https://link.aps.org/doi/10.1103/47hn-h8kc}
}

```

If you use the observed log R or Bayesian suspiciousness values, please also
cite L25:

```bibtex
@article{10.1093/mnras/staf1621,
    author = {Land-Strykowski, Mali and Lewis, Geraint F and Murphy, Tara},
    title = {Cosmic dipole tensions: confronting the cosmic microwave background with infrared and radio populations of cosmological sources},
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {543},
    number = {4},
    pages = {3229-3241},
    year = {2025},
    month = {11},
    doi = {10.1093/mnras/staf1621},
    url = {https://doi.org/10.1093/mnras/staf1621},
}
```

## License

MIT
