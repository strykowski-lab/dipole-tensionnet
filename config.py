"""Project-wide paths and constants.

Every script imports its paths from here, so the whole pipeline can be pointed
somewhere else (e.g. HPC scratch) by setting two environment variables before
running a script:

    DIPOLE_TENSIONNET_DATA     directory holding the raw inputs  (default: data/)
    DIPOLE_TENSIONNET_OUTPUTS  directory for everything produced (default: outputs/)
"""

import os

# -------- top-level paths --------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('DIPOLE_TENSIONNET_DATA', os.path.join(ROOT, 'data'))
OUTPUTS = os.environ.get('DIPOLE_TENSIONNET_OUTPUTS', os.path.join(ROOT, 'outputs'))
FIGURES = os.path.join(OUTPUTS, 'figures')

# -------- raw inputs (all from the Dropbox linked in the README) --------
PLANCK_FILE = os.path.join(DATA, 'BP_030_IQU_n0512_v2.fits')
PLANCK_FOREGROUND_FILE = os.path.join(DATA, 'LFI_CompMap_Foregrounds-smica-030_R3.00.fits')
RACS_CSV = os.path.join(DATA, 'AS110_Derived_Catalogue_racs_dr1_sources_galacticcut_v2021_08_v02_5725.csv')
NVSS_DAT = os.path.join(DATA, 'full_NVSS_combined_named.dat')
LOCAL_SOURCES_CSV = os.path.join(DATA, 'local_sources_ned_2mrs.csv')  # 2MRS + NED local sources (z < 0.1 removed from the radio samples)
RACS_MASK_FILE = os.path.join(DATA, 'racs_mask.npy')      # RACS-low footprint mask (equatorial pixels)
NVSS_MASK_FILE = os.path.join(DATA, 'nvss_mask.npy')      # NVSS footprint mask (equatorial pixels)
CATWISE_FITS = os.path.join(DATA, 'catwise_agns.fits')    # CatWISE AGN catalogue (Secrest et al. 2022)
CATWISE_EXCLUDE_FITS = os.path.join(DATA, 'exclude_master_revised.fits')  # CatWISE flagged regions (Secrest et al. 2022)

# -------- observed log R and Bayesian suspiciousness (shipped with this repo) --------
# R_{A}_{B}.npy / errorR_{A}_{B}.npy: observed log R between the real data sets,
# with errorR stored as a 2 sigma error (scripts divide by 2 to get 1 sigma).
R_DIR = os.path.join(DATA, 'observed_logR')
# {A}_{B}_suspiciousness_stats.npy: [T, +err, -err] from Land-Strykowski et al. (2025)
SUSPICIOUSNESS_DIR = os.path.join(DATA, 'suspiciousness')

# -------- simulations (HPC-grade: ~73 GB per data set at full size) --------
SIMULATION_DIR = os.path.join(OUTPUTS, 'simulations') + '/'

# -------- validation: nested sampling ground truth (Planck-RACS-low only) --------
VALIDATION_CHAINS_DIR = os.path.join(OUTPUTS, 'validation', 'chains')        # one job{N}/ directory per 02a job
VALIDATION_LOGR_DIR = os.path.join(OUTPUTS, 'validation', 'logR') + '/'       # log R of each pair (02b)
VALIDATION_DIR = os.path.join(OUTPUTS, 'validation', 'distribution') + '/'    # ground-truth distribution (02c)

# -------- Optuna hyperparameter searches (optional) --------
OPTUNA_DIR = os.path.join(OUTPUTS, 'optuna')  # search1/, search2/ and the paired simulations they read

# -------- NRE outputs, one directory per data set combination --------
def nre_dir(FIRST_dataset, SECOND_dataset):
    return os.path.join(OUTPUTS, 'nre', f'{FIRST_dataset}_{SECOND_dataset}') + '/'

# -------- toy problem (suspiciousness vs. in concordance R tension) --------
TOY_PROBLEM_DIR = os.path.join(OUTPUTS, 'toy_problem') + '/'

# -------- scratch space for training (node SSD on the HPC) --------
def jobfs_dir():
    """Fast local disk to copy the paired simulations onto before training.

    On Gadi this is /jobfs/<PBS_JOBID>/; elsewhere it is outputs/paired_simulations/.
    """
    if os.environ.get('PBS_JOBID') is not None:
        return f"/jobfs/{os.environ.get('PBS_JOBID')}/paired_simulations/"
    return os.path.join(OUTPUTS, 'paired_simulations') + '/'

# -------- data set combinations used in the paper --------
DATASETS = ['planck', 'racs', 'nvss', 'catwise', 'catsim']
COMBINATIONS = [
    ('planck', 'racs'), ('planck', 'nvss'), ('planck', 'catwise'),
    ('racs', 'nvss'), ('racs', 'catwise'), ('nvss', 'catwise'),
    ('planck', 'catsim'), ('racs', 'catsim'), ('nvss', 'catsim'),
]

# -------- simulation / training sizes used in the paper --------
N_CHUNKS = int(20*4.5)       # 90 chunks of simulations
N_SAMPLES = int(200000*4.5)  # 900 000 simulation pairs (10% concordant, 60% training, 30% validation)
N_RUNS = 5                   # number of NREs (each trained from fresh random weights) per combination

# -------- sanity checks on the NRE in concordance log R distribution --------
# If an NRE's in concordance distribution fails any of these, its weights are
# discarded and a new NRE is trained in its place (see 04_train_nre.py).
MIN_PEAK_DENSITY = 0.13  # below this the distribution is too broad (likely catastrophic initial weights)
MAX_PEAK_DENSITY = 0.25  # above this the distribution is too narrow (likely mode collapse during training)
# The median must also be >= 0: an in concordance distribution cannot mostly be negative
