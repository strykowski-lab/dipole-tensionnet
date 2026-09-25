"""02b_validation_logR.py: log R of every nested-sampled Planck-RACS simulation pair.

Validation: only needed to make Figure 2 (and the ground-truth tension in
Figure 3) and to run the optional Optuna searches.

*** HPC-GRADE *** Reads 30 000 UltraNest runs with anesthetic, in parallel over
the CPUs of one node (18 CPUs, 192 GB in the paper; see pbs/validation_logR.pbs).

For every pair i:
    log R_i    = log Z_joint - log Z_planck - log Z_racs
    log R_err_i = sqrt(err_planck^2 + err_racs^2 + err_joint^2) / 2   (1 sigma, [+, -])
saved as logR_{i}.npy and logRerr_{i}.npy in outputs/validation/logR/.

Usage:
    python 02b_validation_logR.py [--n-pairs 10000] [--n-scripts 20]
"""

# Import necessary packages and functions
import argparse
import os
import tarfile
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import config
from utils import getRstats

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--n-pairs', type=int, default=10000, help='number of nested-sampled pairs (paper: 10000)')
parser.add_argument('--n-scripts', type=int, default=20, help='number of 02a jobs whose outputs to read (paper: 20)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Gather the UltraNest runs of every 02a job into one directory
# ---------------------------------------------------------------------------
if os.environ.get('PBS_JOBID') is not None:
    # On the HPC: extract all tar files to the node SSD
    CHAINS_DIR = f'/jobfs/{os.environ.get("PBS_JOBID")}/'
    print('Extracting tar files to the node SSD...')
    for SCRIPT_NUMBER in range(args.n_scripts):
        TAR_FILE = os.path.join(config.VALIDATION_CHAINS_DIR, f'job{SCRIPT_NUMBER}.tar.gz')
        with tarfile.open(TAR_FILE, 'r:gz') as tar:
            tar.extractall(CHAINS_DIR)
        print(f'Extracted {TAR_FILE} to {CHAINS_DIR}')
    chain_dirs = [CHAINS_DIR]
else:
    # Locally: read straight from each job's output directory
    chain_dirs = [os.path.join(config.VALIDATION_CHAINS_DIR, f'job{SCRIPT_NUMBER}') + '/' for SCRIPT_NUMBER in range(args.n_scripts)]
    chain_dirs = [d for d in chain_dirs if os.path.exists(d)]

def find_run(name):
    for d in chain_dirs:
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    raise FileNotFoundError(f'{name} not found in {chain_dirs}')

def computeR(A=None,B=None):
    logZ_A,logZ_Aerr = getRstats(find_run(A))
    logZ_B,logZ_Berr = getRstats(find_run(B))
    logZ_AB,logZ_ABerr = getRstats(find_run(A+'+'+B))
    logR = logZ_AB - logZ_A - logZ_B
    logRerr = (logZ_Aerr**2+logZ_Berr**2+logZ_ABerr**2)**(1/2)/2 # convert to 1 sigma error
    return logR, logRerr

# Set up the directory to save stats
statdir = config.VALIDATION_LOGR_DIR
if not os.path.exists(statdir):
    os.makedirs(statdir)

# Define parameters for the analysis
runs = np.arange(args.n_pairs)

# Prepare input data for multiprocessing (skip pairs already done)
input = np.array([])
for run in runs:
    if not (os.path.exists(statdir+f'/logR_{run}.npy') and os.path.exists(statdir+f'/logRerr_{run}.npy')):
        filename = f'plancksim_{run}+racssim_{run}'
        input = np.append(input, filename)

# Define the task function for multiprocessing
def task(string):
    string1, string2 = string.split('+')
    run = string1.split('_')[1]
    print(f'[Run {run}] Processing {string}...')
    logR, logRerr = computeR(string1, string2)
    np.save(os.path.join(statdir, f'logR_{run}.npy'), logR)
    np.save(os.path.join(statdir, f'logRerr_{run}.npy'), logRerr)
    print(f'[Run {run}] Finished {string}.')

# Run the multiprocessing workflow
if __name__ == '__main__':
    threads = max(1, int(os.environ.get("PBS_NCPUS", 3)) - 2)
    print(f'Using {threads} threads for multiprocessing...')
    print(f'Using {2} threads for background processes...')
    max_proccesses = threads
    with Pool(processes=threads) as pool:
        for i in tqdm(range(0, len(input), max_proccesses)):
            chunk = input[i:i + max_proccesses]
            pool.map(task, chunk)
