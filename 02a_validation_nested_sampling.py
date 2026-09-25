"""02a_validation_nested_sampling.py: nested sampling of in concordance Planck-RACS simulation pairs.

Validation: only needed to make Figure 2 (and the ground-truth tension in
Figure 3) and to run the optional Optuna searches.

*** HPC-GRADE *** The ground-truth in concordance log R distribution needs
30 000 UltraNest runs (10 000 pairs x [Planck, RACS-low, joint]). In the paper
this was split over 20 jobs (SCRIPT_NUMBER=0..19) of 500 pairs each, 4 CPUs and
up to 48 hours per job (see pbs/validation_nested_sampling.pbs).

It uses the first in concordance chunk of the Planck and RACS-low simulations
made by 01a_make_simulations.py, i.e. the same 10 000 pairs whose predicted
log R is compared with the truth in Figure 2.

For every pair i it runs:
    plancksim_i                 : Planck alone   (Gaussian likelihood)
    racssim_i                   : RACS-low alone (Poisson likelihood, masked)
    plancksim_i+racssim_i       : joint fit with a shared dipole (v, theta, phi)
and log R_i = log Z_joint - log Z_planck - log Z_racs is computed by 02b.

Usage:
    SCRIPT_NUMBER=0 python 02a_validation_nested_sampling.py
    python 02a_validation_nested_sampling.py --script-number 0
"""

# Import packages / functions
import argparse
import os
import tarfile
import healpy as hp
import numpy as np
import scipy as sp
import ultranest
from tqdm import tqdm
import config
from utils import ang2vec, r2d, equatorial_mask_to_galactic

# ---------------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--script-number', type=int, default=os.environ.get('SCRIPT_NUMBER'),
                    help='which block of pairs to run (default: $SCRIPT_NUMBER)')
parser.add_argument('--runs-per-script', type=int, default=500, help='pairs per job (paper: 500, with 20 jobs)')
parser.add_argument('--nlive', type=int, default=None, help='UltraNest min_num_live_points (paper: UltraNest default, 400)')
args = parser.parse_args()
if args.script_number is None:
    parser.error('set --script-number or the SCRIPT_NUMBER environment variable')

# ---------------------------------------------------------------------------
# Ultranest function: fit one data set
# ---------------------------------------------------------------------------
def run_single(data, D_cmb_survey, savedir=None, name=None, output=False, type=None):
    # process data
    data_counts, data_positions = data
    N_data = np.mean(data_counts) # sources/px
    param_names = ['v', 'theta', 'phi', 'N'] # note: convert to D/Dcmb,b,l after sampling complete

    if type == 'poisson':
        def loglike(x):
            # 'theta' = dipole amplitude, dipole colatitude, dipole longitude
            v, theta, phi, N = x

            D = v*D_cmb_survey

            expected_number_density_list = N*(1+(D*np.sum(ang2vec(theta,phi)*data_positions, axis=1)))

            return np.sum(sp.stats.poisson.logpmf(data_counts,expected_number_density_list))

    if type == 'gaussian':
        std = np.std(data_counts)
        def loglike(x):
            # 'theta' = dipole amplitude, dipole colatitude, dipole longitude
            v, theta, phi, N = x

            D = v*D_cmb_survey

            expected_number_density_list = N*(1+(D*np.sum(ang2vec(theta,phi)*data_positions, axis=1)))

            return np.sum(sp.stats.norm.logpdf(data_counts,expected_number_density_list,std))

    def ptform(u):
        u_v, u_theta, u_phi, u_N = u

        N = N_data * (u_N * 0.2 + 0.9) # this is the mean +/- 10%

        v = u_v * 20  # between 0 and 20 (it is v/v_cmb)

        theta = np.arccos(2*u_theta-1)

        phi = (2*np.pi)*u_phi

        return np.array((v,theta,phi,N))

    # "Ultranest" nested sampling.
    sampler = ultranest.ReactiveNestedSampler(param_names, loglike, ptform, log_dir=savedir)
    if args.nlive is None:
        sampler.run()
    else:
        sampler.run(min_num_live_points=args.nlive)

    # name the sampler run
    if savedir is not None and name is not None:
        os.rename(savedir+'/run1', savedir+'/'+name)

    if output:
        sampler.print_results()

        # convert from theta,phi to b,l
        results = sampler.results
        results['samples'][:,1] = 90-r2d(results['samples'][:,1])
        results['samples'][:,2] = r2d(results['samples'][:,2])

        return results

# ---------------------------------------------------------------------------
# Joint ultranest function: fit two data sets with a shared dipole
# ---------------------------------------------------------------------------
def run_joint(dataA, D_cmb_surveyA, dataB, D_cmb_surveyB, savedir=None, name=None, output=False, type1=None, type2=None):
    data_countsA, data_positionsA = dataA
    data_countsB, data_positionsB = dataB
    N_dataA = np.mean(data_countsA) # sources/px
    N_dataB = np.mean(data_countsB) # sources/px
    param_names = ['v', 'theta', 'phi', 'NA', 'NB'] # note: convert to D/Dcmb,b,l after sampling complete

    if type1 == 'poisson':
        def loglikeA(x):
            # 'theta' = dipole amplitude, dipole colatitude, dipole longitude
            v, theta, phi, NA, _ = x

            DA = v*D_cmb_surveyA

            expected_number_density_listA = NA*(1+(DA*np.sum(ang2vec(theta,phi)*data_positionsA, axis=1)))

            return np.sum(sp.stats.poisson.logpmf(data_countsA,expected_number_density_listA))
    if type1 == 'gaussian':
        stdA = np.std(data_countsA)
        def loglikeA(x):
            # 'theta' = dipole amplitude, dipole colatitude, dipole longitude
            v, theta, phi, NA, _ = x

            DA = v*D_cmb_surveyA

            expected_number_density_listA = NA*(1+(DA*np.sum(ang2vec(theta,phi)*data_positionsA, axis=1)))

            return np.sum(sp.stats.norm.logpdf(data_countsA,expected_number_density_listA,stdA))

    if type2 == 'poisson':
        def loglikeB(x):
            # 'theta' = dipole amplitude, dipole colatitude, dipole longitude
            v, theta, phi, _, NB = x

            DB = v*D_cmb_surveyB

            expected_number_density_listB = NB*(1+(DB*np.sum(ang2vec(theta,phi)*data_positionsB, axis=1)))

            return np.sum(sp.stats.poisson.logpmf(data_countsB,expected_number_density_listB))
    if type2 == 'gaussian':
        stdB = np.std(data_countsB)
        def loglikeB(x):
            # 'theta' = dipole amplitude, dipole colatitude, dipole longitude
            v, theta, phi, _, NB = x

            DB = v*D_cmb_surveyB

            expected_number_density_listB = NB*(1+(DB*np.sum(ang2vec(theta,phi)*data_positionsB, axis=1)))

            return np.sum(sp.stats.norm.logpdf(data_countsB,expected_number_density_listB,stdB))

    def loglike(x):
        return loglikeA(x) + loglikeB(x)

    # LOG PRIOR

    def ptform(u):
        u_v, u_theta, u_phi, u_NA, u_NB = u

        NA = N_dataA * (u_NA * 0.2 + 0.9) # this is the mean +/- 10%

        NB = N_dataB * (u_NB * 0.2 + 0.9) # this is the mean +/- 10%

        v = u_v * 20

        theta = np.arccos(2*u_theta-1)

        phi = (2*np.pi)*u_phi

        return np.array((v,theta,phi,NA,NB))

    # "Ultranest" nested sampling.
    sampler = ultranest.ReactiveNestedSampler(param_names, loglike, ptform, log_dir=savedir)
    if args.nlive is None:
        sampler.run()
    else:
        sampler.run(min_num_live_points=args.nlive)

    # name the sampler run
    if savedir is not None and name is not None:
        os.rename(savedir+'/run1', savedir+'/'+name)

    if output:
        sampler.print_results()

        # convert from theta,phi to b,l
        results = sampler.results
        results['samples'][:,1] = 90-r2d(results['samples'][:,1])
        results['samples'][:,2] = r2d(results['samples'][:,2])

        return results

#################
SCRIPT_NUMBER = int(args.script_number)
print('='*40)
print(f'SCRIPT_NUMBER = {SCRIPT_NUMBER}')
print('='*40)
#################

# Set D_cmb for different surveys (expected kinematic dipole amplitudes, from the analysis in
# Land-Strykowski et al. 2025; dipole-tensions)
D_cmb_D1 = 369.82e3/sp.constants.c # Planck: beta = v_cmb / c
D_cmb_D2 = 0.00427 # RACS-low (sample B)

# Set savedir: node SSD on the HPC (compressed and moved at the end), otherwise straight to outputs/
SAVE_DIR = f'job{SCRIPT_NUMBER}'
if os.environ.get('PBS_JOBID') is not None:
    WORK_DIR = f"/jobfs/{os.environ.get('PBS_JOBID')}/{SAVE_DIR}/"
else:
    WORK_DIR = os.path.join(config.VALIDATION_CHAINS_DIR, SAVE_DIR) + '/'
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)

# Setup runs
runlist = np.arange(10000)
runs = runlist[args.runs_per_script*SCRIPT_NUMBER:args.runs_per_script+args.runs_per_script*SCRIPT_NUMBER]

# Load concordant data chunk
data_chunk1 = np.load(config.SIMULATION_DIR + 'planck_concordant/chunk1.npy')
data_chunk2 = np.load(config.SIMULATION_DIR + 'racs_concordant/chunk1.npy')

# Load mask
RACS_MASK = equatorial_mask_to_galactic(np.load(config.RACS_MASK_FILE).astype(bool)) # footprint mask, in galactic pixels

# ---------------------------------------------------------------------------
# Run sampler: Planck alone, RACS-low alone, and jointly, for every pair
# ---------------------------------------------------------------------------
pos = np.array(hp.pix2vec(64, np.arange(49152))).T
for run in tqdm(runs):
    # Skip pairs that are already done (e.g. a resumed job)
    if os.path.exists(WORK_DIR + 'plancksim_'+str(run)+'+racssim_'+str(run)):
        continue

    # Select data
    dataset1 = data_chunk1[run]
    dataset2 = data_chunk2[run]

    dset1 = dataset1, pos
    dset2 = dataset2[RACS_MASK], pos[RACS_MASK]

    # Run individual
    run_single(dset1, D_cmb_D1, savedir=WORK_DIR, name='plancksim_'+str(run), type='gaussian')
    run_single(dset2, D_cmb_D2, savedir=WORK_DIR, name='racssim_'+str(run), type='poisson')

    # Run joint
    run_joint(dset1, D_cmb_D1, dset2, D_cmb_D2, savedir=WORK_DIR, name='plancksim_'+str(run)+'+racssim_'+str(run), type1='gaussian', type2='poisson')

# Compress the node SSD directory as tar and move to scratch
if os.environ.get('PBS_JOBID') is not None:
    print('Compressing and moving to scratch...')
    os.makedirs(config.VALIDATION_CHAINS_DIR, exist_ok=True)
    with tarfile.open(os.path.join(config.VALIDATION_CHAINS_DIR, SAVE_DIR + '.tar.gz'), 'w:gz') as tar:
        tar.add(WORK_DIR, arcname=os.path.basename(WORK_DIR)) # the run directories sit at the top level of the tar
