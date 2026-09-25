"""01b_make_catsim_simulations.py: CatSIM (Eddington bias) forward-modelled CatWISE simulations.

*** HPC-GRADE *** Each job makes 9 chunks of 10 000 CatSIM skies on 48 CPUs
(see pbs/make_catsim_simulations.pbs). Ten jobs are needed: SCRIPT_NUMBER=0
makes the in concordance chunks and SCRIPT_NUMBER=1..9 make the 81 mixed chunks.

Requires the CatSIM simulator (https://github.com/o-oayda/catsim), see
Oayda et al. (2026, https://arxiv.org/abs/2602.05070). Here it is used with the
Secrest et al. (2022, S22) CatWISE mask and selection.

It reuses the shared theta parameters written by 01a_make_simulations.py
(run `--dataset planck --role first` beforehand), so CatSIM is always the
SECOND data set and uses theta_concordant / theta_2 like the other surveys.
The extra CatSIM parameter is the extra W1 photometric error (0 to 8).

Usage:
    SCRIPT_NUMBER=0 python 01b_make_catsim_simulations.py
    python 01b_make_catsim_simulations.py --script-number 3
"""

import argparse
import os
import healpy as hp
import numpy as np
import config
from catsim import CatwiseConfig, Catwise, batch_simulate, prng_key

# ---------------------------------------------------------------------------
# Which part of the simulations this job makes (0 = concordant, 1-9 = mixed)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--script-number', type=int, default=os.environ.get('SCRIPT_NUMBER'),
                    help='0 for the concordant chunks, 1-9 for the mixed chunks (default: $SCRIPT_NUMBER)')
parser.add_argument('--nsamples', type=int, default=config.N_SAMPLES)
parser.add_argument('--nchunks', type=int, default=config.N_CHUNKS)
parser.add_argument('--n-workers', type=int, default=48)
args = parser.parse_args()
if args.script_number is None:
    parser.error('set --script-number or the SCRIPT_NUMBER environment variable')

#################
SCRIPT_NUMBER = int(args.script_number)
print('='*40)
print(f'SCRIPT_NUMBER = {SCRIPT_NUMBER}')
print('='*40)
#################

# Simulation settings
nChunks = args.nchunks
nSamples = args.nsamples
dtype = np.float32 # np.float32
SECOND_dataset = 'catsim'

chunk_size = nSamples // nChunks
concordant_chunks = int(nChunks * 0.1)
training_chunks = int(nChunks * 0.6)
test_chunks = nChunks - concordant_chunks - training_chunks
mixed_chunks = training_chunks + test_chunks
chunks_per_script = mixed_chunks // 9 # 9 mixed chunks per job in the paper

# Set directories
SIMULATION_DIR = config.SIMULATION_DIR
if not os.path.exists(SIMULATION_DIR):
    os.makedirs(SIMULATION_DIR)
SECOND_dataset_dir_mixed = SIMULATION_DIR + SECOND_dataset + '_mixed_second'
if not os.path.exists(SECOND_dataset_dir_mixed):
    os.makedirs(SECOND_dataset_dir_mixed)
SECOND_dataset_dir_concordant = SIMULATION_DIR + SECOND_dataset + '_concordant'
if not os.path.exists(SECOND_dataset_dir_concordant):
    os.makedirs(SECOND_dataset_dir_concordant)

# Conversion array (CatSIM maps are in NESTED ordering, everything else is RING)
nested_to_ring = hp.reorder(np.arange(hp.nside2npix(64)), n2r=True)

# ---------------------------------------------------------------------------
# Priors (unit cube -> CatSIM parameters)
# ---------------------------------------------------------------------------
def catsim_ptform(u):
    u_extra_err = u
    extra_err = u_extra_err * 8
    return np.array((extra_err))

def ptform(u):
    u_v, u_theta, u_phi, u_NA, u_NB = u
    v = u_v * 20
    theta = np.arccos(2*u_theta-1)
    phi = (2*np.pi)*u_phi
    NA = 1 # value does not matter
    NB = 7 + np.log10(u_NB + 3) # log10_n_initial_samples
    return np.array((v,theta,phi,NA,NB))

def nre_prior(u_array, ptfunc=ptform):
    return np.array([ptfunc(u) for u in u_array])

def random_variables(nSamples, N):
    return np.array([np.random.uniform(size=N) for _ in range(nSamples)])

# ---------------------------------------------------------------------------
# Set up the CatSIM simulator (S22 selection)
# ---------------------------------------------------------------------------
rng_key = prng_key(42)
catsim_config = CatwiseConfig(
    cat_w12_min=0.5,
    cat_w1_max=17.0,
    magnitude_error_dist='gaussian',
    base_mask_version='S22',
    use_float32=True,
    use_common_extra_error=True
)
sim = Catwise(catsim_config)
sim.initialise_data()
simulator_function = sim.generate_dipole

# Load extra u's for CatSIM extra error parameter
if not (os.path.exists(SIMULATION_DIR + 'catsim_theta_concordant.npy') and
        os.path.exists(SIMULATION_DIR + 'catsim_theta_mixed.npy')):
    np.random.seed(42*5)
    catsim_theta_concordant = random_variables(concordant_chunks*chunk_size, 1)
    np.save(SIMULATION_DIR + 'catsim_theta_concordant.npy', catsim_theta_concordant)
    np.random.seed(42*5)
    catsim_theta_mixed = random_variables(mixed_chunks*chunk_size, 1)
    np.save(SIMULATION_DIR + 'catsim_theta_mixed.npy', catsim_theta_mixed)
    del catsim_theta_concordant, catsim_theta_mixed

# Select thetas for this script
if SCRIPT_NUMBER == 0:
    extra_theta = np.load(SIMULATION_DIR + 'catsim_theta_concordant.npy')
    full_theta = np.load(SIMULATION_DIR + 'theta_concordant.npy')
else:
    extra_theta = np.load(SIMULATION_DIR + 'catsim_theta_mixed.npy')[(SCRIPT_NUMBER-1)*chunks_per_script*chunk_size:(SCRIPT_NUMBER)*chunks_per_script*chunk_size]
    full_theta = np.load(SIMULATION_DIR + 'theta_2.npy')[(SCRIPT_NUMBER-1)*chunks_per_script*chunk_size:(SCRIPT_NUMBER)*chunks_per_script*chunk_size]

chunks_to_make = extra_theta.shape[0] // chunk_size
w1_max = np.full(chunk_size, 16.5)

# ---------------------------------------------------------------------------
# Generate simulations in batches of chunk_size (10 000)
# ---------------------------------------------------------------------------
for chunk in range(chunks_to_make):
    # Setup params
    chunk_theta = nre_prior(full_theta[chunk*chunk_size:(chunk+1)*chunk_size])
    chunk_extra_theta = nre_prior(extra_theta[chunk*chunk_size:(chunk+1)*chunk_size], catsim_ptform)
    v, theta, phi, _, N = chunk_theta.T
    extra_err = chunk_extra_theta.T[0]

    # Run simulations
    batch_theta = {
        'w1_max': w1_max,
        'log10_n_initial_samples': N,
        'observer_speed': v,
        'dipole_longitude': np.degrees(phi),
        'dipole_latitude': np.degrees(np.pi/2 - theta),
        'w1_extra_error': extra_err,
    }
    dmap, _ = batch_simulate(
        theta=batch_theta,
        model_callable=simulator_function,
        n_workers=args.n_workers,
        rng_key=rng_key
    )

    # Convert (masked NaN pixels -> 0, NESTED -> RING) and save simulations
    dmap = np.where(np.isnan(dmap), 0, dmap)
    dmap = dmap[:, nested_to_ring]
    if SCRIPT_NUMBER == 0:
        np.save(f"{SECOND_dataset_dir_concordant}/chunk{chunk+1}.npy", dmap.astype(dtype))
    else:
        np.save(f"{SECOND_dataset_dir_mixed}/chunk{(SCRIPT_NUMBER-1)*chunks_to_make+chunk+1}.npy", dmap.astype(dtype))
    del dmap
