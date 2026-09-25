"""01a_make_simulations.py: kinematic dipole simulations for Planck, RACS-low, NVSS and CatWISE.

*** HPC-GRADE *** At the paper's size (900 000 simulations per data set) this
takes many CPU-hours and ~73 GB of disk per data set. See pbs/make_simulations.pbs.

The data set is chosen with --dataset, and whether it is the FIRST or SECOND
data set of a combination with --role.

What it does:
    1. Loads the real data set to get the quantities the simulations need
       (mean counts per pixel, D_cmb for the survey, mask, Planck noise level).
    2. Makes (or loads) the shared theta parameters (seeded, so identical for
       every data set): theta_concordant, theta_1 (FIRST data sets), theta_2
       (SECOND data sets) and the matched/mismatched labels.
    3. Simulates the in concordance chunks  -> {dataset}_concordant/chunkN.npy
       and the mixed (training + validation) chunks -> {dataset}_mixed_{role}/chunkN.npy

Roles used in the paper:
    planck  : first  (planck_racs, planck_nvss, planck_catwise, planck_catsim)
    racs    : second (planck_racs) and first (racs_nvss, racs_catwise, racs_catsim)
    nvss    : second (planck_nvss, racs_nvss) and first (nvss_catwise, nvss_catsim)
    catwise : second
    catsim  : second (see 01b_make_catsim_simulations.py)

Note: the concordant chunks of a data set are shared by all of its combinations
and are only made once, by whichever role is run first, so run the roles in the
order listed in the README.

Usage:
    python 01a_make_simulations.py --dataset planck --role first
    python 01a_make_simulations.py --dataset racs --role second
"""

# Import packages
print('Importing packages...')
import argparse
import os
import numpy as np
from tqdm import tqdm
import healpy as hp
import scipy as sp
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import config
from utils import fluxcut, crossmatch, binsources, applymask, ang2vec, r2d, equatorial_mask_to_galactic
from catwise_utils import SkyMap

# ---------------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--dataset', required=True, choices=['planck', 'racs', 'nvss', 'catwise'])
parser.add_argument('--role', required=True, choices=['first', 'second'],
                    help='whether this data set is the FIRST or SECOND data set of the combination')
parser.add_argument('--nsamples', type=int, default=config.N_SAMPLES, help='total number of simulations (paper: 900000)')
parser.add_argument('--nchunks', type=int, default=config.N_CHUNKS, help='number of chunks to split them into (paper: 90)')
args = parser.parse_args()

if args.dataset == 'planck' and args.role != 'first':
    parser.error('Planck is always the FIRST data set.')
if args.dataset == 'catwise' and args.role != 'second':
    parser.error('CatWISE is always the SECOND data set.')

# Simulation settings
nChunks = args.nchunks
nSamples = args.nsamples
dtype = np.float32 # np.float32
DATASET = args.dataset
ROLE = args.role
nside_in = 64

# Calculate chunk sizes
chunk_size = nSamples // nChunks
concordant_chunks = int(nChunks * 0.1)
training_chunks = int(nChunks * 0.6)
test_chunks = nChunks - concordant_chunks - training_chunks
mixed_chunks = training_chunks + test_chunks

# Set directories
SIMULATION_DIR = config.SIMULATION_DIR
if not os.path.exists(SIMULATION_DIR):
    os.makedirs(SIMULATION_DIR)
dataset_dir_mixed = SIMULATION_DIR + DATASET + f'_mixed_{ROLE}'
if not os.path.exists(dataset_dir_mixed):
    os.makedirs(dataset_dir_mixed)
dataset_dir_concordant = SIMULATION_DIR + DATASET + '_concordant'
if not os.path.exists(dataset_dir_concordant):
    os.makedirs(dataset_dir_concordant)

# ---------------------------------------------------------------------------
# Load the real data set to set up the simulations
# ---------------------------------------------------------------------------
# The FIRST data set uses D_cmb_D1/monopole_D1 and the SECOND uses D_cmb_D2/monopole_D2.
# Only one data set is simulated here,
# so the monopole of the other one is set to 1 (its value does not matter).
def load_datasets():
    global D_cmb_D1, monopole_D1, std_D1, D_cmb_D2, monopole_D2, MASK, cecl, catwise_ecl_lat_unmasked, pos_nside_in
    D_cmb_D1, monopole_D1, D_cmb_D2, monopole_D2 = None, 1, None, 1
    pos_nside_in = np.array(hp.pix2vec(nside_in, np.arange(hp.nside2npix(nside_in)))).T

    if DATASET == 'planck':
        # PLANCK: BeyondPlanck 30 GHz map with the SMICA foreground removed, in K
        nside = 64
        freq = 30  # 30, 44, 70
        lead = 'LFI' if freq < 100 else 'HFI'
        end = '0512' if freq < 70 else '1024'
        freq = str(freq)
        freq = freq.zfill(3)
        T_cmb = 2.7255
        file_path = config.PLANCK_FILE
        hdul = fits.open(file_path)
        data_loaded = hdul[1].data['I_MEAN']
        counts = data_loaded.flatten() * 1e-6  # convert from microK to K
        file_path = config.PLANCK_FOREGROUND_FILE  # load the foreground data
        hdul = fits.open(file_path)
        data_loaded = hdul[1].data['INTENSITY']
        intcounts = data_loaded.flatten()
        intcounts = hp.ud_grade(intcounts, nside_out=hp.get_nside(counts))
        realcounts = counts - intcounts
        realcounts += T_cmb
        realcounts = hp.ud_grade(realcounts, nside_out=64)
        nside = hp.get_nside(realcounts)
        npix = hp.nside2npix(nside)
        pos = np.array(hp.pix2vec(nside, np.arange(npix))).T
        data_planck = realcounts, pos
        c = sp.constants.c
        vel_cmb = 369.82e3
        D_cmb_D1 = vel_cmb / c # expected kinematic dipole amplitude for the CMB (beta = v_cmb / c), as in Land-Strykowski et al. (2025; dipole-tensions)
        monopole_D1 = np.mean(data_planck[0])
        std_D1 = np.std(data_planck[0])
        MASK = None # Planck is not masked

    if DATASET == 'racs':
        # RACS-low (sample B): 15 mJy < S < 1000 mJy, local sources (z < 0.1) removed,
        # binned at nside 64, galactic mask applied
        LOCAL_SOURCES = pd.read_csv(config.LOCAL_SOURCES_CSV)
        LOCAL_SOURCES['LS_id'] = LOCAL_SOURCES['LS_id'].astype(object)
        LOCAL_labels = {'ra':'ra','dec':'dec','id':'LS_id'}
        RACS_SOURCES = pd.read_csv(config.RACS_CSV)
        RACS_MASK = equatorial_mask_to_galactic(np.load(config.RACS_MASK_FILE).astype(bool)) # footprint mask, in galactic pixels
        RACS_labels = {'ra':'ra','dec':'dec','id':'source_id'}
        RACS_flux = 'total_flux_source'
        RACS_cut = fluxcut(RACS_SOURCES, RACS_flux, 15, 1000).reset_index(drop=True)
        RACS_matched = crossmatch(LOCAL_SOURCES, RACS_cut, LOCAL_labels, RACS_labels, radius=5, z=0.1)
        RACS_binned = binsources(RACS_matched, 64)
        RACS_masked = applymask(RACS_binned, RACS_MASK)
        monopole = np.mean(RACS_masked[0])
        D_cmb = 0.00427 # expected kinematic dipole amplitude for RACS-low (sample B), from the analysis in Land-Strykowski et al. (2025; dipole-tensions)
        MASK = RACS_MASK

    if DATASET == 'nvss':
        # NVSS (sample B): 15 mJy < S < 1000 mJy, local sources (z < 0.1) removed,
        # binned at nside 64, galactic mask applied
        LOCAL_SOURCES = pd.read_csv(config.LOCAL_SOURCES_CSV)
        LOCAL_SOURCES['LS_id'] = LOCAL_SOURCES['LS_id'].astype(object)
        LOCAL_labels = {'ra':'ra','dec':'dec','id':'LS_id'}
        NVSS_SOURCES = pd.read_csv(config.NVSS_DAT, sep=r'\s+', header=None)
        NVSS_SOURCES.columns = NVSS_SOURCES.iloc[0] # Set the first row as the header
        NVSS_SOURCES = NVSS_SOURCES[1:] # Remove the first row
        NVSS_SOURCES.reset_index(drop=True, inplace=True)
        NVSS_MASK = equatorial_mask_to_galactic(np.load(config.NVSS_MASK_FILE).astype(bool)) # footprint mask, in galactic pixels
        NVSS_labels = {'ra':'ra','dec':'dec','id':'source_name'}
        NVSS_flux = 'integrated_flux'
        NVSS_cut = fluxcut(NVSS_SOURCES, NVSS_flux, 15, 1000).reset_index(drop=True)
        NVSS_matched = crossmatch(LOCAL_SOURCES, NVSS_cut, LOCAL_labels, NVSS_labels, radius=5, z=0.1)
        NVSS_binned = binsources(NVSS_matched, 64)
        NVSS_masked = applymask(NVSS_binned, NVSS_MASK)
        monopole = np.mean(NVSS_masked[0])
        D_cmb = 0.00431 # expected kinematic dipole amplitude for NVSS (sample B), from the analysis in Land-Strykowski et al. (2025; dipole-tensions)
        MASK = NVSS_MASK

    if DATASET == 'catwise':
        # CatWISE
        # Secrest et al. (2022) selection, as in Land-Strykowski et al. (2025; dipole-tensions):
        # W1 coverage >= 80, zero-coverage areas and flagged regions masked, W1 < 16.5,
        # |b| < 30 deg masked, sources binned at nside 64 in galactic coordinates
        sm = SkyMap(nside=64, frame='galactic')
        w1cut = 16.5
        tbl = sm.load_tbl(config.CATWISE_FITS)
        tbl = tbl[tbl['w1cov'] >= 80]
        sm.mask_zeros(tbl) # Get zero-coverage areas of full map
        sm.fits2mask(config.CATWISE_EXCLUDE_FITS) # Modify mask with flagged regions
        tbl = tbl[tbl['w1'] < w1cut] # Make sky map on flux density cut
        hpm = sm.mk_hpmap(tbl, extra_keys=['ebv', 'Tb'])
        sm.set_mask(np.abs(hpm['b']) < 30) # Galactic plane cut
        CATWISE_MASK = ~sm.get_mask() # True = pixel is used
        CATWISE_binned = np.bincount(hp.ang2pix(64, np.array(tbl['l']), np.array(tbl['b']), lonlat=True), minlength=hp.nside2npix(64)).astype(np.float64)
        # Ecliptic latitude of every pixel, for the linear ecliptic bias correction
        theta, phi = hp.pix2ang(64, np.arange(hp.nside2npix(64)))
        lon, lat = r2d(phi), r2d(np.pi/2-theta)
        galactic_coords = SkyCoord(lon*u.deg, lat*u.deg, frame='galactic')
        ecliptic_coords = galactic_coords.transform_to('barycentricmeanecliptic')
        catwise_ecl_lat_unmasked = ecliptic_coords.lat.deg
        cecl = 9.15*10**-4 # ecliptic bias coefficient: 7.4 for S21, 9.15 for S22 (Secrest et al. 2022), as in Land-Strykowski et al. (2025; dipole-tensions)
        pos = np.array(hp.pix2vec(64, np.arange(hp.nside2npix(64)))).T
        CATWISE_masked = CATWISE_binned[CATWISE_MASK], pos[CATWISE_MASK]
        monopole = np.mean(CATWISE_masked[0])
        D_cmb = 0.00725 # expected kinematic dipole amplitude for CatWISE, from the analysis in Land-Strykowski et al. (2025; dipole-tensions)
        MASK = CATWISE_MASK

    # Survey data sets: store as data set 1 or data set 2 depending on the role
    if DATASET != 'planck':
        if ROLE == 'first':
            D_cmb_D1, monopole_D1 = D_cmb, monopole
        else:
            D_cmb_D2, monopole_D2 = D_cmb, monopole

# ---------------------------------------------------------------------------
# Priors (unit cube -> physical parameters)
# ---------------------------------------------------------------------------
def ptform(u):
    u_v, u_theta, u_phi, u_NA, u_NB = u
    v = u_v * 20                   # v / v_cmb, between 0 and 20
    theta = np.arccos(2*u_theta-1) # isotropic dipole direction
    phi = (2*np.pi)*u_phi
    NA = monopole_D1 * (u_NA * 0.2 + 0.9) # mean counts of data set 1, +/- 10%
    NB = monopole_D2 * (u_NB * 0.2 + 0.9) # mean counts of data set 2, +/- 10%
    return np.array((v,theta,phi,NA,NB))

def catwise_ptform(u):
    u_bias = u
    bias = u_bias * 4 - 2 # ecliptic bias parameter, between -2 and 2
    return np.array((bias))

def nre_prior(u_array, ptfunc=ptform):
    return np.array([ptfunc(u) for u in u_array])

def random_variables(nSamples, N):
    return np.array([np.random.uniform(size=N) for _ in range(nSamples)])

# ---------------------------------------------------------------------------
# Shared theta parameters: matched (label 1) and mismatched (label 0) pairs
# ---------------------------------------------------------------------------
def generate_thetas(nSamples):
    np.random.seed(42)
    theta_all = random_variables(nSamples, 5)
    theta_1 = theta_all
    theta_2 = np.copy(theta_all)
    concordant_samples = int(nSamples * 0.1)
    mixed_samples = nSamples - concordant_samples
    # Mismatch the second half of the mixed samples with a derangement (no sample keeps its own theta)
    idx = np.arange(mixed_samples//2)
    while True:
        np.random.shuffle(idx)
        if not np.any(idx == np.arange(len(idx))):
            break
    theta_2[concordant_samples:][mixed_samples//2:] = theta_2[concordant_samples:][mixed_samples//2:][idx]
    # Shuffle the matched and mismatched pairs together, keeping track of the labels
    idx = np.arange(mixed_samples)
    np.random.shuffle(idx)
    labels = np.concatenate((np.ones(mixed_samples//2), np.zeros(mixed_samples//2)))
    labels = labels[idx]
    theta_1[concordant_samples:] = theta_1[concordant_samples:][idx]
    theta_2[concordant_samples:] = theta_2[concordant_samples:][idx]
    return theta_1[:concordant_samples].astype(dtype), theta_1[concordant_samples:].astype(dtype), theta_2[concordant_samples:].astype(dtype), labels.astype(dtype)

loaded = False
def load_thetas():
    global loaded, theta_concordant, theta_1, theta_2
    if not loaded:
        load_datasets()
        if not (os.path.exists(SIMULATION_DIR + 'theta_concordant.npy') and
                os.path.exists(SIMULATION_DIR + 'theta_1.npy') and
                os.path.exists(SIMULATION_DIR + 'theta_2.npy') and
                os.path.exists(SIMULATION_DIR + 'labels.npy')):
            print('Generating new theta parameters for simulations...')
            theta_concordant, theta_1, theta_2, labels = generate_thetas(nSamples)
            np.save(SIMULATION_DIR + 'theta_concordant.npy', theta_concordant)
            np.save(SIMULATION_DIR + 'theta_1.npy', theta_1)
            np.save(SIMULATION_DIR + 'theta_2.npy', theta_2)
            np.save(SIMULATION_DIR + 'labels.npy', labels)
        else:
            print('Loading existing theta parameters for simulations...')
            theta_concordant = np.load(SIMULATION_DIR + 'theta_concordant.npy')
            theta_1 = np.load(SIMULATION_DIR + 'theta_1.npy')
            theta_2 = np.load(SIMULATION_DIR + 'theta_2.npy')
        loaded = True

# ---------------------------------------------------------------------------
# Simulators: one sky per theta, lambda_i = N (1 + D cos theta_i)
# ---------------------------------------------------------------------------
def mock_planck(x):
    v, theta, phi, N, _ = x
    D = v*D_cmb_D1
    expected_number_density_list = N*(1+(D*np.sum(ang2vec(theta,phi)*pos_nside_in, axis=1)))
    data_counts = np.random.normal(expected_number_density_list, std_D1)
    data_counts = data_counts*(10**6) # convert from K to microK
    return data_counts.astype(dtype)

def mock_racs(x):
    v, theta, phi, NA, NB = x
    D, N = (v*D_cmb_D1, NA) if ROLE == 'first' else (v*D_cmb_D2, NB)
    expected_number_density_list = N*(1+(D*np.sum(ang2vec(theta,phi)*pos_nside_in, axis=1)))
    data_counts = np.random.poisson(expected_number_density_list)
    data_counts[~MASK] = 0 # masked pixels are set to zero
    return data_counts.astype(dtype)

def mock_nvss(x):
    v, theta, phi, NA, NB = x
    D, N = (v*D_cmb_D1, NA) if ROLE == 'first' else (v*D_cmb_D2, NB)
    expected_number_density_list = N*(1+(D*np.sum(ang2vec(theta,phi)*pos_nside_in, axis=1)))
    data_counts = np.random.poisson(expected_number_density_list)
    data_counts[~MASK] = 0 # masked pixels are set to zero
    return data_counts.astype(dtype)

def mock_catwise(x):
    v, theta, phi, _, N, bias = x
    D = v*D_cmb_D2
    fecl = 1 - bias * cecl * np.abs(catwise_ecl_lat_unmasked) # linear ecliptic bias correction
    expected_number_density_list = fecl*N*(1+(D*np.sum(ang2vec(theta,phi)*pos_nside_in, axis=1)))
    data_counts = np.random.poisson(expected_number_density_list)
    data_counts[~MASK] = 0 # masked pixels are set to zero
    return data_counts.astype(dtype)

mock_functions = {'planck': mock_planck, 'racs': mock_racs, 'nvss': mock_nvss, 'catwise': mock_catwise}
generator = mock_functions[DATASET]

def generate_simulations(generator, theta, savedir, extra_theta=None, extra_ptfunc=None):
    np.random.seed(42*2)
    theta = nre_prior(theta)
    if extra_theta is not None:
        extra_theta = nre_prior(extra_theta, extra_ptfunc)
        theta = np.concatenate((theta, extra_theta), axis=1)
    for i in tqdm(range(len(theta)//chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < len(theta)//chunk_size - 1 else len(theta)
        t = theta[start:end]
        simulations = []
        for j in tqdm(range(len(t))):
            tt = t[j]
            simulation = generator(tt)
            simulations.append(simulation)
        simulations = np.array(simulations).astype(dtype)
        np.save(savedir + f'/chunk{i+1}.npy', simulations)

# ---------------------------------------------------------------------------
# Make the simulations that are missing
# ---------------------------------------------------------------------------
# In concordance simulations (shared by every combination this data set is in)
files = [dataset_dir_concordant + f'/chunk{idx+1}.npy' for idx in range(concordant_chunks)]
if not all(os.path.exists(f) for f in files):
    load_thetas()
    print(f'Generating new concordant simulations for {DATASET}...')
    if DATASET == 'catwise':
        np.random.seed(42*3)
        catwise_theta = random_variables(concordant_chunks*chunk_size, 1)
        generate_simulations(generator, theta_concordant, dataset_dir_concordant, catwise_theta, catwise_ptform)
    else:
        generate_simulations(generator, theta_concordant, dataset_dir_concordant)

# Mixed (matched + mismatched) simulations for training and validation
files = [dataset_dir_mixed + f'/chunk{idx+1}.npy' for idx in range(mixed_chunks)]
if not all(os.path.exists(f) for f in files):
    load_thetas()
    print(f'Generating new mixed simulations for {DATASET} ({ROLE} data set)...')
    theta_mixed = theta_1 if ROLE == 'first' else theta_2
    if DATASET == 'catwise':
        np.random.seed(42*4)
        catwise_theta = random_variables(mixed_chunks*chunk_size, 1)
        generate_simulations(generator, theta_mixed, dataset_dir_mixed, catwise_theta, catwise_ptform)
    else:
        generate_simulations(generator, theta_mixed, dataset_dir_mixed)

print(f'All {DATASET} ({ROLE}) simulations are present.')
