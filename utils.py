# Shared helper functions used across the pipeline.

import os
import numpy as np
import healpy as hp
from anesthetic import read_chains
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.stats import gaussian_kde

################################################################################################################
# QUANTILES AND NESTED SAMPLING STATISTICS

def healpy_quantile(x, q=[0.025, 0.5, 0.975], weights=None):

    # note: [0.025, 0.5, 0.975] is 2sigma

    if weights is None:
        weights = np.ones(x.shape)

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    weights = np.atleast_1d(weights)
    idx = np.argsort(x)  # sort samples
    sw = weights[idx]  # sort weights
    cdf = np.cumsum(sw)[:-1]  # compute CDF
    cdf /= cdf[-1]  # normalize CDF
    cdf = np.append(0, cdf)  # ensure proper span
    quantiles = np.interp(q, cdf, x[idx]).tolist()
    return quantiles

def _quantile(samples):
    # 1std -> [0.16, 0.5, 0.84]
    # 2std -> [0.025, 0.5, 0.975]
    ql, qm, qh = healpy_quantile(samples, [0.025, 0.5, 0.975], weights=None)
    q_minus, q_plus = qm - ql, qh - qm
    return qm, np.array((q_plus, q_minus))

def get_disc_stats(dir,type='double'):
    nested_samples = read_chains(dir)
    nsamples = nested_samples.shape[0]
    bayesian_stats = nested_samples.stats(nsamples)
    if type == 'single':
        return nested_samples.logZ(), bayesian_stats['logZ'].std()*2, nested_samples.D_KL(), bayesian_stats['D_KL'].std()*2, nested_samples.d_G(), np.abs(bayesian_stats['d_G'].std()*2)
    elif type == 'double':
        return *_quantile(bayesian_stats['logZ']), *_quantile(bayesian_stats['D_KL']), *_quantile(bayesian_stats['d_G']),

def getRstats(dir):
    # Returns logZ and its (+, -) 2 sigma errors for one ultranest run directory
    nested_samples = read_chains(dir)
    nsamples = nested_samples.shape[0]
    bayesian_stats = nested_samples.stats(nsamples)
    return *_quantile(bayesian_stats['logZ']),

################################################################################################################
# CATALOGUE AND HEALPIX HELPERS

def fluxcut(sources, flux_col, flux_min, flux_max):
    internal_sources = sources.copy()
    internal_sources[flux_col] = internal_sources[flux_col].astype(np.float64)
    return internal_sources[(internal_sources[flux_col] >= flux_min) & (internal_sources[flux_col] <= flux_max)]

def binsources(sources, nside=64, plot=False):
    if isinstance(sources, pd.DataFrame):
        sources = sources.to_records(index=False)
    # Assuming ra and dec are in degrees
    ra = sources['ra'].astype(float)
    dec = sources['dec'].astype(float)

    # Create SkyCoord object
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    # Convert to galactic coordinates
    galactic_coords = coords.galactic

    # Extract l and b in degrees
    l = galactic_coords.l.degree
    b = galactic_coords.b.degree

    # bin the sources
    counts_galactic = np.zeros(hp.nside2npix(nside)) + np.bincount(hp.ang2pix(nside,l,b,lonlat=1), minlength=hp.nside2npix(nside))

    # Plot the healpix map
    if plot:
        hp.mollview(counts_galactic, title="Mollview", unit="Counts", coord=['G'])

    return counts_galactic

def applymask(map,mask,nside=64):
    # Returns (counts, positions) of the unmasked pixels only
    npix = hp.nside2npix(nside)
    pos = np.array(hp.pix2vec(nside, np.arange(npix))).T
    pos = pos[mask]
    counts = map[mask]
    data = np.copy(counts), np.copy(pos)
    return data

def equatorial_mask_to_galactic(mask_equatorial, nside=64):
    # The RACS-low and NVSS footprint masks are defined on equatorial (RA, Dec) pixels, but the
    # sources and simulations here are binned in galactic pixels. A galactic pixel is kept if
    # its centre falls in a kept equatorial pixel.
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    galactic_centres = SkyCoord(l=r2d(phi)*u.deg, b=r2d(np.pi/2-theta)*u.deg, frame='galactic').icrs
    equatorial_pixels = hp.ang2pix(nside, galactic_centres.ra.deg, galactic_centres.dec.deg, lonlat=True)
    return np.asarray(mask_equatorial, dtype=bool)[equatorial_pixels]

################################################################################################################
# CROSS-MATCHING FUNCTIONS (remove local sources from the radio catalogues, as in Land-Strykowski et al. 2025)

def _fixmatches(real_matches, B_labels, first=False):
    # Resolves duplicate cross-matches by keeping the closest local -> radio
    # separation and promoting backup duplicates. Called iteratively until no
    # duplicates remain.

    # Find duplicates in the ['B'][B_labels['id']] column
    duplicates = real_matches[real_matches.duplicated(subset=[('B', B_labels['id'])], keep=False)]
    non_duplicates = real_matches[~real_matches.duplicated(subset=[('B', B_labels['id'])], keep=False)]

    # For each duplicate in the B_labels['id'] column, ensure all same datatype (str)
    for idx, row in duplicates.iterrows():
        if isinstance(row[('B', B_labels['id'])], (np.str_, np.ndarray)):
            duplicates.at[idx, ('B', B_labels['id'])] = str(row[('B', B_labels['id'])])
    # For each duplicate in the B_labels['id'] column, find the local_labels['id'] with the smallest separation
    unique_ids = duplicates[('B', B_labels['id'])].unique()
    closest_matches = []
    for uid in unique_ids:
        subset = duplicates[duplicates[('B', B_labels['id'])] == uid]
        min_separation_idx = subset[('AB', 'separation')].idxmin()
        closest_matches.append(subset.loc[min_separation_idx])

    closest_matches = pd.DataFrame(closest_matches)

    # Remove closest matches from duplicates
    farthest_matches = duplicates[~duplicates.index.isin(closest_matches.index)]

    # Create a new dataframe of farthest_matches but only where ['AB']['duplicates'] is not None
    farthest_matches_with_backups = farthest_matches.dropna(subset=[('AB', 'duplicates')])

    # For each row in farthest_matches_with_backups, replace ['B'][B_labels['id']] with the first element of ['AB']['duplicates']
    farthest_matches_with_backups_fixed = farthest_matches_with_backups.copy()
    max_dupes = 0
    if not farthest_matches_with_backups_fixed.empty:
        for idx, row in farthest_matches_with_backups_fixed.iterrows():
            duplicates_val = row[('AB', 'duplicates')]
            distances = row[('AB', 'distances')]
            if distances.size == 1:
                try:
                    farthest_matches_with_backups_fixed.at[idx, ('AB', 'separation')] = distances[0]
                except Exception:
                    farthest_matches_with_backups_fixed.at[idx, ('AB', 'separation')] = distances
                farthest_matches_with_backups_fixed.at[idx, ('AB', 'distances')] = None
                try:
                    farthest_matches_with_backups_fixed.at[idx, ('B', B_labels['id'])] = duplicates_val[0]
                except Exception:
                    farthest_matches_with_backups_fixed.at[idx, ('B', B_labels['id'])] = duplicates_val
                farthest_matches_with_backups_fixed.at[idx, ('AB', 'duplicates')] = None
                max_dupes = max(max_dupes, 1)
            else:
                farthest_matches_with_backups_fixed.at[idx, ('AB', 'separation')] = distances[0]
                farthest_matches_with_backups_fixed.at[idx, ('AB', 'distances')] = distances[1:]
                farthest_matches_with_backups_fixed.at[idx, ('B', B_labels['id'])] = duplicates_val[0]
                farthest_matches_with_backups_fixed.at[idx, ('AB', 'duplicates')] = duplicates_val[1:]
                max_dupes = max(max_dupes, distances.size)

    # Combine non_duplicates, closest_matches, and farthest_matches_with_backups_fixed
    combined_matches = pd.concat([non_duplicates, closest_matches, farthest_matches_with_backups_fixed])

    # Reset index for the combined dataframe
    combined_matches.reset_index(drop=True, inplace=True)

    if not first:
        return combined_matches
    else:
        return combined_matches, max_dupes

def crossmatch(A_sources, B_sources, A_labels, B_labels, radius, z, returnz=False):
    # Cross-matches A (local) sources against B (radio catalogue) sources within
    # `radius` arcsec, using only local sources with redshift <= z, and returns the
    # B sources that were NOT matched (i.e. the catalogue with local sources removed).
    from xmatch import xmatch as _xmatch

    A_sources = A_sources[A_sources['z'] <= z]

    matches = _xmatch(A_sources, B_sources, A_labels, B_labels, radius)

    initial_matches = matches.dropna(subset=[('B', 'ra')])

    # Separate real_matches into the part with entries in AB duplicates, and the other part where AB duplicates is None
    matches_with_duplicates = initial_matches.dropna(subset=[('AB', 'duplicates')])
    matches_without_duplicates = initial_matches[initial_matches[('AB', 'duplicates')].isna()]

    # Sort the distances and duplicates in matches_with_duplicates and convert to arrays
    for idx, row in matches_with_duplicates.iterrows():
        distances_raw = row[('AB', 'distances')]
        ids_raw = row[('AB', 'duplicates')]
        distances = np.array(distances_raw.split(';'), dtype=np.float64)
        ids = ids_raw.split(';')
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_ids = np.array(ids)[sorted_indices]
        matches_with_duplicates.at[idx, ('AB', 'distances')] = sorted_distances
        matches_with_duplicates.at[idx, ('AB', 'duplicates')] = sorted_ids

    # Combine matches_with_duplicates and matches_without_duplicates
    real_matches = pd.concat([matches_with_duplicates, matches_without_duplicates])

    # Reset index for the combined dataframe
    real_matches.reset_index(drop=True, inplace=True)

    # Iteratively fix duplicates
    iterated_matches, num_iterations = _fixmatches(real_matches, B_labels, first=True)
    print(f"Number of iterations required: {num_iterations}")
    for i in range(num_iterations):
        iterated_matches = _fixmatches(iterated_matches, B_labels)

    # Verify
    if not iterated_matches[iterated_matches.duplicated(subset=[('B', B_labels['id'])], keep=False)].empty:
        print('There are still duplicates present.')
        fixed = False
        attempts = 0
        while not fixed:
            print('Fixing duplicates...')
            iterated_matches = _fixmatches(iterated_matches, B_labels)
            fixed = iterated_matches[iterated_matches.duplicated(subset=[('B', B_labels['id'])], keep=False)].empty
            attempts += 1
            if attempts > 10:
                raise ValueError("Too many attempts to fix duplicates.")

    if returnz:
        print('Done. Returning matches...')
        return iterated_matches

    print('Done. Removing sources...')

    # Remove sources
    final_sources = B_sources[~B_sources[B_labels['id']].isin(iterated_matches[('B', B_labels['id'])])]

    print('Done.')

    return final_sources

################################################################################################################
# COORDINATES

def ang2vec(theta, phi):
    sintheta = np.sin(theta)
    return np.array([np.cos(phi)*sintheta,np.sin(phi)*sintheta,np.cos(theta)])

def r2d(rad):
    return rad*180/np.pi
def d2r(deg):
    return deg*np.pi/180

################################################################################################################
# PAIRED SIMULATIONS FOR THE OPTUNA SEARCHES

def pair_simulations_for_optuna(out_dir, nSamples, nChunks, simulation_dir, full_nChunks,
                                FIRST_dataset='planck', SECOND_dataset='racs', nside_in=64):
    """Write Planck-RACS simulation pairs in the layout read by the Optuna searches.

    The searches read pre-paired chunks (one row = [FIRST sky, SECOND sky]):
        nside64_data_train_chunk{k}.npy  + nside64_labels_train_chunk{k}.npy
        nside64_data_test_chunk{k}.npy   + nside64_labels_test_chunk{k}.npy   (validation)
        nside64_data_concordant_chunk{k}.npy
    These are taken from the first chunks of the training, validation and in
    concordance simulations made by 01a_make_simulations.py (whose chunk size
    must be the same, nSamples // nChunks).
    """
    chunk_size = nSamples // nChunks
    concordant_chunks = int(nChunks * 0.1)
    training_chunks = int(nChunks * 0.6)
    test_chunks = nChunks - concordant_chunks - training_chunks
    full_training_chunks = int(full_nChunks * 0.6) # training chunks of the full simulation set

    files = ([out_dir + f'nside{nside_in}_data_train_chunk{k+1}.npy' for k in range(training_chunks)]
             + [out_dir + f'nside{nside_in}_data_test_chunk{k+1}.npy' for k in range(test_chunks)]
             + [out_dir + f'nside{nside_in}_data_concordant_chunk{k+1}.npy' for k in range(concordant_chunks)])
    if all(os.path.exists(f) for f in files):
        return
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    labels = np.load(simulation_dir + 'labels.npy', mmap_mode='r') # labels of the mixed simulations, in chunk order
    def pair(kind, chunk):
        first = np.load(simulation_dir + f'{FIRST_dataset}_{kind}' + ('_first' if kind == 'mixed' else '') + f'/chunk{chunk}.npy')
        second = np.load(simulation_dir + f'{SECOND_dataset}_{kind}' + ('_second' if kind == 'mixed' else '') + f'/chunk{chunk}.npy')
        if first.shape[0] != chunk_size:
            raise ValueError(f'simulation chunks hold {first.shape[0]} skies but this search expects {chunk_size}')
        return np.concatenate((first, second), axis=-1)
    for k in range(training_chunks):
        mixed_chunk = k + 1 # first training chunks
        np.save(out_dir + f'nside{nside_in}_data_train_chunk{k+1}.npy', pair('mixed', mixed_chunk))
        np.save(out_dir + f'nside{nside_in}_labels_train_chunk{k+1}.npy', np.array(labels[(mixed_chunk-1)*chunk_size:mixed_chunk*chunk_size]))
    for k in range(test_chunks):
        mixed_chunk = full_training_chunks + k + 1 # first validation chunks
        np.save(out_dir + f'nside{nside_in}_data_test_chunk{k+1}.npy', pair('mixed', mixed_chunk))
        np.save(out_dir + f'nside{nside_in}_labels_test_chunk{k+1}.npy', np.array(labels[(mixed_chunk-1)*chunk_size:mixed_chunk*chunk_size]))
    for k in range(concordant_chunks):
        np.save(out_dir + f'nside{nside_in}_data_concordant_chunk{k+1}.npy', pair('concordant', k + 1))

################################################################################################################
# NRE SANITY CHECK

def check_concordance_distribution(r, min_peak=0.13, max_peak=0.25, min_median=0.0):
    """Check that an NRE's in concordance log R distribution is physical.

    An in concordance log R distribution from a well-trained NRE peaks at a
    probability density between ~0.13 and ~0.25 (see Figures 2, 5 and 8 of the
    paper), and cannot have most of its mass at negative log R. We flag:
      * peak density < min_peak -> too broad  (likely catastrophic initial weights)
      * peak density > max_peak -> too narrow (likely mode collapse during training)
      * median < min_median (0) -> mostly negative (likely catastrophic initial weights)

    Returns (passed, peak_density, median, reason).
    """
    r = np.asarray(r).flatten()
    r = r[np.isfinite(r)]
    median = np.median(r)
    # Peak of the PDF, from a Gaussian KDE evaluated on a grid spanning the samples
    kde = gaussian_kde(r)
    xgrid = np.linspace(r.min(), r.max(), 1000)
    peak_density = np.max(kde.evaluate(xgrid))

    if peak_density < min_peak:
        return False, peak_density, median, f'peak density {peak_density:.3f} < {min_peak} (too broad)'
    if peak_density > max_peak:
        return False, peak_density, median, f'peak density {peak_density:.3f} > {max_peak} (too narrow)'
    if median < min_median:
        return False, peak_density, median, f'median log R {median:.3f} < {min_median} (mostly negative)'
    return True, peak_density, median, 'passed'
