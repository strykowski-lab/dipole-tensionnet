#!/usr/bin/env python
"""
This is used for the data prep of CatWISE, imported from Secrest+22.

HEALPix utilities for working with dipoles.

Mix of code written by N. Secrest and S. von Hausegger, with a few
modified Healpy routines included.
"""
from os import path
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, sem, percentileofscore
from scipy.stats import poisson as scipy_poisson
from scipy.stats import lognorm as scipy_lognorm
from scipy.stats import f as statsF
from scipy.optimize import curve_fit
from scipy.special import erfc, erfcinv
from scipy.interpolate import interp2d
import astropy.units as u
from astropy.table import Table, vstack, hstack, Column, MaskedColumn, join, unique
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import healpy as hp
from time import time

rc('font', family='serif', size=14)
rc('text', usetex='true')

T408MHz_map = path.join(
    path.dirname(path.realpath(__file__)),
    'reference/haslam408_dsds_Remazeilles2014_512.fits'
    )


###############################################################################
# ALIASES #####################################################################
###############################################################################
NoneType = type(None)

pi = np.pi
log10 = np.log10
deg2rad = pi / 180.
rad2deg = 180. / pi

rng = np.random.default_rng

ang2pix_ring = hp._healpy_pixel_lib._ang2pix_ring


###############################################################################
# PHYSICAL VALUES #############################################################
###############################################################################
c = 299792458

skyarea = 4 * pi * (180 / pi)**2

# Planck values
lon_CMBdipole = 264.021
lat_CMBdipole = 48.253
velocity_CMBframe = 369820. # m/s

sc_cmb = SkyCoord(lon_CMBdipole, lat_CMBdipole, unit=u.deg, frame='galactic')

# nside npix    deg2/pix
# 16     3072   13.4
# 32    12288   3.4
# 64    49152   0.8


###############################################################################
# MATH FUNCTIONS ##############################################################
###############################################################################
def divide(x1, x2):
    """divide(x1, x2)

    Perform true division of x1 by x2. Divide-by-zero set to np.nan.

    Parameters
    ----------
    x1, x2 : arrays of floats or ints
        x1 will be divided by x2.

    Returns
    -------
    result : array of floats
        x1 divided by x2. Where x2=0 set to np.nan
    """

    return np.divide(x1, x2, out=np.full_like(x1, np.nan), where=x2!=0)


def omega_to_theta(omega):
    """omega_to_theta(omega)

    Convert solid angle omega in steradians to theta in radians for a
    cone section of a sphere.
    """
    return np.arccos(1 - omega / (2 * pi)) * u.rad


def rcumsum(x):
    """rcumsum(x)

    Count number of elements in x above x_i for every ith element in x.

    x must be sorted.
    """
    s = np.empty(x.size, dtype=int)
    for i in range(x.size):
        s[i] = x[i:].size
    return s


###############################################################################
# STATS FUNCTIONS #############################################################
###############################################################################
def wavg(x, ex):
    """wavg(x, ex)

    Returns arithmetic weighted mean of x, given uncertainty ex.
    """
    w = ex**(-2)
    return np.sum(x * w) / np.sum(w), np.sum(w)**(-0.5)

def nanwavg(x, w):
    msk = np.isfinite(x) & np.isfinite(w)
    return (x[msk] * w[msk]).sum() / w[msk].sum()


def get_chi2(x, xerr, x_fit, k):
    chi2 = np.sum( ( x - x_fit )**2 / xerr**2 )
    dof = x.size - k
    return chi2, dof


def ftest(chi2_1, dof_1, chi2_0, dof_0):
    """ftest(chi2_1, dof_1, chi2_0, dof_0)

    Calculate F-statistic and probability for dof_1 < dof_0
    """
    F = ((chi2_0 - chi2_1) / (dof_0 - dof_1)) / (chi2_1 / dof_1)
    p = statsF.sf(F, dof_0 - dof_1, dof_0)
    return F, p

def residual(fx, x, y, w):
    r = y - fx
    z = r * w   # z-score
    chi2 = np.sum(z**2)
    dof = x.size - 2
    print("z-score stdev: %.2f" % z.std())
    print("chi2/dof: %.2f/%i = %.2f" % (chi2, dof, chi2/dof))

    return r, z, chi2, dof


def get_MAD(x):
    md = np.median(x)
    MAD = np.median(np.abs(x - md))
    return md, 1.4826 * MAD


def standardize(x, mask=None):
    try:
        mu, sig = x[~mask].mean(), x[~mask].std(ddof=1)
    except TypeError:
        mu, sig = x.mean(), x.std(ddof=1)
    return (x - mu) / sig


def fastbincount(x, minlength):
    """fastbincount(x, minlength)

    numpy.bincount method with a jit decorator. Takes about 3/4 the time
    for minlength=49152
    """
    return np.bincount(x, minlength=minlength)


def p2nsig(p, sided=1):
    if sided==1:
        return erfcinv(p * 2) * np.sqrt(2)
    elif sided==2:
        return erfcinv(p) * np.sqrt(2)
    else:
        raise ValueError("sided must be 1 or 2")


def mk_interp2d_p(x, y, xedges, yedges):
    # np.histogram2d takes x=col, y=row, so swap.
    h = np.histogram2d(y, x, [yedges, xedges])[0]
    xc = xedges[:-1] + np.ediff1d(xedges) / 2
    yc = yedges[:-1] + np.ediff1d(yedges) / 2

    pdf = h / h.sum()
    pdf_flat = pdf.flatten()
    idx_sort = np.argsort(pdf_flat)

    ps = np.cumsum(pdf_flat[idx_sort])
    ps = ps[np.argsort(idx_sort)].reshape(pdf.shape)

    # Create interpolation function.
    interp2d_p = interp2d(
        xc, yc, ps, kind='linear',
        bounds_error=False, fill_value=0.
        )

    return h, pdf, ps, interp2d_p


###############################################################################
# MISC FUNCTIONS ##############################################################
###############################################################################
def mk_unq(x, y):
    """mk_unq(x, y)

    Make unique x for vector containing repeated x, each with a value of y.
    The returned value of y is the mean value for each unique x.
    """
    unq, unq_idx = np.unique(x, return_index=True)
    ys = np.empty(unq.size)
    for i in range(unq.size - 1):
        ys[i] = y[unq_idx[i]:unq_idx[i+1]].mean()
    return unq, ys


def logSlogN(logS):
    """logSlogN(logS)

    Get logN vs. logS such that N is cumulative.

    Returns a sorted copy of logS, and logN
    """
    logS_sorted = logS[np.argsort(logS)]
    logN = np.log10(rcumsum(logS_sorted))
    return mk_unq(logS_sorted, logN)


def alpha_at_S(logS, alphas, logS_min, logS_max, deg, dp):
    """alpha_at_S(logS, alphas, logS_min, logS_max, deg, dp)

    Estimate value of alpha s.t. S ~ nu ** -alpha at min(S). logS_min is
    the value to calculate alpha for, which may lie outside of the range of
    logS (e.g., if logS vs alphas are binned quantities). Precision is given
    in decimal places dp.
    """

    msk = logS < logS_max
    logS_fit, alpha_fit = logS[msk], alphas[msk]
    p = np.polyfit(logS_fit, alpha_fit, deg=deg)
    alpha_mdl = np.polyval(p, logS_fit)
    res = (alpha_fit - alpha_mdl) / alpha_fit * 100 # As a percentage
    print("Residuals standard deviation: %.2g%%" % res.std(ddof=1))
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0.scatter(logS_fit, alpha_fit, s=1)
    ax0.plot(logS_fit, alpha_mdl, c='k')
    ax1.scatter(logS_fit, res, s=2)
    ax1.axhline(y=0, ls=':', c='k')
    ax1.set_xlabel('log(S)')
    ax0.set_ylabel('alpha')
    ax1.set_ylabel('residual (\%)')
    plt.show()

    # Alpha at flux limit
    alpha = np.round(np.polyval(p, logS_min), dp)

    print("alpha at flux density limit: %s" % alpha)

    return alpha


def x_at_S(logS, logN, logS_min, logS_max, deg, dp):
    """x_at_S(logS, logN, logS_min, logS_max, deg, dp)

    Estimate value of x s.t. dN(>S)/dS ~ S ** -x at min(S). logS_min is
    the value to calculate x for. Precision is given in decimal places dp.
    """
    msk = logS < logS_max
    logS_fit, logN_fit = logS[msk], logN[msk]
    p = np.polyfit(logS_fit, logN_fit, deg=deg)
    logN_mdl = np.polyval(p, logS_fit)
    res = (logN_fit - logN_mdl) / logN_fit * 100
    print("Residuals standard deviation: %.2g%%" % res.std(ddof=1))
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0.scatter(logS_fit, logN_fit, s=1)
    ax0.plot(logS_fit, logN_mdl, c='k')
    ax1.scatter(logS_fit, res, s=2)
    ax1.axhline(y=0, ls=':', c='k')
    ax1.set_xlabel('log(S)')
    ax0.set_ylabel('log($N > S$)')
    ax1.set_ylabel('residual (\%)')
    plt.show()

    # Calculate x at flux limit
    idx = np.arange(deg)
    p = p[idx]
    idx = np.flip(idx)
    p *= idx + 1
    p *= logS_min ** idx # Evaluate derivative at flux density limit
    x = -p.sum()

    # The formal errors of the fit are not representative of the uncertainty
    # from variations in choice of flux cut and polynomial degree, so round to
    # desired precision.
    x = np.round(x, dp)
    print("x at flux density limit: %s" % x)

    return x


def eb84(alpha, x, vel=velocity_CMBframe):
    return (2 + x * (1 + alpha)) * vel/c


def moll(obj, frame='galactic', mask=None):
    if isinstance(mask, np.ndarray):
        pass
    else:
        mask = np.zeros(len(obj), dtype=bool)
    if isinstance(obj, SkyCoord):
        obj = obj[~mask]
        if frame=='icrs':
            x, y = obj.icrs.ra, obj.icrs.dec
        elif frame=='galactic':
            x, y = obj.galactic.l, obj.galactic.b
        elif frame=='supergalactic':
            x, y = obj.supergalactic.sgl, obj.supergalactic.sgb
        else:
            raise ValueError("Supported coordinate frames: icrs, " + \
                             "galactic, supergalactic")

        x = -x.wrap_at(180 * u.deg).value
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='mollweide')
        ax.scatter(np.radians(x), np.radians(y), s=1)
        _ = ax.set_xticklabels([])
        _ = ax.set_yticklabels([])
        plt.show()
    elif isinstance(obj, np.ndarray):
        obj_mask = obj.astype(float)
        obj_mask[mask] = hp.UNSEEN
        hp.visufunc.mollview(obj_mask, cmap='RdBu_r')
        plt.show()
    else:
        raise TypeError("Unrecognized input type.")
    return None


def get_T_408MHz(tbl):
        """get_T_408MHz(tbl)

        Determine 408MHz brightness temperature for positions in table
        tbl, given ICRS coordinates ra and dec.
        """

        sc = SkyCoord(tbl['ra'], tbl['dec'], unit=u.deg, frame='icrs')
        phi, theta = lonlat2phitheta(sc.galactic.l.value,
                                     sc.galactic.b.value)
        ipix = hp.ang2pix(512, theta, phi)
        m = hp.fitsfunc.read_map(T408MHz_map)
        return m[ipix]


def smooth_map(hpm, theta, keys=['density']):
    """smooth_map(hpm, theta, keys=['density'])

    Smooth sky map by taking a running mean with radius theta.
    Modifies hpm in-place by adding columns with _mean, and
    _sem (standard error of the mean) suffix.
    """

    print("Calculating running mean quantities...")
    npix = len(hpm)
    sc = SkyCoord(hpm['ra'], hpm['dec'], unit='deg', frame='icrs')

    for key in keys:
        hpm[key + '_mean'] = np.nan * np.ones(npix, dtype=float)

    s = hpm.copy()
    s.remove_columns(set(s.keys()) - set(keys))
    # Ensure that row values are NaN
    for key in keys:
        s[key][hpm['mask']==True] = np.nan

    for i in range(npix):
        d2d = sc[i].separation(sc)
        msk = d2d < theta
        if np.isfinite(s[key][msk]).sum() < 2:
            continue

        for key in keys:
            hpm[key + '_mean'][i] = np.nanmean(s[key][msk])

        print("\t%.1f%%" % ((i + 1) / npix * 100), end='\r')

    for key in keys:
        hpm[key + '_mean'][hpm['mask']] = np.nan

    return None


###############################################################################
# MASK FUNCTIONS ##############################################################
###############################################################################
def mask_dir2pix(tmask, nside, frame_out, lonkey='ra', latkey='dec',
                 radiuskey='radius', frame_in='icrs', nest=False):
    """mask_dir2pix(tmask, nside, frame_out, lonkey='ra', latkey='dec',
                    radiuskey='radius', frame_in='icrs'):

    Convert table of mask coordinates and radii into HEALPix map. Only
    circular regions are suppoorted. All pixels overlapping these regions
    are masked. ICRS or Galactic coordinates supported.

    """
    tmask_sc = SkyCoord(tmask['ra'], tmask['dec'], unit=u.deg,
                        frame=frame_in)
    radius = np.radians(tmask['radius'].data)

    if frame_out=='galactic':
        phi, theta = lonlat2phitheta(tmask_sc.galactic.l.value,
                                     tmask_sc.galactic.b.value)
    elif frame_out=='icrs':
        phi, theta = lonlat2phitheta(tmask_sc.ra.value,
                                     tmask_sc.dec.value)
    else:
        raise Exception(
            "Error: frame_out must be either 'galactic' or 'icrs'."
            )

    vec = hp.ang2vec(theta, phi)

    ipix_mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    for i in range(vec.shape[0]):
        disc = hp.query_disc(nside=nside, vec=vec[i], radius=radius[i],
                             inclusive=True, nest=nest)
        for ipix in disc:
            ipix_mask[ipix] = True
    return ipix_mask



def mask_tbl(tbl, mask):
    tpix = Table()
    tpix['ipix'] = np.arange(mask.size)[~mask]
    return join(tbl, tpix, keys='ipix')


###############################################################################
# SKY VECTOR FUNCTIONS ########################################################
###############################################################################
def lon_std(lon, lat, ddof=1):
    """lon_std(lon, lat, ddof=1)

    Calculate standard deviation of longitude expressed in degrees, accounting
    for the cos(lat) term.
    """
    dlon = (lon - lon.mean()) * np.cos(np.radians(lat))
    return np.sqrt(1 / (lon.size - ddof) * np.sum( dlon ** 2 ))


def getIsotropicDistributionVectors(N, seed=None):
    """getIsotropicDistributionVectors(N, seed=None)

    Returns a sample of N vectors drawn from an isotropic distribution
    """

    num = np.random.default_rng(seed).standard_normal((3,N))
    return num/((num**2).sum(axis=0))**(0.5)


def getRotationMatrix_Z(lon=lon_CMBdipole) :

    """
    Computes the rotation matrix around the Z-axis for moving a vector with
    longitude lon to the zero meridian
    """

    a1 = -lon

    c1 = np.cos(deg2rad * a1)
    s1 = np.sin(deg2rad * a1)

    rot_mat = np.array([[c1, -s1, 0],[s1, c1, 0.],[0, 0, 1.]])

    return rot_mat


def rotateVectors(vec, rot_mat):
    """rotateVectors(vec, rot_mat)

    Given a rotation matrix, computes the rotated vector of an input
    Cartesian vector with unit length.
    """
    return np.matmul(rot_mat,vec)


def vec2dir(vec) :
    """vec2dir(vec)

    Converts a Cartesian vector with unit length into longitude and
    latitude in degrees.
    """

    x = vec[0]
    y = vec[1]
    z = vec[2]

    theta = np.arccos(z)
    phi = np.arctan2(y,x)

    lat,lon = 90.-rad2deg*theta,rad2deg*phi

    return lon,lat


def lonlat2phitheta(lon, lat):
    return deg2rad*lon, deg2rad*(90.-lat)


def mkdipole(vec, xyz, nest=False):
    """mkdipole(vec, xyz):

    Return a normalized (min, max = -1, 1) map of a dipole with direction
    described by vec.
    """
    dipole = np.dot(vec, xyz)
    return dipole / (dipole @ dipole) ** 0.5


def scattomap(theta, phi, nside, npix, extra=[]):
    """scattomap(theta, phi, nside, npix, extra=[])

    Returns a histogram of celestial objects whose position is given
    in latitute and longitude by bins chosen by HEALPix (at resolution
    nside).
    """

    ipix = ang2pix_ring(nside, theta, phi)

    cnt = fastbincount(ipix, minlength=npix)

    lenextra = len(extra)
    extra_array = np.empty((lenextra, npix))
    for i in range(lenextra):
        extra_array[i] = np.bincount(ipix, weights=extra[i], minlength=npix)
    extra_array = divide(extra_array, cnt)

    return cnt, extra_array


def great_circle_path(lon_1, lat_1, lon_2, lat_2, n_point=0):
    """great_circle_path(lon_1, lat_1, lon_2, lat_2, n_point=0)

    Calculate the great circle distance between two points, and return
    evenly-spaced points between.

    Parameters
    ----------
        lon_1, lat_1, lon_2, lat_2 : floats
            Longitudes and latitudes of two points, in degrees.
        n_points : int
            Number of evenly-spaced coordinates between two points to return.
            If n_point=0 (default), then only sigma_12 is returned.

    Returns
    -------
        sigma_12 : float
            Great distance between two input points, in degrees.
        coords : numpy.array
            Array of longitudes and latitudes of evenly-spaced points.

    Notes
    -----
    See https://en.wikipedia.org/wiki/Great-circle_navigation
    """

    lambda_1, phi_1 = -np.radians(lon_1), np.radians(lat_1)
    lambda_2, phi_2 = -np.radians(lon_2), np.radians(lat_2)
    lambda_12 = lambda_2 - lambda_1

    # Minimize clutter
    sin, cos, tan, atan2 = np.sin, np.cos, np.tan, np.arctan2

    alpha_1 = atan2(
        cos(phi_2) * sin(lambda_12),
        cos(phi_1) * sin(phi_2) - sin(phi_1) * cos(phi_2) * cos(lambda_12)
        )

    alpha_2 = atan2(
        cos(phi_1) * sin(lambda_12),
        -cos(phi_2) * sin(phi_1) + sin(phi_2) * cos(phi_1) * cos(lambda_12)
        )

    sigma_12 = atan2(
        np.hypot(
        cos(phi_1) * sin(phi_2) - sin(phi_1) * cos(phi_2) * cos(lambda_12),
        cos(phi_2) * sin(lambda_12)
        ),
        sin(phi_1) * sin(phi_2) + cos(phi_1) * cos(phi_2) * cos(lambda_12)
        )

    if n_point==0:
        return np.degrees(sigma_12)

    alpha_0 = atan2(
        sin(alpha_1) * cos(phi_1),
        np.hypot(cos(alpha_1), sin(alpha_1) * sin(phi_1))
        )

    sigma_01 = atan2(tan(phi_1), cos(alpha_1))
    sigma_02 = sigma_01 - sigma_12

    lambda_01 = atan2(
        sin(alpha_0) * sin(sigma_01),
        cos(sigma_01)
        )

    lambda_0 = lambda_1 - lambda_01

    # Step size
    sigmas = sigma_01 + np.linspace(0, sigma_12, n_point)

    coord_array = np.empty((n_point, 3))
    for i in range(n_point):
        phi_i = atan2(
            cos(alpha_0) * sin(sigmas[i]),
            np.hypot(cos(sigmas[i]), sin(alpha_0) * sin(sigmas[i]))
            )
        lambda_i = atan2(
            sin(alpha_0) * sin(sigmas[i]),
            cos(sigmas[i])
            ) + lambda_0
        alpha_i = atan2(
            tan(alpha_0),
            cos(sigmas[i])
            )
        coord_array[i] = -lambda_i, phi_i, alpha_i

    return np.degrees(sigma_12), np.degrees(coord_array[:, 0:2])


###############################################################################
# REGRESSION FUNCTIONS ########################################################
###############################################################################
def binxy(xvar, yvar, binsize, logx=False, logy=False):
    msk = np.isfinite(xvar) & np.isfinite(yvar)
    x, y = xvar[msk], yvar[msk]
    if logx==True:
        x = log10(x)
    if logy==True:
        y = log10(y)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    bins = np.arange(0, x.size, binsize)
    xc, stat = np.empty(bins.size-1), np.empty((2, bins.size-1))
    for i in range(bins.size-1):
        xc[i] = (x[bins[i]:bins[i+1]]).mean()
        y_i = y[bins[i]:bins[i+1]]
        stat[0][i] = y_i.mean()
        stat[1][i] = y_i.std(ddof=1)

    return xc, stat[0], stat[1] / np.sqrt(binsize)


def nbinxy(xvar, yvar, nbins):
    msk = np.isfinite(xvar) & np.isfinite(yvar)
    x, y = xvar[msk], yvar[msk]
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    bin_edges = np.linspace(x.min(), np.nextafter(x.max(), np.inf), nbins + 1)
    idx = np.digitize(x, bin_edges, right=False) - 1
    stat = np.nan * np.empty((nbins, 3))
    for i in range(nbins):
        xi, yi = x[idx==i], y[idx==i]
        if yi.size < 2:
            continue
        if yi.size < 20:
            print("Warning: bin size < 20!")
        stat[i] = xi.mean(), yi.mean(), sem(yi)
    stat = stat[np.isfinite(stat[:,2])]
    return stat[:,0], stat[:,1], stat[:,2]



def dbin(phi, res, nbins):
    msk = np.isfinite(phi) & np.isfinite(res)
    x, y = phi[msk], res[msk]
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    bin_edges = np.linspace(x.min(), np.nextafter(x.max(), np.inf), nbins + 1)
    idx = np.digitize(x, bin_edges, right=False) - 1
    stat = np.nan * np.empty((nbins, 3))
    for i in range(nbins):
        xi, yi = x[idx==i], y[idx==i]
        if yi.size < 2:
            continue
        stat[i] = xi.mean(), yi.mean(), sem(yi)

    stat[:,1] = np.abs(stat[:,1])

    stat[:,0] = 90 * (1 - np.abs(stat[:,0]/90 - 1))
    stat[:,1] = np.abs(stat[:,1])
    stat = stat[np.argsort(stat[:,0])]

    return stat[:,0], stat[:,1], stat[:,2]


def xyfit(hpm, mask, xkey, ykey, deg=0, n=200, bintype='binsize',
          verbose=True, return_xy=False, errorbars=True,
          xlabel=None, ylabel=None, grid=True, show_plot=True, d=None,
          shuffles=None):

    """def xyfit(hpm, mask, xkey, ykey, deg=0, n=200, bintype='binsize',
                 verbose=True, return_xy=False, errorbars=True,
                 xlabel=None, ylabel=None, grid=True, show_plot=True, d=None,
                 shuffles=None):

    Bin HEALPix map ykey with respect to xkey, for pixels outside of mask.
    Will work with any table as long as a mask of the same size is provided.
    The mask denotes pixels/sources that will NOT be used.
    """
    if bintype == 'nbin':
        binner = nbinxy
    elif bintype == 'binsize':
        binner = binxy
    else:
        raise Exception("Unrecognized bintype '%s'" % bintype)

    if isinstance(d, float):
        binner = dbin

    xvar, yvar = hpm[~mask][xkey].data, hpm[~mask][ykey].data
    x, y, s = binner(xvar, yvar, n)

    mu, se = yvar.mean(), sem(yvar)
    if deg==0:
        p, pcov = np.array([mu.copy()]), np.array([se**2])
        fx = np.full_like(x, mu)

    else:
        p, pcov = np.polyfit(x, y, deg=deg, full=False, w=1/s, cov=True)
        fx = np.polyval(p, x)

    k = deg + 1
    chi2, dof = get_chi2(y, s, fx, k=k)

    if isinstance(shuffles, int) and shuffles >= 2 and deg == 0:
        rchi2s = np.empty(shuffles)
        for i in range(shuffles):
            np.random.shuffle(yvar) # Not using again later
            xi, yi, si = binner(xvar, yvar, n)
            rchi2s[i] = divide(*get_chi2(yi, si, fx, k=k))
        mu_rchi2, std_rchi2 = rchi2s.mean(), rchi2s.std(ddof=1)
        print(
            "\nrchi2 for shuffled data: %.2f +/- %.2f" % (mu_rchi2, std_rchi2)
            )

    if verbose==True:
        print("\n%s vs. %s: " % (ykey, xkey))
        r, rho, tau = pearsonr(x, y), spearmanr(x, y), kendalltau(x, y)
        print("Pearson  r = %.2g (p=%.2g)" % r)
        print("Spearman r = %.2g (p=%.2g)" % rho)
        print("Kendall's tau = %.2g (p=%.2g)" % tau)
        print("Degree %i polynomial:" % deg)
        print("chi2/dof = %.2g/%i = %.2g" % (chi2, dof, chi2/dof))

    if isinstance(d, float):
        fx = d * np.cos(np.radians(x))
        chi2, dof = get_chi2(y, s, fx, k=1)
        print("d = %.3g dipole:" % d)
        print("chi2/dof = %.2g/%i = %.2g" % (chi2, dof, chi2/dof))

    if grid==True:
        plt.grid(alpha=0.5, zorder=0)
    if isinstance(xlabel, NoneType):
        xlabel = xkey
        ylabel = ykey

    if errorbars==True:
        plt.errorbar(x, y, yerr=s, ls='none', marker='.', zorder=1)
    else:
        plt.scatter(x, y, marker='.', zorder=1)

    plt.plot(x, fx, ls='--', c='k', zorder=2)
    plt.xlabel('%s' % xlabel)
    plt.ylabel('%s' % ylabel)

    if show_plot==True:
        plt.show()

    if return_xy == True:
        return x, y, s, fx
    else:
        return p, pcov, chi2, dof


def linreg(x, y, w):
    print("Pearson r: %.2f" % pearsonr(x, y)[0])
    p, pcov = np.polyfit(x, y, deg=1, w=w, cov=True)
    perr = np.sqrt(np.diag(pcov))
    print("Equation of fit: " +
          "y = %.3f(%.3f) * x + %.1f(%.1f)" % (p[0], perr[0],
                                               p[1], perr[1]))
    fx = np.polyval(p, x)
    r, z, chi2, dof = residual(fx, x, y, w)

    return p, pcov, fx, z, chi2, dof


###############################################################################
# SkyMap CLASS ################################################################
###############################################################################
class SkyMap:
    def __init__(self, nside, frame):
        if frame=='galactic':
            self.__lonlat_str = '(l,b)'
        elif frame=='icrs':
            self.__lonlat_str = '(ra,dec)'
        else:
            raise("ERROR: only galactic and icrs frames supported.")

        self.nside = nside
        self.__nest = False # Everything is RING ordered
        self.frame = frame
        self.npix = hp.nside2npix(nside)
        self.__ones = np.ones(self.npix) # Used for uniform weighting of maps
        self.scl = self.npix / skyarea
        self.ipix = np.arange(self.npix)

        # Do not allow outside modification of the mask
        self.__mask = np.zeros(self.npix, dtype=bool)
        self.__seen = ~self.__mask  # Minor speed-up


        # Get coordinates of pixels
        self.lon, self.lat = hp.pix2ang(self.nside,
                                        np.arange(self.npix),
                                        nest=self.__nest, lonlat=True)

        self.sc = SkyCoord(self.lon, self.lat, unit=u.deg, frame=self.frame)

        self.xyz = np.vstack(
            hp.pix2vec(self.nside, np.arange(self.npix), nest=self.__nest)
            )


    def __repr__(self):
        print("\nNside = %i" % self.nside)
        print("Frame = %s" % self.frame)
        print("Scale = %.1f pix / deg2" % self.scl)
        print(
            "Percentage of sky masked: %i%%" % \
            np.round(self.__mask.sum() / self.npix * 100)
            )
        try:
            print("Velocity = %.3gc towards %s " % \
                  (self.beta, self.__lonlat_str) + \
                  "= (%.1f,%.1f)." % (self.lon_vel, self.lat_vel))
        except AttributeError:  # set_vel not invoked yet.
            pass
        return '\r'


    def load_tbl(self, tbl, ra='ra', dec='dec'):
        if isinstance(tbl, str):
            tbl = Table.read(tbl)   # Allow FileNotFoundError
            inplace = False

        elif isinstance(tbl, Table):
            inplace = True
        else:
            raise TypeError("Unrecognized tbl type: %s" % type(tbl))

        rd = SkyCoord(tbl[ra], tbl[dec], unit=u.deg, frame='icrs')

        if self.frame=='galactic':
            tbl['phi'], tbl['theta'] = lonlat2phitheta(rd.galactic.l.value,
                                                       rd.galactic.b.value)
        elif self.frame=='icrs':
            tbl['phi'], tbl['theta'] = lonlat2phitheta(rd.ra.value,
                                                       rd.dec.value)
        else:
            raise Exception("Frame must be either 'galactic' or 'icrs'")

        tbl['ipix'] = hp.ang2pix(self.nside, tbl['theta'].data, tbl['phi'].data)

        if inplace==False:
            return tbl
        return None


    def set_mask(self, mask):
        self.__mask[mask] = True
        self.__seen = ~self.__mask
        self.set_coefficient_matrix()
        return None


    def get_mask(self): # To access, but not be able to modify, mask
        return self.__mask.copy()


    def load_map(self, map_file):
        hpm = Table.read(map_file)
        if len(hpm) != hp.nside2npix(self.nside):
            raise ValueError(
                "%s does not have Nside=%i" % (map_file, self.nside)
                )

        try:
            if self.frame != hpm.meta['FRAME']:
                raise ValueError(
                    "%s does not have frame=%s" % (map_file, self.frame)
                    )
        except KeyError:
            print(
                "Warning: FRAME keyword not found in %s" % map_file
                )

        try:
            if self.__nest != hpm.meta['NEST']:
                raise ValueError(
                    "%s does not have nest=%s" % (map_file, self.__nest)
                    )
        except KeyError:
            print(
                "Warning: NEST keyword not found in %s" % map_file
                )

        self.set_mask(hpm['mask'].data)
        return hpm


    def simulate_sky(self, N, d, coords, seed=None):
        """simulate_sky(N, d, coords, seed=None)

        Simulate a random dipole sky with N objects in the non-masked (seen)
        regions, given dipole amplitude d and direction coords.

        This is quite slow.
        """
        mask_frac = (self.npix - self.__mask.sum()) / self.npix
        Ntot = int(N / mask_frac)
        vecs = getIsotropicDistributionVectors(Ntot)
        lon, lat = vec2dir(vecs)
        sc = SkyCoord(lon, lat, unit=u.deg, frame=self.frame)
        sc_d = SkyCoord(*coords, unit=u.deg, frame=self.frame)
        theta = sc.separation(sc_d).to(u.radian)
        p = np.cos(theta) + 1/d
        p /= p.sum()
        choice = rng(seed).choice(np.arange(Ntot), size=Ntot, p=p)
        vecs_dip = vecs.T[choice].T
        lon, lat = vec2dir(vecs_dip)
        sc = SkyCoord(lon, lat, unit=u.deg, frame=self.frame)
        tsim =Table()
        tsim['ra'] = sc.icrs.ra
        tsim['dec'] = sc.icrs.dec
        self.load_tbl(tsim)
        return tsim


    def mk_hpmap(self, tbl, extra_keys=[]):
        phi, theta = tbl['phi'].data, tbl['theta'].data

        extra = [tbl['%s' % s] for s in extra_keys]

        cnt, extra_array = scattomap(theta, phi, self.nside, self.npix,
                                     extra=extra)

        # Convert to sky density
        density = cnt * self.npix / skyarea

        hpm = Table()
        hpm['ipix'] = self.ipix.copy()
        hpm['count'] = Column(cnt, unit=u.ct / u.pix)
        hpm['density'] = Column(density, unit=u.ct / u.deg**2)
        for i in range(len(extra_keys)):
            hpm[extra_keys[i]] = Column(extra_array[i],
                                        name=extra_keys[i],
                                        unit=tbl[extra_keys[i]].unit)
        hpm['ra'] = self.sc.icrs.ra
        hpm['dec'] = self.sc.icrs.dec
        hpm['l'] = self.sc.galactic.l
        hpm['b'] = self.sc.galactic.b
        hpm['elon'] = self.sc.barycentricmeanecliptic.lon
        hpm['elat'] = self.sc.barycentricmeanecliptic.lat
        hpm['sgl'] = self.sc.supergalactic.sgl
        hpm['sgb'] = self.sc.supergalactic.sgb

        # Store HEALPix metadata
        hpm.meta['NSIDE'] = self.nside
        hpm.meta['NEST'] = self.__nest
        hpm.meta['FRAME'] = self.frame
        return hpm


    def save_map(self, hpm, map_file):
        hpm['mask'] = self.__mask
        hpm.write(map_file, overwrite=True)
        return None


    def save_tbl(self, tbl, tbl_file, return_tbl=False):
        tbl_ipix = Table()
        tbl_ipix['ipix'] = self.ipix.copy()
        tbl_ipix['masked'] = self.__mask.copy()
        tbl = join(tbl, tbl_ipix, keys='ipix')
        tbl.meta['NSIDE'] = self.nside
        tbl.meta['NEST'] = self.__nest
        tbl.meta['FRAME'] = self.frame
        tbl.write(tbl_file, overwrite=True)

        # To avoid breaking downstream code written before modifying this
        # function to keep the masked rows in the output table.
        tbl = tbl[tbl['masked']==False]
        tbl.remove_column('masked')

        if return_tbl==True:
            return tbl
        else:
            return None


    def mask_zeros(self, tbl):
        """mask_zeros(tbl, frame)

        Find no-coverage areas in catalog, and produce a mask
        corresponding to sky map produced in a given frame.

        Parameters
        ----------
        tbl : astropy.table.Table
            Table original catalog (with no cuts applied).
        frame : str
            Either 'galactic' or 'icrs'

        """

        hpm = self.mk_hpmap(tbl)

        idx0 = np.where(hpm['density']==0)[0]
        indices = np.empty((idx0.size, 8), dtype=int)
        for i in range(idx0.size):
            indices[i] = hp.pixelfunc.get_all_neighbours(self.nside, idx0[i])

        mask = np.zeros(self.npix, dtype=bool)
        mask[idx0] = True
        mask[indices] = True
        self.set_mask(mask)
        return None


    def fits2mask(self, fitsfile, usekey='use'):
        tmask = Table.read(fitsfile)
        try:
            tmask = tmask[tmask[usekey]==True]
        except KeyError:
            pass

        self.set_mask(
            mask_dir2pix(tmask, self.nside, frame_out=self.frame)
            )
        return None


    def mask_Tcut(self, tbl, Tcut=None, binsize=100, show=False):
        hpm = self.mk_hpmap(tbl, extra_keys=['Tb'])
        xc, uy, sy = binxy(hpm['Tb'][self.__seen],
                           hpm['density'][self.__seen],
                           binsize=binsize, logx=True)

        xc = 10**xc
        plt.errorbar(xc, uy, yerr=sy, ls='none', marker='o')
        plt.xscale('log')
        plt.xlabel('408 MHz brightness temperature (K)')
        plt.ylabel('source density (deg$^{-2}$)')
        plt.grid(which='both', alpha=0.5)
        if Tcut != None:
            plt.axvline(x=Tcut, c='k', ls='--', lw=1, zorder=3)
            self.set_mask(hpm['Tb'] > Tcut)
        if show==True:
            plt.show()
        else:
            plt.close()
        return None


    def poiss_hist(self, hpm, fname=None):
        """poiss_hist(hpm, fname=None)

        Make histogram of sky pixel counts overlaid on Poissonian
        expectation.

        Parameters
        ----------
        hpm : astropy.table.Table
            Table containing 'count' column that contains actual pixel
            counts, NOT the counts scaled to density per square degree.

        fname : str, optional
            Prefix to use for output file name.

        """
        # Distribution for Poisson-distributed data
        cnt = hpm['count'][self.__seen].data  # Actual sky pixel counts
        mu = cnt.mean()

        bin_min, bin_max = cnt.min() - 0.5, cnt.max() + 0.5
        bins = np.arange(bin_min, bin_max + 1, 1)

        pltx = np.arange(cnt.min(), cnt.max() + 1)
        plty = scipy_poisson.pmf(pltx, mu)

        # 1=(np.ediff1d(bins) * h).sum()=(np.ediff1d(bins * scl) * h/scl).sum()
        pltx, plty = pltx * self.scl, plty / self.scl

        h = np.histogram(cnt, bins=bins)[0]
        eh = np.sqrt(h)
        h_sum = h.sum()
        h = h / h_sum
        eh /= h_sum
        h, eh = h / self.scl, eh / self.scl

        fig, ax = plt.subplots()
        ax.plot(pltx, plty, drawstyle='steps-mid', c='gray', zorder=0)
        ax.scatter(pltx, h, zorder=1)
        ax.errorbar(pltx, h, yerr=eh, linestyle='none', c='k', zorder=2)
        plt.grid(alpha=0.5)
        ax.set_xlabel('sky pixel value (deg$^{-2}$)')
        ax.set_ylabel('PDF')
        if isinstance(fname, str):
            fmt = fname.split('.')[-1]
            fig.savefig(fname, dpi=600, format=fmt)
        plt.show()
        return None


    def set_vel(self, sc=sc_cmb, vel=velocity_CMBframe):
        self.lon_vel = sc.data.lon.value
        self.lat_vel = sc.data.lat.value
        self.beta = vel / c
        self.gamma = (1 - self.beta**2)**(-0.5)
        self.theta_vel = self.sc.separation(sc).radian
        self.delta = self.gamma * (1 + self.beta * np.cos(self.theta_vel))
        return None


    def vel2map(self, x, alpha, mono):
        """vel2map(self, x, alpha, mono)

        Get kinematic expectation map for CMB dipole direction, calculated as:

        delta ** (2 + x * (1 + alpha)) * mono

        where delta is the Doppler factor:

        delta = gamma * (1 + beta * cos(theta))

        x is the power law slope of the integral source counts above the flux
        density cut of the catalog, as a function of flux density, and alpha
        is the mean spectral index (e.g., Ellis & Baldwin 1984).

        """
        return self.delta ** (2 + x * (1 + alpha)) * mono


    def set_coefficient_matrix(self):
        ipix = self.ipix[self.__seen]
        self.x, self.y, self.z = self.xyz.T[ipix].T
        self.a = np.zeros((4, 4))
        self.a[0, 0] = ipix.size
        self.a[1, 0] = self.x.sum()
        self.a[2, 0] = self.y.sum()
        self.a[3, 0] = self.z.sum()
        self.a[1, 1] = (self.x ** 2).sum()
        self.a[2, 1] = (self.x * self.y).sum()
        self.a[3, 1] = (self.x * self.z).sum()
        self.a[2, 2] = (self.y ** 2).sum()
        self.a[3, 2] = (self.y * self.z).sum()
        self.a[3, 3] = (self.z ** 2).sum()
        self.a[0, 1] = self.a[1, 0]
        self.a[0, 2] = self.a[2, 0]
        self.a[0, 3] = self.a[3, 0]
        self.a[1, 2] = self.a[2, 1]
        self.a[1, 3] = self.a[3, 1]
        self.a[2, 3] = self.a[3, 2]
        return None


    def set_ordinate_matrix(self, m, w, poiss=False, seed=None):
        m_fit = m[self.__seen]

        # Permuting on the masked map reduces overhead from poisson
        if poiss==True:
            m_fit = rng(seed).poisson(m_fit)

        if isinstance(w, NoneType)==False:
            m_fit = m_fit * w[self.__seen]

        self.b = np.zeros(4)
        self.b[0] = m_fit.sum()
        self.b[1] = (m_fit * self.x).sum()
        self.b[2] = (m_fit * self.y).sum()
        self.b[3] = (m_fit * self.z).sum()
        return None


    def fit_dipole(self, m, w=None, poiss=False, seed=None):
        """fit_dipole(self, m, w=None, poiss=False, seed=None):

        Fit a dipole and a monopole to the map, excluding masked pixels, with
        option to weight pixels and permute by shot noise. Simplification of
        Healpy's dipole fit function.


        Requires RING ordering, and nside <= 128.

        Parameters
        ----------
        m : float, array-like
            The map to be fitted. Masking is done automatically by
            using the SkyMap mask, set beforehand.

        w : float, array-like
            Weighting function to be applied to map. Weighting is done as
            m_weighted = m * w (default=None)

        poiss : bool
            Resample map using Poisson random numbers (shot noise). Weighting
            w will be applied after resampling (default=False)

        seed : int
            Random seed to use (default=None)

        Returns
        -------
        mono : float
            Monopole
        d : float
            Dipole amplitude
        vec : numpy.array
            Normalized dipole orientation vector
        coords : tuple
            Dipole orientation coordinates
        """

        self.set_ordinate_matrix(m, w, poiss=poiss, seed=seed)
        x = np.linalg.solve(self.a, self.b)
        mono = x[0]
        vec = x[1:4]
        norm = (vec @ vec) ** 0.5
        d = norm / mono
        coords = vec2dir(vec / norm)
        return mono, d, vec, coords


    def mkdipole(self, mono, d, vec):
        """mkdipole(mono, d, vec):

        Return a dipole map with monopole mono and dipole amplitude d,
        oriented in the direction described by vec.
        """
        dipole = np.dot(vec / (vec @ vec) ** 0.5, self.xyz)
        return mono * (1 + d * dipole)


if __name__ == "__main__":
    print("\nhpmap_utilities.py: " +
          "support functions for HEALPix maps and dipole work, written by " +
          "Nathan Secrest and Sebastian von Hausegger, with some modified " +
          "healpy code.")
    print("\nIf you find this code useful in your work, please consider " +
          "citing 2021ApJ...908L..51S and 2022arXiv220605624S, as well as " +
          "2019JOSS....4.1298Z (healpy) and 2005ApJ...622..759G (HEALPix).")

    lis = ['n','a','t','h','a','n','s','e','c','r','e','s','t',' ', '[','a',
           't',']',' ','m','s','n',' ','[','d','o','t',']',' ','c','o','m']

    print("\nQuestions or bug reporting: %s" % ''.join(lis))
