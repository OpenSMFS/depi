import numpy as np

valid_model_names = ('gaussian', 'wlc', 'radial_gaussian')


def assert_valid_model_name(name):
    name = name.lower()
    if name not in valid_model_names:
        raise TypeError(f'Distance model name "{name}" not recognized.'
                        f'Valid model names are:\n {valid_model_names})')


def gaussian(r, mu=0, sig=1):
    u = (r - mu) / sig
    return np.exp(-(u**2) / 2) / (sig * np.sqrt(2 * np.pi))


def wormlike_chain(r, L, lp, offset=0):
    res = np.zeros_like(r)
    ro = r - offset
    valid = (ro < L) * (ro > 0)
    res[~valid] = 0
    ro_valid = ro[valid]
    F = 1 - (ro_valid / L)**2
    res[valid] = np.exp(- (9 / 8) * (L / lp) / F) * ro_valid**2 / F**(9 / 2)
    return res


def radial_gaussian(r, mu, sig):
    c = np.sqrt(2 / np.pi)
    r0 = (r - mu)
    valid = r0 > 0
    res = np.zeros_like(r)
    res[~valid] = 0
    r0_squared = r0[valid]**2
    res[valid] = c * r0_squared * np.exp(-r0_squared / (2 * sig**2)) / sig**3
    return res


def _get_radial_gaussian_pdf(dr, mu, sig):
    s = sig**2 * (3 * np.pi - 8) / np.pi
    r = np.arange(dr, mu + 5 * s + dr, dr)
    rg_pdf = radial_gaussian(r, mu=mu, sig=sig)
    return r, rg_pdf


def _get_wlc_pdf(dr, L, lp, offset):
    r = np.arange(dr + offset, L + dr + offset, dr)
    wlc_pdf = wormlike_chain(r, L=L, lp=lp, offset=offset)
    wlc_pdf /= np.trapz(wlc_pdf, r)
    return r, wlc_pdf


def _get_dd_pdf(dr, params):
    params = params.copy()
    name = params.pop('name')
    if name.lower().startswith('wlc'):
        return _get_wlc_pdf(dr, **params)
    elif name.lower().startswith('radial_gauss'):
        return _get_radial_gaussian_pdf(dr, **params)
    else:
        raise TypeError(f'Distribution name `{name}` not recognized.')


def _get_norm_cdf(du, u_max):
    temp = np.arange(0, u_max + du, du)
    u = np.hstack([-temp[::-1][:-1], temp])
    assert len(u) % 2 == 1, 'Size of `u` has to be an odd number.'
    idx_offset = (u.size - 1) // 2
    norm_pdf = gaussian(u)
    norm_cdf = np.cumsum(norm_pdf) * du
    return norm_cdf, idx_offset


def get_r_dist_distrib(du, u_max, dr, dd_params):
    """Computes R axis and index offset for mapping a distance distribution.

    Arguments:
        du (float): step-size for the x-axis of the unitary Gaussian PDF
        u_max (float): max range of x-axis for the unitary Gaussian PDF.
            The PFD is evaluated between -u_max and u_max.
        dr (float): step-size of the R axis of the new distance distribution.
            `dr` should smaller than `du`, for example `dr = 0.1 * du`.
        dd_params (dict): parameters of the new distance distribution.
            `dd_params['name']` must be a string with the name of the
            distance distribution (i.e. 'WLC' or 'radial_gaussian').
            The other elements need to match the parameter names of the
            chosen distance distribution.

    Returns:
        - r_dist (array): array of distances with same CDF values as
          a unitary Normal CDF evaluated on an array `u` going from
          `-u_max` to `u_max` with step `du`.
        - idx_offset (int): size of the positive side of the `u` array,
          not including 0.

    Note:
        Given a unitary (sigma = 1) normal random number `x`,
        a random number `R` from an arbitrary distribution can
        be obtained as::

            ix = int(round(x / du)) + idx_offset
            R = r_dist[ix]

        where `r_dist` and `idx_offset` are returned by this function,
        and `du` is the input argument to this function.
    """
    # CDF of the standard normal distribution with sigma = 1
    norm_cdf, idx_offset = _get_norm_cdf(du, u_max)
    # PDF and CDF of the new distance distribution
    r, dd_pdf = _get_dd_pdf(dr, dd_params)
    dd_cdf = np.cumsum(dd_pdf) * dr
    assert dd_cdf[-1] <= 1 + 1e-10, 'CDF is larger than 1!'
    assert 1 - dd_cdf[-1] < 1e-3, 'CDF axis range too small. '
    # Build the R-axis at positions matching the normal CDF values
    r_dist = np.interp(norm_cdf, dd_cdf, r)
    return r_dist, idx_offset
