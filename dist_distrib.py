import numpy as np


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


def get_r_wlc(du, u_max, dr, L, lp, offset=0):
    """Computes R axis and index offset for mapping a WLC distance distribution.

    Returns:
        - r_wlc (array): array of distances with same CDF values as
          a unitary Normal CDF evaluated on an array `u` going from
          `-u_max` to `u_max` with step `du`.
        - idx_offset (int): size of the positive side of the `u` array,
          not including 0.

    Note:
        Given a unitary random number `x`, a random number `R`
        from a WLC distribution can be obtained as::

            ix = int(round(x / du)) + idx_offset
            R = r_wlc[ix]

        where `r_wlc` and `idx_offset` are returned by this function,
        and `du` beign the input argument to this function.
    """
    temp = np.arange(0, 6 + du, du)
    u = np.hstack([-temp[::-1][:-1], temp])
    assert len(u) % 2 == 1, 'Size of `u` has to be an odd number.'
    idx_offset = (u.size - 1) // 2
    norm_pdf = gaussian(u)
    norm_cdf = np.cumsum(norm_pdf) * du
    r = np.arange(dr + offset, L + dr + offset, dr)
    wlc_pdf = wormlike_chain(r, L=L, lp=lp, offset=offset)
    wlc_pdf /= np.trapz(wlc_pdf, r)
    wlc_cdf = np.cumsum(wlc_pdf) * dr
    r_wlc = np.interp(norm_cdf, wlc_cdf, r)
    return r_wlc, idx_offset
