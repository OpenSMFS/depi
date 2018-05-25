import numpy as np
import pandas as pd


def correct_E_gamma_leak_dir(Eraw, gamma=1, leakage=0, dir_ex_t=0):
    """Compute corrected FRET efficiency from proximity ratio `Eraw`.

    This function is the inverse of :func:`uncorrect_E_gamma_leak_dir`.

    Arguments:
        Eraw (float or array): proximity ratio (only background correction,
            no gamma, leakage or direct excitation)
        gamma (float): gamma factor
        leakage (float): leakage coefficient
        dir_ex_t (float): coefficient expressing the direct excitation as
            n_dir = dir_ex_t * (na + gamma*nd). In terms of physical
            parameters it is the ratio of acceptor over donor absorption
            cross-sections at the donor-excitation wavelength.

    Returns
        Corrected FRET effciency
    """
    Eraw = np.asarray(Eraw)
    return ((Eraw * (leakage + dir_ex_t * gamma + 1) - leakage - dir_ex_t * gamma)
            / (Eraw * (leakage - gamma + 1) - leakage + gamma))


def uncorrect_E_gamma_leak_dir(E, gamma=1, leakage=0, dir_ex_t=0):
    """Compute proximity ratio from corrected FRET efficiency `E`.

    This function is the inverse of :func:`correct_E_gamma_leak_dir`.

    Arguments:
        E (float or array): corrected FRET efficiency
        gamma (float): gamma factor
        leakage (float): leakage coefficient
        dir_ex_t (float): direct excitation coefficient expressed as
            n_dir = dir_ex_t * (na + gamma*nd). In terms of physical
            parameters it is the ratio of absorption cross-section at
            donor-excitation wavelengths of acceptor over donor.

    Returns
        Proximity ratio (reverses gamma, leakage and direct excitation)
    """
    E = np.asarray(E)
    return ((E * (gamma - leakage) + leakage + dir_ex_t * gamma)
            / (E * (gamma - leakage - 1) + leakage + dir_ex_t * gamma + 1))


def E_from_dist(x, R0):
    """Return E computed from D-A distance and R0
    """
    x = np.asarray(x)
    E = 1 / (1 + (x / R0)**6)
    if not np.isscalar(x):
        E[x < 0] = 1
    elif x < 0:
        E = 0
    return E


def dist_from_E(E, R0):
    """Return the D-A distance for a give E and R0
    """
    E = np.asarray(E)
    return R0 * (1 / E - 1)**(1 / 6)


def mean_E_from_gauss_PoR(R_mean, R_sigma, R0):
    """Mean E from integration of a Gaussian distribution of distances (PoR)
    """
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    R_axis = np.arange(0, R_mean + 5 * R_sigma, R_sigma / 200)
    R_pdf = gaussian(R_axis, R_mean, R_sigma)
    R_pdf /= np.trapz(R_pdf, R_axis)
    EoR = E_from_dist(R_axis, R0)
    E_mean_integral = np.trapz(EoR * R_pdf, R_axis)
    return E_mean_integral


def calc_E_burst(burstsph):
    """Calculate E from photon-data (photon timestamp and stream)

    Arguments:
        burstsph (DataFrame): photon data DataFrame with at least two
            columns: 'timestamp' and 'stream'. The index needs to have
            two levels: ('burst', 'ph').

    Returns:
        Array of E values, one for each burst.
    """
    bursts = pd.DataFrame(burstsph.groupby('burst').size(),
                          columns=['size_raw'])
    bursts['istart'] = np.hstack([[0], np.cumsum(bursts.size_raw)[:-1]])
    bursts['istop'] = np.cumsum(bursts.size_raw.values)
    A_em = burstsph.stream == 'DexAem'
    D_ex = (burstsph.stream == 'DexDem') | (burstsph.stream == 'DexAem')
    E = [A_em[istart:istop].sum() / D_ex[istart:istop].sum()
         for istart, istop, bsize in
         zip(bursts.istart, bursts.istop, bursts.size_raw)]
    return E
