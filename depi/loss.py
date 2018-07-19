import numpy as np
import pandas as pd
from randomgen import RandomGenerator, Xoroshiro128

import pycorrelate as pyc
import depi
from depi import fret
from depi import tcspc


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Utility functions for nanotimes, FCS, etc...
#


def add_irf_nanot(burstsph_sim, irf, seed=1):
    np.random.seed(seed)
    nt_d = burstsph_sim.loc[burstsph_sim.stream == 'DexDem', 'nanotime']
    nt_a = burstsph_sim.loc[burstsph_sim.stream == 'DexAem', 'nanotime']
    IRF_t_sim = irf.index - irf.index[0]
    # smooth1 is the IRF normalized so that the sum of bin counts is 1
    # With this normalization the area of the IRF is not 1.
    _DD_valid = ~irf.DD_smooth1.isnull()
    _DA_valid = ~irf.DA_smooth1.isnull()
    IRF_DD_rnd = np.random.choice(IRF_t_sim[_DD_valid],
                                  p=irf.DD_smooth1.loc[_DD_valid],
                                  size=len(nt_d))
    IRF_DA_rnd = np.random.choice(IRF_t_sim[_DA_valid],
                                  p=irf.DA_smooth1.loc[_DA_valid],
                                  size=len(nt_a))
    nt_d_conv = nt_d + IRF_DD_rnd  # - IRF_DD_rnd.mean()
    nt_a_conv = nt_a + IRF_DA_rnd  # - IRF_DA_rnd.mean()
    assert nt_d_conv.shape[0] + nt_a_conv.shape[0] == burstsph_sim.shape[0]
    burstsph_sim['nanotime_conv'] = pd.concat([nt_d_conv, nt_a_conv]).sort_index()
    assert (burstsph_sim.loc[burstsph_sim.stream == 'DexDem', 'nanotime_conv'] == nt_d_conv).all()
    assert (burstsph_sim.loc[burstsph_sim.stream == 'DexAem', 'nanotime_conv'] == nt_a_conv).all()
    return burstsph_sim


def nanot_hist_from_burstph(burstsph, bins, col='nanotime', offset=0):
    """Return D and A nanotime histograms from burst photon-data DataFrame.
    """
    DD = burstsph.stream == 'DexDem'
    DA = burstsph.stream == 'DexAem'
    nanotimes_d = burstsph.loc[DD, col] - offset
    nanotimes_a = burstsph.loc[DA, col] - offset
    hist_params = dict(bins=bins, density=False)
    nanot_hist_d, _ = np.histogram(nanotimes_d, **hist_params)
    nanot_hist_a, _ = np.histogram(nanotimes_a, **hist_params)
    return nanot_hist_d, nanot_hist_a


def calc_nanot_hist_irf_da(bph_sim, irf, nt_bins, tcspc_unit,
                           nanot_hist_d_exp, nanot_hist_a_exp, irf_seed=1):
    """Compute simulated D and A fluorescnecen decays histograms with IRF.
    Results are normalized to the same area as the experimental decays passed as
    argument.
    """
    # Add IRF to nanotimes
    bph_sim = add_irf_nanot(bph_sim, irf, seed=irf_seed)  # adds column `nanotime_conv`
    # Convert nanotimes in raw units as the experiment
    bph_sim['nanotime_conv_unit'] = (
        (bph_sim.nanotime_conv * (1e-9 / tcspc_unit)).round().astype('int'))
    # Find the offset, the raise-time of the decay
    offset_Dex_sim = tcspc.decay_hist_offset(bph_sim.nanotime_conv_unit,
                                             tcspc_unit, 4095, rebin=4) * 1e9
    # Compute DexDem and DexAem nanotime histograms
    nanot_hist_d_sim, nanot_hist_a_sim = nanot_hist_from_burstph(
        bph_sim, bins=nt_bins, col='nanotime_conv_unit',
        offset=offset_Dex_sim / (1e9 * tcspc_unit)
    )
    # Normalize to the same area as experimental decays
    nanot_hist_d_sim = nanot_hist_d_sim * nanot_hist_d_exp.sum() / nanot_hist_d_sim.sum()
    nanot_hist_a_sim = nanot_hist_a_sim * nanot_hist_a_exp.sum() / nanot_hist_a_sim.sum()
    return nanot_hist_d_sim, nanot_hist_a_sim


def calc_fcs_dd_da(burstsph, bins, seed=1):
    ts_d = burstsph.loc[burstsph.stream == 'DexDem', 'timestamp'].values.copy()
    ts_a = burstsph.loc[burstsph.stream == 'DexAem', 'timestamp'].values.copy()
    np.random.seed(seed)
    ts_d += np.random.randint(-25, 25, size=ts_d.size)
    ts_a += np.random.randint(-25, 25, size=ts_a.size)
    CC_DA = pyc.pcorrelate(ts_d, ts_a, bins, normalize=True) + 1
    AC_DD = pyc.pcorrelate(ts_d, ts_d, bins, normalize=True) + 1
    return CC_DA, AC_DD


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Metrics functions
#


def mse_metric(y_sim, y_data, weights=None):
    """Mean Square Error statistics"""
    if weights is None:
        weights = np.ones_like(y_sim)
    return (((y_sim - y_data) / weights)**2).mean()


def loglike_metric(y_sim, y_data):
    """Log-likelihood statistics for Poisson data (example: bin counts)
    """
    v = y_sim > 0
    return (y_sim[v] - y_data[v] * np.log(y_sim[v])).sum()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Loss functions starting with `iloss` take the DataFrame of simulated photons as input
#


def iloss_loglike_E(bph_sim, E_exp, E_bins=np.arange(0, 1.1, 0.03)):
    """Log-likelihood of simulated E histograms vs experimental
    """
    E = fret.calc_E_burst(bph_sim)
    E_hist_sim, _ = np.histogram(E, bins=E_bins)
    E_hist_exp, _ = np.histogram(E_exp, bins=E_bins)
    ll = loglike_metric(E_hist_sim, E_hist_exp)
    ll0 = loglike_metric(E_hist_exp, E_hist_exp)
    return ll - ll0


def iloss_loglike_nanot(bph_sim, irf, nt_bins, tcspc_unit,
                        nanot_hist_d_exp, nanot_hist_a_exp,
                        loglike_d_std, loglike_a_std,
                        irf_seed=1, return_da_losses=False):
    """Log-likelihood of simulated nantimes histograms vs experimental
    """
    nanot_hist_d_sim, nanot_hist_a_sim = calc_nanot_hist_irf_da(
        bph_sim, irf, nt_bins, tcspc_unit, nanot_hist_d_exp, nanot_hist_a_exp,
        irf_seed=irf_seed)
    # Compute log-likelihood metrics
    ll0_d = loglike_metric(nanot_hist_d_exp, nanot_hist_d_exp)
    ll0_a = loglike_metric(nanot_hist_a_exp, nanot_hist_a_exp)
    loglike_d = loglike_metric(nanot_hist_d_sim, nanot_hist_d_exp) - ll0_d
    loglike_a = loglike_metric(nanot_hist_a_sim, nanot_hist_a_exp) - ll0_a
    loglike_tot = loglike_d / loglike_d_std + loglike_a / loglike_a_std
    if return_da_losses:
        return loglike_tot, loglike_d, loglike_a
    else:
        return loglike_tot


def iloss_residuals_fcs(bph_sim, bins, CC_DA_exp, AC_DD_exp,
                        CC_DA_std_dev, AC_DD_std_dev,
                        CC_DA_loss_std=1, AC_DD_loss_std=1,
                        tot_loss_std=1):
    """Residuals of FCS curves vs experimental
    """
    CC_DA_sim, AC_DD_sim = calc_fcs_dd_da(bph_sim, bins)
    loss_cc_da = mse_metric(CC_DA_sim, CC_DA_exp, weights=CC_DA_std_dev)
    loss_ac_dd = mse_metric(AC_DD_sim, AC_DD_exp, weights=AC_DD_std_dev)
    loss_fcs = (loss_cc_da / CC_DA_loss_std + loss_ac_dd / AC_DD_loss_std)
    return loss_fcs


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Functions starting with `loss` take the simulation parameters as input arguments.
# The first 4 positional arguments are always: params_vary, params_const, ts, recolor_func
# `recolor_func` is a specific "recoloring function" (see next section) which determines
# the D-A distance distribution and which arguments are varying.
#
# The keyword argument `cache` is passed to the recoloring function.
#

def loss_E(params_vary, params_const, ts, recolor_func, E_exp, E_bins,
           cache=False, seed=1):
    """Loss function for E histogram.
    """
    bph_sim = recolor_func(params_vary, params_const, ts, cache=cache, seed=seed)
    loss_E = iloss_loglike_E(bph_sim, E_exp=E_exp, E_bins=E_bins)
    return loss_E


def loss_nanot(params_vary, params_const, ts, recolor_func,
               irf, nt_bins, tcspc_unit, nanot_hist_d_exp, nanot_hist_a_exp,
               loglike_d_std, loglike_a_std, irf_seed=1,
               cache=False, seed=1):
    """Loss function for fluorescence decays.
    """
    bph_sim = recolor_func(params_vary, params_const, ts, cache=cache, seed=seed)
    loss_nanot = iloss_loglike_nanot(
        bph_sim, irf=irf, nt_bins=nt_bins, tcspc_unit=tcspc_unit,
        nanot_hist_d_exp=nanot_hist_d_exp, nanot_hist_a_exp=nanot_hist_a_exp,
        loglike_d_std=loglike_d_std, loglike_a_std=loglike_a_std, irf_seed=irf_seed)
    return loss_nanot


def loss_E_fcs(params_vary, params_const, ts, recolor_func, E_exp, E_bins,
               fcs_bins, CC_DA_exp, AC_DD_exp, CC_DA_std_dev, AC_DD_std_dev,
               CC_DA_loss_std, AC_DD_loss_std,
               E_loss_std=1, FCS_loss_std=1,
               cache=False, seed=1):
    """Loss function combining E histogram and FCS curves (CC_DA and AC_DD).
    """
    bph_sim = recolor_func(params_vary, params_const, ts, cache=cache, seed=seed)
    loss_E = iloss_loglike_E(bph_sim, E_exp=E_exp, E_bins=E_bins)
    loss_fcs = iloss_residuals_fcs(
        bph_sim, bins=fcs_bins, CC_DA_exp=CC_DA_exp, AC_DD_exp=AC_DD_exp,
        CC_DA_std_dev=CC_DA_std_dev, AC_DD_std_dev=AC_DD_std_dev,
        CC_DA_loss_std=CC_DA_loss_std, AC_DD_loss_std=AC_DD_loss_std)
    return (loss_E / E_loss_std + loss_fcs / FCS_loss_std)


def loss_E_nanot(params_vary, params_const, ts, recolor_func, E_exp, E_bins,
                 irf, nt_bins, tcspc_unit, nanot_hist_d_exp, nanot_hist_a_exp,
                 loglike_d_std, loglike_a_std, irf_seed=1,
                 E_loss_std=1, nanot_loss_std=1,
                 cache=False, seed=1):
    """Loss function combining E histogram and fluorescence decays.
    """
    bph_sim = recolor_func(params_vary, params_const, ts, cache=cache, seed=seed)
    loss_E = iloss_loglike_E(bph_sim, E_exp=E_exp, E_bins=E_bins)
    loss_nanot = iloss_loglike_nanot(
        bph_sim, irf=irf, nt_bins=nt_bins, tcspc_unit=tcspc_unit,
        nanot_hist_d_exp=nanot_hist_d_exp, nanot_hist_a_exp=nanot_hist_a_exp,
        loglike_d_std=loglike_d_std, loglike_a_std=loglike_a_std, irf_seed=irf_seed)
    return (loss_E / E_loss_std + loss_nanot / nanot_loss_std)


def loss_E_fcs_nanot(params_vary, params_const, ts, recolor_func, E_exp, E_bins,
                     fcs_bins, CC_DA_exp, AC_DD_exp, CC_DA_std_dev, AC_DD_std_dev,
                     CC_DA_loss_std, AC_DD_loss_std,
                     irf, nt_bins, tcspc_unit, nanot_hist_d_exp, nanot_hist_a_exp,
                     loglike_d_std, loglike_a_std, irf_seed=1,
                     E_loss_std=1, FCS_loss_std=1, nanot_loss_std=1,
                     cache=False, seed=1):
    """Loss function combining E histogram, FCS and fluorescence decays.
    """
    bph_sim = recolor_func(params_vary, params_const, ts, cache=cache, seed=seed)
    loss_E = iloss_loglike_E(bph_sim, E_exp=E_exp, E_bins=E_bins)
    loss_fcs = iloss_residuals_fcs(
        bph_sim, bins=fcs_bins, CC_DA_exp=CC_DA_exp, AC_DD_exp=AC_DD_exp,
        CC_DA_std_dev=CC_DA_std_dev, AC_DD_std_dev=AC_DD_std_dev,
        CC_DA_loss_std=CC_DA_loss_std, AC_DD_loss_std=AC_DD_loss_std)
    loss_nanot = iloss_loglike_nanot(
        bph_sim, irf=irf, nt_bins=nt_bins, tcspc_unit=tcspc_unit,
        nanot_hist_d_exp=nanot_hist_d_exp, nanot_hist_a_exp=nanot_hist_a_exp,
        loglike_d_std=loglike_d_std, loglike_a_std=loglike_a_std, irf_seed=irf_seed)
    return (loss_E / E_loss_std + loss_fcs / FCS_loss_std + loss_nanot / nanot_loss_std)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# The functions `recolor_burstsph_*` (or "recoloring functions") call the recoloring
# simulation for different distance distribution and different sets of varying
# parameters.
# The first 4 arguments are always: params, params_const, ts, cache
#


def _recolor_burstsph(ts, params, cache=False, seed=1):
    if cache:
        bph = depi.recolor_burstsph_cache(ts, **params)
    else:
        rg = RandomGenerator(Xoroshiro128(seed))
        bph = depi.recolor_burstsph(ts, rg=rg, **params)
    return bph


def recolor_burstsph_gauss_noblink(params, params_const, ts, cache=True, seed=1):
    R_mean, R_sigma, τ_relax = params
    params_tot = params_const.copy()
    params_tot.update(R_mean=R_mean, R_sigma=R_sigma, τ_relax=τ_relax)
    return _recolor_burstsph(ts, params_tot, cache=cache, seed=1)


def recolor_burstsph_gauss(params, params_const, ts, cache=True, seed=1):
    R_mean, R_sigma, τ_relax, prob_A_dark, τ_A_dark = params
    τ_A_dark *= 1e6
    params_tot = params_const.copy()
    params_tot.update(R_mean=R_mean, R_sigma=R_sigma,
                      τ_relax=τ_relax, prob_A_dark=prob_A_dark,
                      τ_A_dark=τ_A_dark)
    return _recolor_burstsph(ts, params_tot, cache=cache, seed=1)


def recolor_burstsph_wlc(params, params_const, ts, cache=True, seed=1):
    L, lp, τ_relax, prob_A_dark, τ_A_dark = params
    τ_A_dark *= 1e6
    params_tot = params_const.copy()
    params_tot.update(L=L, lp=lp,
                      τ_relax=τ_relax, prob_A_dark=prob_A_dark,
                      τ_A_dark=τ_A_dark)
    return _recolor_burstsph(ts, params_tot, cache=cache, seed=1)


def recolor_burstsph_wlco(params, params_const, ts, cache=True, seed=1):
    L, lp, offset, τ_relax, prob_A_dark, τ_A_dark = params
    τ_A_dark *= 1e6
    params_tot = params_const.copy()
    params_tot.update(L=L, lp=lp, offset=offset,
                      τ_relax=τ_relax, prob_A_dark=prob_A_dark,
                      τ_A_dark=τ_A_dark)
    return _recolor_burstsph(ts, params_tot, cache=cache, seed=1)


def recolor_burstsph_gauss_nodiff(params, params_const, ts, cache=True, seed=1):

    R_mean, R_sigma, prob_A_dark, τ_A_dark = params
    τ_A_dark *= 1e6
    params_tot = params_const.copy()
    params_tot.update(R_mean=R_mean, R_sigma=R_sigma,
                      prob_A_dark=prob_A_dark,
                      τ_A_dark=τ_A_dark)
    return _recolor_burstsph(ts, params_tot, cache=cache, seed=1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# These function are here for compatibility with old notebooks.
# Do not use in new code. Eventually they will be deleted.
# All the functions below can be replace by the function `loss_E`
# which takes a recoloring function. The recoloring function determines
# the distance distribution and which arguments are varying.
#

def loss_function_E(params, params_const, ts, E_exp,
                    bins=np.arange(0, 1.1, 0.03), cache=True):
    """Loss function for E histograms and Gaussian distance distribution
    """
    bph = recolor_burstsph_gauss(params, params_const, ts, cache=cache)
    return iloss_loglike_E(bph, E_exp, E_bins=bins)


def loss_function_E_wlc(params, params_const, ts, E_exp,
                        bins=np.arange(0, 1.1, 0.03), cache=True):
    """Loss function for E histograms and WLC distance distribution (no offset)
    """
    bph = recolor_burstsph_wlc(params, params_const, ts, cache=cache)
    return iloss_loglike_E(bph, E_exp, E_bins=bins)


def loss_function_E_wlco(params, params_const, ts, E_exp,
                         bins=np.arange(0, 1.1, 0.03), cache=True):
    """Loss function for E histograms and WLC distance distribution with offset
    """
    bph = recolor_burstsph_wlco(params, params_const, ts, cache=cache)
    return iloss_loglike_E(bph, E_exp, E_bins=bins)


def loss_function_E_gauss_nodiff(params, params_const, ts, E_exp,
                                 bins=np.arange(0, 1.1, 0.03), cache=True):
    """Loss function for E histograms and Gaussian distance (fixed τ_relax)
    """
    bph = recolor_burstsph_gauss_nodiff(params, params_const, ts, cache=cache)
    return iloss_loglike_E(bph, E_exp, E_bins=bins)
