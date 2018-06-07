import pytest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from randomgen import RandomGenerator, Xoroshiro128

from depi import depi_ref
from depi import depi_cy
from depi import fret
from depi import depi
from depi import dist_distrib as dd


def load_burstsph():
    fname = Path('results/E3BD_GdmHCl_3.0M_merge_FRET_burst_photons.csv')
    burstsph = pd.read_csv(fname, skiprows=1, index_col=(0, 1))
    header = fname.read_text().split('\n')[0]
    meta = json.loads(header)
    meta['timestamp_unit_hw'] = meta['timestamp_unit']
    meta['timestamp_unit'] = 1e-9
    scale = meta['timestamp_unit'] * 1e9
    scale = int(scale) if round(scale) == scale else scale
    burstsph.timestamp *= scale
    return burstsph, meta


def test_calc_intrisic_nanotime():
    rg = RandomGenerator(Xoroshiro128(1))
    n = int(1e6)
    τ = 2
    fract = None
    nanot = depi._calc_intrisic_nanotime(n, τ, fract, rg)
    assert np.allclose(nanot.mean(), τ, rtol=5e-3)

    n = int(1e6)
    τ = np.array([1, 7])
    fract = np.array([0.3, 0.7])
    nanot = depi._calc_intrisic_nanotime(n, τ, fract, rg)
    assert np.allclose(sum(fract * τ), nanot.mean(), rtol=1e-2)


def test_py_vs_cy():
    ns = 1.0
    nm = 1.0
    δt = 1e-1 * ns
    R0 = 6 * nm
    R_mean = 5.5 * nm
    R_sigma = 0.01 * nm
    τ_relax = 0.08 * ns
    τ_D = 4 * ns
    k_D = 1 / τ_D

    burstsph, _ = load_burstsph()
    ts = burstsph.timestamp.values[:1000]

    rg = RandomGenerator(Xoroshiro128(1))
    A_em, R_ph, T_ph = depi_ref.sim_DA_from_timestamps(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
    rg = RandomGenerator(Xoroshiro128(1))
    R_phc, T_phc = depi_cy.sim_DA_from_timestamps_cy(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
    assert all([np.allclose(R_ph, R_phc),
                np.allclose(T_ph, T_phc)])
    # Test R from nanotime vs R_mean
    R_est = fret.dist_from_E(1 - T_ph.mean() / τ_D, R0)
    assert abs(R_est - R_mean) < 0.12 * nm
    # Test E from nanotime vs ratiometric vs from P(R)
    E_PoR = fret.mean_E_from_gauss_PoR(R_mean, R_sigma, R0)
    E_ratio = A_em.sum() / A_em.size
    E_lifetime = 1 - T_ph.mean() / τ_D
    assert abs(E_PoR - E_ratio) < 0.03
    assert abs(E_PoR - E_lifetime) < 0.03

    rg = RandomGenerator(Xoroshiro128(1))
    A_em, R_ph, T_ph = depi_ref.sim_DA_from_timestamps2_p(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
    rg = RandomGenerator(Xoroshiro128(1))
    R_phc, T_phc = depi_cy.sim_DA_from_timestamps2_p_cy(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
    assert all([np.allclose(R_ph, R_phc),
                np.allclose(T_ph, T_phc)])
    # Test R from nanotime vs R_mean
    R_est = fret.dist_from_E(1 - T_ph.mean() / τ_D, R0)
    assert abs(R_est - R_mean) < 0.12 * nm
    # Test E from nanotime vs ratiometric vs from P(R)
    E_PoR = fret.mean_E_from_gauss_PoR(R_mean, R_sigma, R0)
    E_ratio = A_em.sum() / A_em.size
    E_lifetime = 1 - T_ph.mean() / τ_D
    assert abs(E_PoR - E_ratio) < 0.03
    assert abs(E_PoR - E_lifetime) < 0.03


def test_approx_vs_correct():
    """
    With α=np.inf, ndt=0 the adaptive functions should give the same
    results as the approximeated function.
    """
    burstsph, _ = load_burstsph()
    ns = 1.0
    nm = 1.0
    δt = 1e-1 * ns
    R0 = 6 * nm
    R_mean = 6.5 * nm
    R_sigma = 0.01 * nm
    τ_relax = 0.08 * ns
    τ_D = 4 * ns
    k_D = 1 / τ_D
    ts = burstsph.timestamp.values[:1000]

    for R_sigma in (0.01, 0.1, 1, 10):
        for R_mean in (4, 5, 6, 7, 8):
            R_sigma *= nm
            R_mean *= nm
            rg = RandomGenerator(Xoroshiro128(1))
            A_em, R_ph, T_ph = depi_ref.sim_DA_from_timestamps(
                ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
            rg = RandomGenerator(Xoroshiro128(1))
            A_emp, R_php, T_php = depi_ref.sim_DA_from_timestamps_p(
                ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg, α=np.inf, ndt=0)
            rg = RandomGenerator(Xoroshiro128(1))
            A_emp2, R_php2, T_php2 = depi_ref.sim_DA_from_timestamps_p2(
                ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg, α=np.inf, ndt=0)
            assert all([np.allclose(A_em, A_emp),
                        np.allclose(R_ph, R_php),
                        np.allclose(T_ph, T_php)])
            assert all([np.allclose(A_em, A_emp2),
                        np.allclose(R_ph, R_php2),
                        np.allclose(T_ph, T_php2)])


@pytest.mark.skip(reason="This test takes too long. Run it responsibly.")
def test_dt_tollerance():
    """
    Using small enough dt approximated and correct expression should be similar
    """
    burstsph, _ = load_burstsph()
    ns = 1.0
    nm = 1.0
    δt = 5e-4 * ns
    R0 = 6 * nm
    R_mean = 6.5 * nm
    R_sigma = 1 * nm
    τ_relax = 0.5 * ns
    τ_D = 4 * ns
    k_D = 1 / τ_D
    ts = burstsph.timestamp.values[:10000]
    rg = RandomGenerator(Xoroshiro128(1))
    A_em, R_ph, T_ph = depi_ref.sim_DA_from_timestamps(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
    rg = RandomGenerator(Xoroshiro128(1))
    A_emp, R_php, T_php = depi_ref.sim_DA_from_timestamps_p(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
    rg = RandomGenerator(Xoroshiro128(1))
    A_emp2, R_php2, T_php2 = depi_ref.sim_DA_from_timestamps_p2(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
    assert not all([not np.allclose(A_em, A_emp),
                    not np.allclose(R_ph, R_php),
                    not np.allclose(T_ph, T_php)])
    assert not all([not np.allclose(A_em, A_emp2),
                    not np.allclose(R_ph, R_php2),
                    not np.allclose(T_ph, T_php2)])
    # Test R from nanotime vs R_mean
    R_a = fret.dist_from_E(1 - T_ph.mean() / τ_D, R0)
    R_c = fret.dist_from_E(1 - T_php.mean() / τ_D, R0)
    assert abs(R_c - R_a) < 0.12 * nm
    # Test E from nanotime vs ratiometric vs from P(R)
    E_PoR = fret.mean_E_from_gauss_PoR(R_mean, R_sigma, R0)
    E_ratio_a = A_em.sum() / A_em.size
    E_lifetime_a = 1 - T_ph.mean() / τ_D
    E_ratio_c = A_emp.sum() / A_emp.size
    E_lifetime_c = 1 - T_php.mean() / τ_D
    E_ratio_c2 = A_emp2.sum() / A_emp2.size
    E_lifetime_c2 = 1 - T_php2.mean() / τ_D
    assert abs(E_PoR - E_ratio_a) < 0.03
    assert abs(E_PoR - E_lifetime_a) < 0.03
    assert abs(E_ratio_c - E_ratio_a) < 0.03
    assert abs(E_ratio_c2 - E_ratio_a) < 0.03
    assert abs(E_lifetime_c - E_ratio_a) < 0.03
    assert abs(E_lifetime_c2 - E_ratio_a) < 0.03


def test_cdf_vs_dt_python():
    """
    Test CDF vs small-dt correction in python code
    """
    burstsph, _ = load_burstsph()
    ns = 1.0
    nm = 1.0
    δt = 1e-2 * ns
    R0 = 6 * nm
    R_mean = 6.5 * nm
    R_sigma = 1 * nm
    τ_relax = 0.2 * ns
    τ_D = 4 * ns
    k_D = 1 / τ_D
    ts = burstsph.timestamp.values[:2000]
    rg = RandomGenerator(Xoroshiro128(1))
    A_em, R_ph, T_ph = depi_ref.sim_DA_from_timestamps2_p(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg, ndt=0, alpha=np.inf)
    rg = RandomGenerator(Xoroshiro128(1))
    A_emp, R_php, T_php = depi_ref.sim_DA_from_timestamps2_p2(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg, ndt=0, alpha=np.inf)
    assert not all([not np.allclose(A_em, A_emp),
                    not np.allclose(R_ph, R_php),
                    not np.allclose(T_ph, T_php)])


def test_cdf_vs_dt_cy():
    """
    Test CDF vs small-dt correction in cython code
    """
    burstsph, _ = load_burstsph()
    ns = 1.0
    nm = 1.0
    δt = 1e-2 * ns
    R0 = 6 * nm
    R_mean = 6.5 * nm
    R_sigma = 1 * nm
    τ_relax = 0.2 * ns
    τ_D = 4. * ns
    k_D = 1. / τ_D
    D_fract = np.atleast_1d(1.)
    ts = burstsph.timestamp.values[:100000]
    rg = RandomGenerator(Xoroshiro128(1))
    R_ph, T_ph = depi_cy.sim_DA_from_timestamps2_p_cy(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg, ndt=0, alpha=np.inf)
    rg = RandomGenerator(Xoroshiro128(1))
    R_php, T_php = depi_cy.sim_DA_from_timestamps2_p2_cy(
        ts, δt, np.atleast_1d(k_D), D_fract, R0, R_mean, R_sigma, τ_relax, rg=rg,
        ndt=0, alpha=np.inf)
    assert not all([not np.allclose(R_ph, R_php),
                    not np.allclose(T_ph, T_php)])


def test_2states_py_vs_cy():
    burstsph, _ = load_burstsph()
    ns = 1.0
    nm = 1.0
    R0 = 6 * nm
    R_mean = np.array([5.5 * nm, 8 * nm])
    R_sigma = np.array([0.8 * nm, 1 * nm])
    τ_relax = np.array([0.1 * ns, 0.1 * ns])
    k_s = np.array([1 / (1e6 * ns), 1 / (1e6 * ns)])  # transition rates: [0->1, 1->0]
    δt = 1e-2 * ns
    τ_D = 3.8 * ns

    k_D = 1 / τ_D
    D_fract = np.atleast_1d(1.)
    ts = burstsph.timestamp.values[:10000]

    rg = RandomGenerator(Xoroshiro128(1))
    A_em, R_ph, T_ph, S_ph = depi_ref.sim_DA_from_timestamps2_p2_2states(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, k_s, rg=rg)
    rg = RandomGenerator(Xoroshiro128(1))
    R_php, T_php, S_php = depi_cy.sim_DA_from_timestamps2_p2_2states_cy(
        ts, δt, np.atleast_1d(k_D), D_fract, R0, R_mean, R_sigma, τ_relax, k_s, rg=rg, ndt=0)
    assert np.allclose(R_php, R_ph)
    assert np.allclose(T_php, T_ph)
    assert np.allclose(S_php, S_ph)


def test_Nstates_py_vs_cy():
    burstsph, _ = load_burstsph()
    ns = 1.0
    nm = 1.0
    R0 = 6 * nm
    R_mean = np.array([5.5 * nm, 8 * nm])
    R_sigma = np.array([0.8 * nm, 1 * nm])
    τ_relax = np.array([0.1 * ns, 0.1 * ns])
    k_s01, k_s10 = np.array([1 / (1e6 * ns), 1 / (1e6 * ns)])
    K_s = np.array([[-k_s01, k_s01],
                    [k_s10, -k_s10]])
    δt = 1e-2 * ns
    τ_D = 3.8 * ns

    k_D = 1 / τ_D
    D_fract = np.atleast_1d(1.)
    ts = burstsph.timestamp.values[:5000]
    rg = RandomGenerator(Xoroshiro128(1))
    A_em, R_ph, T_ph, S_ph = depi_ref.sim_DA_from_timestamps2_p2_Nstates(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, K_s, rg=rg)
    rg = RandomGenerator(Xoroshiro128(1))
    R_php, T_php, S_php = depi_cy.sim_DA_from_timestamps2_p2_Nstates_cy(
        ts, δt, np.atleast_1d(k_D), D_fract, R0, R_mean, R_sigma, τ_relax, K_s, rg=rg, ndt=0)
    assert np.allclose(R_php, R_ph)
    assert np.allclose(T_php, T_ph)
    assert np.allclose(S_php, S_ph)


def test_2states_vs_Nstates():
    burstsph, _ = load_burstsph()
    ns = 1.0
    nm = 1.0
    R0 = 6 * nm
    R_mean = np.array([5.5 * nm, 8 * nm])
    R_sigma = np.array([0.8 * nm, 1 * nm])
    τ_relax = np.array([0.1 * ns, 0.1 * ns])
    k_s01, k_s10 = np.array([1 / (1e5 * ns), 1 / (1e6 * ns)])
    K_s = np.array([[-k_s01, k_s01],
                    [k_s10, -k_s10]])
    k_s = np.array([k_s01, k_s10])
    δt = 1e-2 * ns
    τ_D = 3.8 * ns

    k_D = 1 / τ_D
    D_fract = np.atleast_1d(1.)
    ts = burstsph.timestamp.values[:10000]

    rg = RandomGenerator(Xoroshiro128(1))
    R_ph, T_ph, S_ph = depi_cy.sim_DA_from_timestamps2_p2_2states_cy(
        ts, δt, np.atleast_1d(k_D), D_fract, R0, R_mean, R_sigma, τ_relax, k_s, rg=rg)
    rg = RandomGenerator(Xoroshiro128(1))
    R_php, T_php, S_php = depi_cy.sim_DA_from_timestamps2_p2_Nstates_cy(
        ts, δt, np.atleast_1d(k_D), D_fract, R0, R_mean, R_sigma, τ_relax, K_s, rg=rg, ndt=0)
    assert np.allclose(R_php, R_ph)
    assert np.allclose(T_php, T_ph)
    assert np.allclose(S_php, S_ph)


def test_corrections():
    N = 50000
    burstsph, _ = load_burstsph()
    burstsph = burstsph.iloc[:N]
    ns = nm = 1.
    params = dict(
        name='gaussian',
        # physical parameters
        R_mean=[6.37 * nm, 4 * ns],
        R_sigma=[0.01 * nm] * 2,
        R0=6. * nm,
        τ_relax=[200 * ns] * 2,
        τ_D=[3.8 * ns, 1 * ns],
        D_fract=[0.5, 0.5],
        τ_A=4 * ns,
        k_s=[1, 1],
        # simulation parameters
        δt=1e-2 * ns,
        ndt=10,
        α=0.1,
        gamma=2,
        lk=0.3,
        dir_ex_t=0.4,
    )
    # 2-states Gaussian
    rg = RandomGenerator(Xoroshiro128(1))
    burstsph_sim = depi.recolor_burstsph(burstsph.timestamp, rg=rg, **params)
    E_corr = fret.E_from_dist(np.array(params['R_mean']), R0=params['R0'])
    state0 = burstsph_sim.state == 0
    state1 = burstsph_sim.state == 1
    FA = burstsph_sim.stream == 'DexAem'
    Eraw0 = (FA & state0).sum() / state0.sum()
    Eraw1 = (FA & state1).sum() / state1.sum()
    Eraw = np.array([Eraw0, Eraw1])
    Eraw2 = fret.uncorrect_E_gamma_leak_dir(E_corr, params['gamma'], params['lk'],
                                            dir_ex_t=params['dir_ex_t'])
    assert np.allclose(Eraw, Eraw2, atol=5e-3, rtol=0)

    # 1-state Gaussian
    params.update(R_mean=params['R_mean'][0], R_sigma=params['R_sigma'][0],
                  τ_relax=params['τ_relax'][0])
    burstsph_sim = depi.recolor_burstsph(burstsph.timestamp, rg=rg, **params)
    E_corr = fret.E_from_dist(np.array(params['R_mean']), R0=params['R0'])
    FA = (burstsph_sim.stream == 'DexAem').sum()
    Eraw = FA / burstsph_sim.shape[0]
    Eraw2 = fret.uncorrect_E_gamma_leak_dir(E_corr, params['gamma'], params['lk'],
                                            dir_ex_t=params['dir_ex_t'])
    assert np.allclose(Eraw, Eraw2, atol=5e-3, rtol=0)

    # 1-state Radial Gaussian
    params.update(name='radial_gaussian', mu=4 * nm, sigma=0.01 * nm, offset=1,
                  du=0.01, u_max=6., dr=0.001)
    del params['R_mean'], params['R_sigma']
    d = dd.distribution(params)
    E_corr = d.mean_E(params['R0'])
    rg = RandomGenerator(Xoroshiro128(1))
    burstsph_sim = depi.recolor_burstsph(burstsph.timestamp, rg=rg, **params)
    FA = (burstsph_sim.stream == 'DexAem').sum()
    Eraw = FA / burstsph_sim.shape[0]
    Eraw2 = fret.uncorrect_E_gamma_leak_dir(E_corr, params['gamma'], params['lk'],
                                            dir_ex_t=params['dir_ex_t'])
    assert np.allclose(Eraw, Eraw2, atol=5e-3, rtol=0)


def test_background():
    N = 100000
    burstsph, meta = load_burstsph()
    burstsph = burstsph.iloc[:N]
    ns = nm = 1.
    params = dict(
        name='gaussian',
        # physical parameters
        R_mean=6.5 * nm,
        R_sigma=0.01 * nm,
        R0=6 * nm,
        τ_relax=200 * ns,
        τ_D=[3.8 * ns, 1 * ns],
        D_fract=[0.5, 0.5],
        τ_A=[4 * ns, 1 * ns],
        A_fract=[0.3, 0.7],
        bg_rate_d=1000,
        bg_rate_a=1500,
        ts_unit=meta['timestamp_unit'],
        tcspc_unit=30 * ns,
        # simulation parameters
        δt=1e-2 * ns,
        ndt=10,
        α=0.1,
        gamma=0.6,
        lk=0.4,
        dir_ex_t=0.3,
    )
    rg = RandomGenerator(Xoroshiro128(1))
    burstsph_sim = depi.recolor_burstsph(burstsph.timestamp, rg=rg, **params)

    def func(burst):
        bg_counts = burst.bg_ph.sum()
        bg_counts_d = burst.bg_ph.loc[burst.stream == 'DexDem'].sum()
        bg_counts_a = burst.bg_ph.loc[burst.stream == 'DexAem'].sum()
        burst_width = (burst.timestamp.iloc[-1] - burst.timestamp.iloc[0]) * 1e-9
        bg_rate = bg_counts / burst_width
        bg_rate_d = bg_counts_d / burst_width
        bg_rate_a = bg_counts_a / burst_width
        return pd.Series([bg_rate, bg_rate_d, bg_rate_a],
                         index=['bg', 'bg_d', 'bg_a'])
    bg = burstsph_sim.groupby('burst').apply(func)
    assert np.allclose(bg.bg.mean(), 2500, atol=0, rtol=3e-2)
    assert np.allclose(bg.bg_d.mean(), 1000, atol=0, rtol=6e-2)
    assert np.allclose(bg.bg_a.mean(), 1500, atol=0, rtol=6e-2)
