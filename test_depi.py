import pytest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from randomgen import RandomGenerator, Xoroshiro128
import depi
import depi_ref
import depi_cy
import fret


def load_burstsph():
    fname = Path('results/E3BD_GdmHCl_3.0M_merge_FRET_burst_photons.csv')
    burstsph = pd.read_csv(fname, skiprows=1, index_col=(0, 1))
    header = fname.read_text().split('\n')[0]
    meta = json.loads(header)
    scale = meta['timestamp_unit'] * 1e9
    scale = int(scale) if round(scale) == scale else scale
    burstsph.timestamp *= scale
    return burstsph


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

    burstsph = load_burstsph()
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
    burstsph = load_burstsph()
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
    burstsph = load_burstsph()
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
    burstsph = load_burstsph()
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
    burstsph = load_burstsph()
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
    burstsph = load_burstsph()
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
    burstsph = load_burstsph()
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
    burstsph = load_burstsph()
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
