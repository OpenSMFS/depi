import numpy as np
from depi import fret


def test_fretmath_roundtrip():
    """Run a few consistency checks for the correction functions.
    """
    Ex = np.arange(-0.2, 1.2, 0.01)

    # Tests tautology
    assert np.allclose(Ex, fret.correct_E_gamma_leak_dir(Ex))
    assert np.allclose(Ex, fret.uncorrect_E_gamma_leak_dir(Ex))

    # Test round-trip consistency
    for leakage in [0.01, 0.04, 0.2]:
        for dir_ex_t in [0.01, 0.03, 0.09]:
            for gamma in [0.2, 0.5, 0.8, 1.2, 1.8]:
                Ec = fret.correct_E_gamma_leak_dir(Ex, gamma, leakage, dir_ex_t)
                Eu = fret.uncorrect_E_gamma_leak_dir(Ec, gamma, leakage, dir_ex_t)
                assert np.allclose(Ex, Eu)


def test_fret_uncorrections():
    E = 0.4
    gamma = 2.5
    lk = 0.3
    dir_ex_t = 0.5
    Eraw = fret.uncorrect_E_gamma_leak_dir(E, gamma=gamma, leakage=lk, dir_ex_t=dir_ex_t)

    na = 1
    nd = na * (1 - E) / E
    assert na / (nd + na) == E

    NA = na + lk * nd / gamma + dir_ex_t * (nd + na)
    ND = nd / gamma
    Eraw2 = NA / (NA + ND)
    assert np.allclose(Eraw2, Eraw)


def test_fret_corrections():
    Eraw = 0.7
    gamma = 1.5
    lk = 0.4
    dir_ex_t = 0.2
    E = fret.correct_E_gamma_leak_dir(Eraw, gamma=gamma, leakage=lk, dir_ex_t=dir_ex_t)

    NA = 1
    ND = NA * (1 - Eraw) / Eraw
    assert NA / (NA + ND) == Eraw

    # NOTE: dir_ex_t is defined as: Dir = dir_ex_t * (gamma * ND + na)
    na = (NA - dir_ex_t * gamma * ND - lk * ND) / (1 + dir_ex_t)
    nd = ND * gamma
    E2 = na / (na + nd)
    assert np.allclose(E, E2)
