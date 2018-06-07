import numpy as np
from depi import dist_distrib as dd


nm = 1.
ns = 1.

params_1state_list = [
    dict(
        name='gaussian',
        # physical parameters
        R_mean=6.37 * nm,
        R_sigma=1 * nm,
        R0=6 * nm,
        τ_relax=0.2 * ns,
        τ_D=[3.8 * ns, 1 * ns],
        D_fract=[0.5, 0.5],
        τ_A=4 * ns,
        k_s=[1, 1],
        # simulation parameters
        cdf=True,
        δt=1e-2 * ns,
        ndt=10,
        α=0.1,
        gamma=2,
    )
]


def test_1state_distributions():
    for params in params_1state_list:
        d = dd.distribution(params)
        x, y = d.get_pdf()
        area = np.trapz(y, x=x)
        mean = np.trapz(x * y, x)
        std_dev = np.sqrt(np.trapz(((x - mean)**2) * y, x))
        assert np.allclose(area, 1, rtol=0, atol=1e-4)
        assert np.allclose(mean, params['R_mean'])
        assert np.allclose(std_dev, params['R_sigma'])
