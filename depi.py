from math import exp
import numpy as np
import pandas as pd
import depi_cy
from depi_cy import ou_single_step_cy
import ctmc
import dist_distrib as dd
import fret


def validate_params(params):
    d = dd.distribution(params)  # validates the distance-distribution parameters
    _check_args(params['τ_relax'], params['ndt'], params['α'], d.num_states)
    _get_multi_lifetime_components(params['τ_D'], params.get('D_fract'), 'D')
    _get_multi_lifetime_components(params['τ_A'], params.get('A_fract'), 'A')


def _check_args(τ_relax, ndt, α, num_states):
    if all(np.atleast_1d(np.asarray(τ_relax)) == 0) and ndt > 0:
        raise ValueError('When τ_relax = 0 also ndt needs to be 0 '
                         'in order to avoid a 0 time-step size.')
    if α <= 0:
        raise ValueError(f'α needs to be strictly positive. It is {α}.')
    if np.size(τ_relax) != num_states:
        msg = (f'Parameter τ_relax ({τ_relax}) needs to have same size as '
               f'the number of states ({num_states})')
        raise TypeError(msg)


def _get_multi_lifetime_components(τ_X, X_fract, label='D'):
    """Check and type-convert params for D or A fluorescence lifetime components.
    """
    k_X = 1 / np.atleast_1d(τ_X)
    num_X_states = k_X.size
    if num_X_states > 1 and X_fract is None:
        msg = (f'When there is more than one {label} lifetime component, '
               'you need to specify the parameter `D_fract`.')
        raise TypeError(msg)
    X_fract = np.atleast_1d(X_fract).astype('float')
    if X_fract.size != num_X_states:
        msg = (f'Arrays `{label}_fract` ({X_fract}) and τ_{label} '
               f'({τ_X}) need to have the same size.')
        raise ValueError(msg)
    if not np.allclose(X_fract.sum(), 1, rtol=0, atol=1e-10):
        msg = (f'{label}_fract must sum to 1. It sums to {X_fract.sum()} instead.')
        raise ValueError(msg)
    return k_X, X_fract


def recolor_burstsph(
        timestamps, *, R0, τ_relax, δt, τ_D, τ_A, D_fract=1., A_fract=1.,
        gamma=1.0, lk=0., dir_ex_t=0., rg=None, chunk_size=1000, α=0.05, ndt=10,
        **dd_model):
    name = dd_model['name'].lower()
    dd.assert_valid_model_name(name)
    if name.startswith('gauss'):
        # Gaussian distance distributions
        func = recolor_burstsph_OU_gauss_R
    else:
        # Any non-Gaussian distance distribution
        func = recolor_burstsph_OU_dist_distrib
    return func(
        timestamps, R0=R0, gamma=gamma, lk=lk, dir_ex_t=dir_ex_t, τ_relax=τ_relax, δt=δt,
        τ_D=τ_D, D_fract=D_fract, τ_A=τ_A, A_fract=A_fract,
        rg=rg, chunk_size=chunk_size, α=α, ndt=ndt, dd_params=dd_model)


def recolor_burstsph_OU_dist_distrib(
        timestamps, *, R0, gamma, lk, dir_ex_t, τ_relax, dd_params, δt,
        τ_D, τ_A, D_fract=1., A_fract=1.,
        rg=None, chunk_size=1000, α=0.05, ndt=10):
    """Recoloring simulation with non-Gaussian distance distribution.
    """
    print(f'gamma = {gamma}, lk = {lk}, dir_ex_t = {dir_ex_t}')
    if rg is None:
        rg = np.random.RandomState()
    d = dd.distribution(dd_params)  # validates the distance-distribution parameters
    _check_args(τ_relax, ndt, α, d.num_states)
    k_D, D_fract = _get_multi_lifetime_components(τ_D, D_fract, 'D')
    k_A, A_fract = _get_multi_lifetime_components(τ_A, A_fract, 'A')
    ts = timestamps.values
    # Extract the fixed parameters that do not depend on the number of states
    dd_params = dd_params.copy()
    du = dd_params.pop('du')
    u_max = dd_params.pop('u_max')
    dr = dd_params.pop('dr')
    if d.num_states == 1:
        S_ph = None
        p = {k: v if np.isscalar(v) else v[0]
             for k, v in {'τ_relax': τ_relax, **dd_params}.items()}
        # Compute R-axis for the 1-state distance distribution
        r_dd, idx_offset_dd = dd.get_r_dist_distrib(
            du=du, u_max=u_max, dr=dr, dd_params=dd_params)
        # Run the 1-state recoloring simulation
        R_ph, T_ph = depi_cy.sim_DA_from_timestamps2_p2_dist_cy(
            ts, δt, k_D, D_fract, R0, p['τ_relax'], r_dd, idx_offset_dd, du,
            rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
    else:
        funcs = {1: depi_cy.sim_DA_from_timestamps2_p2_2states_dist_cy,
                 2: depi_cy.sim_DA_from_timestamps2_p2_Nstates_dist_cy}
        # Compute R-axis for the multi-state distance distribution
        r_dd_list, idx_offset_list = [], []
        for i in range(d.num_states):
            dd_params_i = {k: v[i] for k, v in dd_params.items() if k != 'name'}
            dd_params_i['name'] = dd_params['name']
            r_dd, idx_offset = dd.get_r_dist_distrib(
                du=du, u_max=u_max, dr=dr, dd_params=dd_params_i)
            r_dd_list.append(r_dd)
            idx_offset_list.append(idx_offset)
        r_dd = np.vstack(r_dd_list)
        idx_offset_dd = np.array(idx_offset_list, dtype='int64')
        # Run the multi-state recoloring simulation
        R_ph, T_ph, S_ph = funcs[d.k_s.ndim](
            ts, δt, k_D, D_fract, R0, np.asfarray(τ_relax), d.k_s, r_dd, idx_offset_dd, du,
            rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
    print('- Coloring photons ... ', flush=True, end='')
    A_em, A_em_nolk, T_ph_dir_ex, lk_ph, dir_ex_ph = _color_photons(
        R_ph, R0, T_ph=T_ph, gamma=gamma, lk=lk, dir_ex_t=dir_ex_t, rg=rg)
    T_ph_complete = _add_acceptor_nanotime(A_em_nolk, T_ph_dir_ex, τ_A, A_fract, rg)
    print('DONE\n- Making dataframe ...', flush=True, end='')
    A_em, T_ph_complete,
    df = _make_burstsph_df(timestamps, T_ph_complete, A_em, R_ph, S_ph, lk_ph, dir_ex_ph)
    print('DONE', flush=True)
    return df


def recolor_burstsph_OU_gauss_R(
        timestamps, *, R0, gamma, lk, dir_ex_t, τ_relax, dd_params, δt,
        τ_D, τ_A, D_fract=1., A_fract=1.,
        rg=None, chunk_size=1000, α=0.05, ndt=10, cdf=True):
    """Recolor burst photons with Ornstein–Uhlenbeck D-A distance diffusion.

    Simulate Gaussian-distributed D-A distances diffusing according
    to an Ornstein–Uhlenbeck (OU) process with relaxation time `τ_relax`.
    Each input timestamp is the D excitation time. The D de-excitation
    happens either via D radiative emission (`τ_D`) or via FRET to A
    (distance dependent). The parameter `δt` is the time-step of the
    D de-excitation simulation which takes into account the D-A diffusion.
    When `τ_relax` is comparable or smaller than `τ_D`, the resulting FRET
    histogram is biased toward higher FRET values, a phenomenon known
    as diffusion-enhanced FRET (Beechem-Haas BJ1989).

    Arguments:
        timestamps (pandas.Series): macrotimes of photons to be recolored.
            The index needs to have two levels: ('burst', 'ph').
        R0 (float): Förster radious of the D-A pair
        R_mean (float): mean D-A distance
        R_sigma (float): standard deviation of the distance distribution
        τ_relax (float): relaxation time of the OU process
        δt (float): nanotime (TCSPC) resolution, same units as
            `timestamps`. The actual time-bin `dt` may be smaller than `δt`
            because of a dependence on `τ_relax` (see argument `ndt`).
            When `cdf = False`, `dt` may be also adaptively reduced
            (see arguments `α` and `cdf`).
        rg (None or RandomGenerator): random number generator object,
            usually `numpy.random.RandomState()`. If None, use
            `numpy.random.RandomState()` with no seed. Use this to pass
            an RNG initialized with a specific seed or to choose a
            RNG other than numpy's default Mersen Twister MT19937.
        chunk_size (int): request random numbers in chunks of this size
        α (float): if `cdf=True`, sets the threshold above which switching
            from approximated probability to a CDF one.
            If `cdf=False`, sets the adaptive time-step `dt` as a fraction `α`
            of the Donor excited-state lifetime `τ_deexcitation`.
            In the latter case, `dt` is computed at each time-step as
            `min(α * τ_deexcitation, δt)` and should be small enough
            so that `δt / τ_deexcitation` is a good approximation of the
            probability of D de-excitation in the current time-bin δt.
        ndt (float): sets the max time-step `δt`, potentially overriding
            the user supplied `δt`. If `τ_relax < ndt * δt`,
            then `δt` is reduced to `τ_relax / ndt`. The
            `δt` adjustment depends only on the input argument `τ_relax`
            and is performed only one time at the beginning of the simulation.
            To avoid any `δt` adjustment, use `ndt = 0`.
        cdf (bool): if True use a fixed `dt` and the exponential CDF to
            compute the D de-excitatation probability in the current time bin.
            If False, use the approximation `p = k_emission * dt`, with `dt`
            adaptively chosen so that `p <= α`.

    Returns:
        burstsph (pandas.DataFrame): DataFrame with 3 columns: 'timestamp'
            (same as input timestamps), 'nanotime' (simulated TCSPC nanotime)
            and 'stream' (color or the photon).
    """
    print(f'gamma = {gamma}, lk = {lk}, dir_ex_t = {dir_ex_t}')
    if rg is None:
        rg = np.random.RandomState()
    d = dd.distribution(dd_params)  # validates the distance-distribution parameters
    _check_args(τ_relax, ndt, α, d.num_states)
    k_D, D_fract = _get_multi_lifetime_components(τ_D, D_fract, 'D')
    k_A, A_fract = _get_multi_lifetime_components(τ_A, A_fract, 'A')
    ts = timestamps.values
    if d.num_states == 1:
        S_ph = None
        if cdf:
            func = depi_cy.sim_DA_from_timestamps2_p2_cy
        else:
            func = depi_cy.sim_DA_from_timestamps2_p_cy
        R_ph, T_ph = func(
            ts, δt, k_D, D_fract, R0, dd_params['R_mean'], dd_params['R_sigma'],
            τ_relax, rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
    else:
        funcs = {1: depi_cy.sim_DA_from_timestamps2_p2_2states_cy,
                 2: depi_cy.sim_DA_from_timestamps2_p2_Nstates_cy}
        params = (np.asfarray(dd_params['R_mean']),
                  np.asfarray(dd_params['R_sigma']),
                  np.asfarray(τ_relax), np.asfarray(d.k_s))
        R_ph, T_ph, S_ph = funcs[d.k_s.ndim](
            ts, δt, k_D, D_fract, R0, *params,
            rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)

    burstsph = _make_burstsph_df(timestamps, T_ph, R_ph, S_ph)
    print('- Coloring photons ... ', flush=True, end='')
    burstsph = _color_photons(burstsph, R0, gamma=gamma, lk=lk, dir_ex_t=dir_ex_t, rg=rg)
    burstsph = _add_acceptor_nanotime(burstsph, τ_A, A_fract, rg)
    print('DONE', flush=True)
    return burstsph


def _make_burstsph_df(timestamps, T_ph, R_ph, S_ph=None):
    burstsph = pd.DataFrame(timestamps)
    burstsph['nanotime'] = T_ph
    burstsph['R_ph'] = R_ph
    if S_ph is not None:
        burstsph['state'] = S_ph
    return burstsph


def _color_photons(df, R0, gamma, lk, dir_ex_t, rg):
    # Color photons (either D or A)
    num_ph = df.shape[0]
    E = fret.E_from_dist(df.R_ph, R0)
    Eraw = fret.uncorrect_E_gamma_leak_dir(E, gamma=gamma, leakage=lk, dir_ex_t=dir_ex_t)
    A_em = rg.binomial(1, p=Eraw, size=num_ph).astype(bool)
    D_em = ~A_em
    NA = A_em.sum()
    ND = A_em.size - NA
    assert D_em.sum() == ND
    df['A_ch'] = A_em
    df['stream'] = pd.Categorical.from_codes(A_em, categories=["DexDem", "DexAem"])
    # These values are shot-noise free
    ND_theor = num_ph * (1 - Eraw)
    NA_theor = num_ph * Eraw

    # Assign D leakage photons
    # NOTE: fract_lk_ph is equal to lk*ND / NA but without the shot noise
    fract_lk_ph = lk * ND_theor[A_em] / NA_theor[A_em]
    assert fract_lk_ph.size == NA
    leaked_photons = rg.binomial(1, p=fract_lk_ph, size=NA).astype(bool)
    A_em_nolk = A_em.copy()
    A_em_nolk[A_em] = ~leaked_photons
    df['leak_ph'] = False
    df.loc[A_em, 'leak_ph'] = leaked_photons
    assert df.leak_ph.sum() == leaked_photons.sum(), f'{df.leak_ph.sum()}, {leaked_photons.sum()}'

    # Assign A direct excitation photons
    Lk = leaked_photons.sum()
    A_em_nolk = A_em & (~df.leak_ph)
    nd = ND_theor[A_em_nolk]
    na_raw = NA_theor[A_em_nolk]
    na = (na_raw - dir_ex_t * gamma * nd - Lk) / (1 + dir_ex_t)
    fract_dir_ex_ph = dir_ex_t * (gamma * nd + na) / (na_raw - Lk)
    assert fract_dir_ex_ph.size == NA - leaked_photons.sum()
    dir_ex_photons = rg.binomial(1, p=fract_dir_ex_ph,
                                 size=(NA - leaked_photons.sum())).astype(bool)
    num_dir_ex_ph = dir_ex_photons.sum()
    df['dir_ex_ph'] = False
    df.loc[A_em_nolk, 'dir_ex_ph'] = dir_ex_photons
    nanot_dir_ex = df.nanotime.copy()
    nanot_dir_ex[df.dir_ex_ph] = 0
    df.nanotime = nanot_dir_ex
    assert df.dir_ex_ph.sum() == num_dir_ex_ph, f'{df.dir_ex_ph.sum()}, {num_dir_ex_ph}'
    assert A_em[df.leak_ph].all()
    assert A_em[df.dir_ex_ph].all()
    return df


def _add_acceptor_nanotime(df, τ_A, A_fract, rg):
    A_mask = df.A_ch & ~df.leak_ph
    if np.size(τ_A) == 1:
        # Add A lifetimes to A nanotimes
        df.loc[A_mask, 'nanotime'] += rg.exponential(scale=τ_A, size=A_mask.sum())
    else:
        num_A_comps = len(τ_A)
        components = np.random.choice(num_A_comps, size=A_mask.sum(), p=A_fract)
        for i, τ_A_i in enumerate(τ_A):
            # Add A lifetimes to nanotimes for component i
            comp_i = A_mask.copy()
            comp_i.loc[A_mask] = (components == i)
            df.loc[comp_i, 'nanotime'] += rg.exponential(scale=τ_A_i, size=comp_i.sum())
    return df
