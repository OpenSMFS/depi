from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import json
from joblib import Memory

from . import dist_distrib as dd
from . import fret
from . import depi_cy


mem = Memory(cachedir='joblib_cache')


@mem.cache
def recolor_burstsph_cache(timestamp, seed=1, **params):
    """Cached version of :func:`recolor_burstsph`."""
    from randomgen import RandomGenerator, Xoroshiro128
    rg = RandomGenerator(Xoroshiro128(seed))
    burstsph = recolor_burstsph(timestamp, rg=rg, **params)
    return burstsph


def save_params(fname, params, bounds=None):
    """Save the simulation parameters `params` to disk."""
    if not fname.lower().endswith('.json'):
        fname = fname + '.json'
    if Path(fname).exists():
        raise IOError(f'File {fname} aready exists. '
                      f'Please move it before saving with the same name.')
    if bounds is not None:
        params = {**params, bounds: bounds}  # doesn't modify the original params
    with open(fname, 'wt') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)


def load_params(fname):
    """Load the simulation parameters from file `fname`."""
    if not fname.lower().endswith('.json'):
        fname = fname + '.json'
    with open(fname) as f:
        params = json.load(f)
    bounds = params.pop('bounds', None)
    return params, bounds


def validate_params(params):
    """Validate consistency of simulation parameters."""
    d = dd.distribution(params)  # validates the distance-distribution parameters
    _check_args(params['τ_relax'], params['ndt'], params['α'], d.num_states)
    _get_multi_lifetime_components(params['τ_D'], params.get('D_fract'), 'D')
    _get_multi_lifetime_components(params['τ_A'], params.get('A_fract'), 'A')


def _check_args(τ_relax, ndt, α, num_states):
    """Check input arguments for recoloring simulation."""
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
        bg_rate_d=0, bg_rate_a=0, ts_unit=1e-9, tcspc_range=50,
        prob_A_dark=0, τ_A_dark=0, gamma=1.0, lk=0., dir_ex_t=0.,
        rg=None, chunk_size=1000, α=0.05, ndt=10,
        **dd_model):
    """Recolor burst photons with Ornstein–Uhlenbeck D-A distance diffusion.

    Simulate D-A distances diffusing with a Gaussian or non-Gaussian distance
    distribution using an Ornstein–Uhlenbeck (OU) process with relaxation
    time `τ_relax`.
    Each input timestamp is the D excitation time. The D de-excitation
    happens either via D radiative emission (`τ_D`) or via FRET to A
    (distance dependent). The parameter `δt` is the time-step of the
    D de-excitation simulation which takes into account the D-A diffusion.
    When `τ_relax` is comparable or smaller than `τ_D`, the resulting FRET
    histogram is biased toward higher FRET values, a phenomenon known
    as diffusion-enhanced FRET (Beechem-Haas BJ1989).

    The input dictionary `dd_model` sets the distance distribution model.
    This dictionary have a key "name" which should be one of the strings:
    'gaussian', 'wlc', 'gaussian_chain', 'radial_gaussian'.
    Multi-state models can be specified when the distance distribution
    parameters are arrays instead of scalars. In this case also `τ_relax`
    needs to be an array (of same length).

    The D and A fluorescence decays can be single of multi-exponential.
    When `τ_D` or `τ_A` are scalars the decays are mono-exponential.
    When `τ_D` or `τ_A` are arrays the argument `D_fract` and `A_fract`
    specify the fractions of the different exponential components.

    Arguments:
        timestamps (pandas.Series): macrotimes of photons to be recolored.
            The index needs to have two levels: ('burst', 'ph').
        R0 (float): Förster radious of the D-A pair
        τ_relax (float): relaxation time of the OU process
        δt (float): nanotime (TCSPC) resolution, same units as
            `timestamps`. The actual time-bin `dt` may be smaller than `δt`
            because of a dependence on `τ_relax` (see argument `ndt`).
        α (float): sets the threshold above which switching
            from approximated probability to a CDF one.
        ndt (float): sets the max allowed time-step `δt`, potentially overriding
            the user supplied `δt`. If `τ_relax < ndt * δt`,
            then `δt` is reduced to `τ_relax / ndt`. The
            `δt` adjustment depends only on the input argument `τ_relax`
            and is performed only once at the beginning of the simulation.
            To avoid any `δt` adjustment, use `ndt = 0`.
        rg (None or RandomGenerator): random number generator object,
            usually `numpy.random.RandomState()`. If None, use
            `numpy.random.RandomState()` with no seed. Use this to pass
            an RNG initialized with a specific seed or to choose a
            RNG other than numpy's default Mersen Twister MT19937.
        chunk_size (int): request random numbers in chunks of this size

    Returns:
        burstsph (pandas.DataFrame): recolored photon data. Columns include: 'timestamp'
            (same as input timestamps); 'nanotime' (simulated TCSPC nanotimes);
            'stream' (color or the photon); 'R_ph' the D-A distance at
            de-excitation time; 'state' the state for each photon.
    """
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
        prob_A_dark=prob_A_dark, τ_A_dark=τ_A_dark,
        bg_rate_d=bg_rate_d, bg_rate_a=bg_rate_a, ts_unit=ts_unit, tcspc_range=tcspc_range,
        rg=rg, chunk_size=chunk_size, α=α, ndt=ndt, dd_params=dd_model)


def recolor_burstsph_OU_dist_distrib(
        timestamps, *, R0, gamma, lk, dir_ex_t, τ_relax, dd_params, δt,
        τ_D, τ_A, D_fract=1., A_fract=1., bg_rate_d, bg_rate_a, ts_unit, tcspc_range,
        prob_A_dark=0, τ_A_dark=0,
        rg=None, chunk_size=1000, α=0.05, ndt=10):
    """Recoloring simulation with non-Gaussian distance distribution.
    """
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
    burstsph_all = _make_burstsph_df(timestamps, T_ph, R_ph, S_ph)
    burstsph, bg = _select_background(burstsph_all, bg_rate_d=bg_rate_d, bg_rate_a=bg_rate_a,
                                      ts_unit=ts_unit, tcspc_range=tcspc_range, rg=rg)
    burstsph = _color_photons(burstsph, R0, gamma=gamma, lk=lk, dir_ex_t=dir_ex_t, rg=rg)
    burstsph = _recolor_A_blinking(burstsph, prob_A_dark, τ_A_dark, lk, τ_D, D_fract, rg)
    burstsph = _add_acceptor_nanotime(burstsph, τ_A, A_fract, rg)
    burstsph = _merge_ph_and_bg(burstsph_all, burstsph, bg)
    return burstsph


def recolor_burstsph_OU_gauss_R(
        timestamps, *, R0, gamma, lk, dir_ex_t, τ_relax, dd_params, δt,
        τ_D, τ_A, D_fract=1., A_fract=1., bg_rate_d, bg_rate_a, ts_unit, tcspc_range,
        prob_A_dark=0, τ_A_dark=0,
        rg=None, chunk_size=1000, α=0.05, ndt=10, cdf=True):
    """Recoloring simulation with Gaussian distance distribution.
    """
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
    burstsph_all = _make_burstsph_df(timestamps, T_ph, R_ph, S_ph)
    burstsph, bg = _select_background(burstsph_all, bg_rate_d=bg_rate_d, bg_rate_a=bg_rate_a,
                                      ts_unit=ts_unit, tcspc_range=tcspc_range, rg=rg)
    burstsph = _color_photons(burstsph, R0, gamma=gamma, lk=lk, dir_ex_t=dir_ex_t, rg=rg)
    burstsph = _recolor_A_blinking(burstsph, prob_A_dark, τ_A_dark, lk, τ_D, D_fract, rg)
    burstsph = _add_acceptor_nanotime(burstsph, τ_A, A_fract, rg)
    burstsph = _merge_ph_and_bg(burstsph_all, burstsph, bg)
    return burstsph


def _recolor_A_blinking(burstsph, prob_A_dark, τ_A_dark, lk, τ_D, D_fract, rg):
    if prob_A_dark == 0:
        burstsph['A_dark_ph'] = False
        return burstsph
    # Acceptor-dye emitted photons (no D leakage), before A photo-blinking
    Aem = (burstsph.stream == 'DexAem') & ~burstsph.leak_ph
    num_Aem = Aem.sum()
    # Simulate acceptor bright to dark transitions
    u = rg.rand(num_Aem)
    A_dark = u <= prob_A_dark
    # Recolor and label photons emitted during A dark state
    # Generate a tau for each `Aem` photon, then discard it unless the photon is "dark"
    A_dark_lifetimes = rg.exponential(scale=τ_A_dark, size=num_Aem)
    ts = burstsph.loc[Aem, 'timestamp'].values
    A_dark = _generate_acceptor_dark_state(A_dark, ts, A_dark_lifetimes)
    num_A_dark = A_dark.sum()
    assert A_dark.size == num_Aem
    burstsph['A_dark_ph'] = False
    burstsph.loc[Aem, 'A_dark_ph'] = A_dark
    burstsph.loc[burstsph.A_dark_ph, 'A_ch'] = False
    assert num_A_dark == burstsph.A_dark_ph.sum()
    assert burstsph.loc[burstsph.A_dark_ph, 'A_dark_ph'].all()
    assert not burstsph.loc[burstsph.A_dark_ph, 'A_ch'].any()
    # Set the D fluorescence lifetime for blinked photons
    burstsph.loc[burstsph.A_dark_ph, 'nanotime'] = _calc_intrisic_nanotime(
        num_A_dark, τ_D, D_fract, rg)
    # Simulate residual D leakage
    if lk > 0:
        # NOTE: leaked photons go into the A channel but retain D nanotime
        new_lk_ph = rg.binomial(1, p=lk, size=num_A_dark)
        burstsph.loc[burstsph.A_dark_ph, 'leak_ph'] |= new_lk_ph
        burstsph.loc[burstsph.leak_ph, 'A_ch'] = True
    # Recompute the categorical "stream" column from A_ch
    burstsph = _set_ph_stream_column(burstsph, burstsph.A_ch)
    return burstsph


def _generate_acceptor_dark_state(A_dark, ts, A_dark_times):
    A_dark = memoryview(A_dark)
    ts = memoryview(ts)
    A_dark_times = memoryview(A_dark_times)
    dark = False
    for i in range(len(A_dark)):
        if A_dark[i]:
            # switch A to the dark state
            dark = True
            dark_start = ts[i]
            dark_time = A_dark_times[i]
        if dark:
            A_dark[i] = True  # label photon as A dark-state
            dark_time_elapsed = ts[i] - dark_start
            if dark_time_elapsed >= dark_time:
                # go back to the bright state
                dark = False
    return np.asarray(A_dark)


def _make_burstsph_df(timestamps, T_ph, R_ph, S_ph=None):
    burstsph = pd.DataFrame(timestamps)
    burstsph['nanotime'] = T_ph
    burstsph['R_ph'] = R_ph
    if S_ph is not None:
        burstsph['state'] = S_ph
    return burstsph


def _select_background(df, bg_rate_d, bg_rate_a, ts_unit, tcspc_range, rg):
    if bg_rate_d == 0 and bg_rate_a == 0:
        return df, pd.DataFrame(columns=['bg_a', 'bg_d'], dtype=bool)
    func = partial(_background_per_burst, bg_rate_d=bg_rate_d, bg_rate_a=bg_rate_a,
                   ts_unit=ts_unit, rg=rg)
    bg = df.timestamp.groupby('burst').apply(func)
    bg_mask = bg.bg_a | bg.bg_d
    bg['nanotime'] = np.nan
    bg.loc[bg_mask, 'nanotime'] = rg.rand(bg_mask.sum()) * tcspc_range
    return df.loc[~bg_mask].copy(), bg


def _background_per_burst(burst, bg_rate_d, bg_rate_a, ts_unit, rg):
    burst = burst.reset_index('burst', drop=True)
    bsize = burst.shape[0]
    bwidth = (burst.iloc[-1] - burst.iloc[0]) * ts_unit
    bg_burst = (bg_rate_d + bg_rate_a) * bwidth
    assert bg_burst < bsize
    prob_A_ph_bg = bg_rate_a / (bg_rate_a + bg_rate_d)
    prob_bg_ph = bg_burst / bsize
    bg_ph = rg.binomial(1, p=prob_bg_ph, size=bsize)
    bg_ph_index = np.where(bg_ph)[0]
    a_bg_ph = rg.binomial(1, p=prob_A_ph_bg, size=bg_ph_index.size).astype(bool)
    a_bg_ph_index = bg_ph_index[a_bg_ph]
    d_bg_ph_index = list(set(bg_ph_index) - set(a_bg_ph_index))
    bg = pd.DataFrame(index=burst.index, columns=['bg_a', 'bg_d'], dtype=bool)
    bg.loc[:] = False
    bg.iloc[a_bg_ph_index, 0] = True  # assign column 'bg_a'
    bg.iloc[d_bg_ph_index, 1] = True  # assign column 'bg_d'
    return bg


def _merge_ph_and_bg(burstsph_all, burstsph, bg):
    if burstsph_all.shape[0] == burstsph.shape[0]:
        burstsph['bg_ph'] = False
        return burstsph
    # This way of merging bg into a bigger DataFrame is ugly and brittle
    # there must be a better way!
    bg_mask = bg.bg_a | bg.bg_d
    bool_cols = ['A_ch', 'leak_ph', 'dir_ex_ph', 'A_dark_ph']
    for col in bool_cols:
        burstsph_all[col] = False  # create the column
    burstsph_all['bg_ph'] = False  # create the bg_ph column
    burstsph_all.loc[bg_mask, 'bg_ph'] = True
    for col in ['nanotime'] + bool_cols:
        burstsph_all.loc[~bg_mask, col] = burstsph[col]
    burstsph_all.loc[bg_mask, 'nanotime'] = bg.loc[bg_mask, 'nanotime']
    burstsph_all.loc[bg_mask, 'A_ch'] = bg.loc[bg_mask, 'bg_a']
    burstsph_all = _set_ph_stream_column(burstsph_all, burstsph_all.A_ch)
    return burstsph_all


def _set_ph_stream_column(df, A_ch):
    df['stream'] = pd.Categorical.from_codes(A_ch, categories=["DexDem", "DexAem"])
    return df


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
    df = _set_ph_stream_column(df, A_em)
    # These values are shot-noise free
    ND_theor = num_ph * (1 - Eraw)
    NA_theor = num_ph * Eraw

    # Assign D leakage photons
    # NOTE: fract_lk_ph is equal to lk*ND / NA but without the shot noise
    fract_lk_ph = lk * ND_theor[A_em] / NA_theor[A_em]
    assert fract_lk_ph.size == NA
    leaked_photons = rg.binomial(1, p=fract_lk_ph, size=NA).astype(bool)
    df['leak_ph'] = False
    df.loc[A_em, 'leak_ph'] = leaked_photons
    A_em_nolk = A_em & (~df.leak_ph)
    assert df.leak_ph.sum() == leaked_photons.sum(), f'{df.leak_ph.sum()}, {leaked_photons.sum()}'

    # Assign A direct excitation photons
    Lk = leaked_photons.sum()
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
    assert df.dir_ex_ph.sum() == num_dir_ex_ph, f'{df.dir_ex_ph.sum()}, {num_dir_ex_ph}'
    assert A_em[df.leak_ph].all()
    assert A_em[df.dir_ex_ph].all()
    return df


def _add_acceptor_nanotime(df, τ_A, A_fract, rg):
    """Modify the D nanotime by adding the A components.
    Leaked photons are not touched. This function sets the nanotime
    of A-direct-excitation photons to 0 before adding the
    A fluorescence components.
    """
    # Save original D-deexcitation nanotimes
    df['nanotime_d'] = df['nanotime'].copy()
    # Set the initial nanotime of A direct excitation photons to 0
    nanot_dir_ex = df.nanotime.copy()
    nanot_dir_ex[df.dir_ex_ph] = 0
    df.nanotime = nanot_dir_ex
    # Add A lifetimes to non-leaked photons
    A_mask = df.A_ch & ~df.leak_ph
    nanotime = _calc_intrisic_nanotime(A_mask.sum(), τ_A, A_fract, rg)
    df.loc[A_mask, 'nanotime'] += nanotime
    return df


def _calc_intrisic_nanotime(num_ph, τ_X, X_fract, rg):
    """Generate the nanotime for single or multi-componet decays.
    """
    if np.size(τ_X) == 1:
        nanotime = rg.exponential(scale=τ_X, size=num_ph)
    else:
        num_X_comps = len(τ_X)
        components = rg.choice(num_X_comps, size=num_ph, p=X_fract)
        nanotime = np.zeros(num_ph)
        for i, τ_X_i in enumerate(τ_X):
            # Compute nanotimes for component `i` of the fluorescence decay
            component_i = components == i
            nanotime[component_i] = rg.exponential(scale=τ_X_i, size=component_i.sum())
    return nanotime
