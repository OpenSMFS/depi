from math import exp
import numpy as np
import pandas as pd
import depi_cy
from depi_cy import ou_single_step_cy
import ctmc
import dist_distrib as dd


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
    E = 1 / (1 + (x / R0)**6)
    if not np.isscalar(x):
        E[x < 0] = 1
    elif x < 0:
        E = 0
    return E


def dist_from_E(E, R0):
    """Return the D-A distance for a give E and R0
    """
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
    gamma_lk = gamma / (1 + lk)
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
        A_em, R_ph, T_ph = depi_cy.sim_DA_from_timestamps2_p2_dist_cy(
            ts, δt, k_D, D_fract, R0, p['τ_relax'], r_dd, idx_offset_dd, du,
            gamma=gamma_lk, rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
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
        A_em, R_ph, T_ph, S_ph = funcs[d.k_s.ndim](
            ts, δt, k_D, D_fract, R0, np.asfarray(τ_relax), d.k_s, r_dd, idx_offset_dd, du,
            gamma=gamma_lk, rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
    A_em_lk, T_ph_dir_ex = _compute_D_leakage_A_dir_ex(A_em, T_ph, gamma, lk, dir_ex_t, rg)
    T_ph_complete = _calc_T_ph_with_acceptor(A_em, T_ph_dir_ex, τ_A, A_fract, rg)
    return _make_burstsph_df(timestamps, T_ph_complete, A_em_lk, R_ph, S_ph)


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
        A_em, R_ph, T_ph = func(
            ts, δt, k_D, D_fract, R0, dd_params['R_mean'], dd_params['R_sigma'],
            τ_relax, rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
    else:
        funcs = {1: depi_cy.sim_DA_from_timestamps2_p2_2states_cy,
                 2: depi_cy.sim_DA_from_timestamps2_p2_Nstates_cy}
        params = (np.asfarray(dd_params['R_mean']),
                  np.asfarray(dd_params['R_sigma']),
                  np.asfarray(τ_relax), np.asfarray(d.k_s))
        A_em, R_ph, T_ph, S_ph = funcs[d.k_s.ndim](
            ts, δt, k_D, D_fract, R0, *params,
            rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
    A_em, A_em_nolk, T_ph_dir_ex, lk_ph, dir_ex_ph = _compute_D_leakage_A_dir_ex(
        R_ph, R0, T_ph=T_ph, gamma=gamma, lk=lk, dir_ex_t=dir_ex_t, rg=rg)
    print('- Coloring photons ... ', flush=True, end='')
    T_ph_complete = _calc_T_ph_with_acceptor(A_em_nolk, T_ph_dir_ex, τ_A, A_fract, rg)
    print('DONE\n- Making dataframe ...', flush=True, end='')
    df = _make_burstsph_df(timestamps, T_ph_complete, A_em, R_ph, S_ph)
    df['leak_ph'] = lk_ph
    df['dir_ex_ph'] = dir_ex_ph
    print('DONE', flush=True)
    return df


def _compute_D_leakage_A_dir_ex(R_ph, R0, T_ph, gamma, lk, dir_ex_t, rg):
    # Color photons (either D or A)
    num_ph = len(R_ph)
    E = E_from_dist(R_ph, R0)
    Eraw = uncorrect_E_gamma_leak_dir(E, gamma=gamma, leakage=lk, dir_ex_t=dir_ex_t)
    A_em = rg.binomial(1, p=Eraw, size=num_ph).astype(bool)
    D_em = ~A_em
    NA = A_em.sum()
    ND = A_em.size - NA
    assert D_em.sum() == ND
    # These values are shot-noise free
    ND_theor = num_ph * (1 - Eraw)
    NA_theor = num_ph * Eraw

    # Assign D leakage photons
    # NOTE: fract_lk_ph is equal to lk*ND / NA but without the shot noise
    fract_lk_ph = lk * ND_theor[A_em] / NA_theor[A_em]
    assert fract_lk_ph.size == NA
    leaked_photons = rg.binomial(1, p=fract_lk_ph, size=NA).astype(bool)
    print(f'Fraction leaked: {fract_lk_ph}')
    print('Leaked ph unique values: ', np.unique(leaked_photons))
    A_em_nolk = A_em.copy()
    A_em_nolk[A_em] = ~leaked_photons
    lk_ph = np.zeros_like(A_em)
    lk_ph[A_em] = leaked_photons
    assert lk_ph.sum() == leaked_photons.sum(), f'{lk_ph.sum()}, {leaked_photons.sum()}'

    # Assign A direct excitation photons
    Lk = leaked_photons.sum()
    nd = ND_theor[A_em * (~lk_ph)]
    na_raw = NA_theor[A_em * (~lk_ph)]
    na = (na_raw - dir_ex_t * gamma * nd - Lk) / (1 + dir_ex_t)
    fract_dir_ex_ph = dir_ex_t * (gamma * nd + na) / (na_raw - Lk)
    assert fract_dir_ex_ph.size == NA - leaked_photons.sum()
    dir_ex_photons = rg.binomial(1, p=fract_dir_ex_ph,
                                 size=(NA - leaked_photons.sum())).astype(bool)
    print('Dir ex unique values:', np.unique(dir_ex_photons))
    dir_ex_ph = np.zeros_like(A_em)
    dir_ex_ph[A_em * (~lk_ph)] = dir_ex_photons
    T_ph_dir_ex = T_ph.copy()
    T_ph_dir_ex[dir_ex_ph] = 0
    print('Initial zero nanotimes: ', (T_ph == 0).sum())
    assert dir_ex_ph.sum() == dir_ex_photons.sum(), f'{dir_ex_ph.sum()}, {dir_ex_photons.sum()}'
    assert (T_ph_dir_ex[dir_ex_ph] == 0).all(), f'{(T_ph_dir_ex[dir_ex_ph] == 0).sum()}'
    assert A_em[lk_ph].all()
    assert A_em[dir_ex_ph].all()
    return A_em, A_em_nolk, T_ph_dir_ex, lk_ph, dir_ex_ph


def _calc_T_ph_with_acceptor(A_em, T_ph, τ_A, A_fract, rg):
    A_mask = A_em.view(bool)
    T_ph = np.asfarray(T_ph)
    if np.size(τ_A) == 1:
        # Add exponentially distributed lifetimes to A nanotimes
        T_ph[A_mask] += rg.exponential(scale=τ_A, size=A_mask.sum())
    else:
        num_A_comps = len(τ_A)
        A_index = np.nonzero(A_mask)[0]
        components = np.random.choice(num_A_comps, size=A_index.size, p=A_fract)
        for i, τ_A_i in enumerate(τ_A):
            comp_i = A_index[components == i]
            T_ph[comp_i] += rg.exponential(scale=τ_A_i, size=comp_i.size)
    return T_ph


def _make_burstsph_df(timestamps, T_ph, A_em, R_ph, S_ph):
    burstsph_sim = pd.DataFrame(timestamps)
    burstsph_sim['nanotime'] = T_ph
    burstsph_sim['stream'] = (
        pd.Categorical.from_codes(A_em, categories=["DexDem", "DexAem"]))
    burstsph_sim['R_ph'] = R_ph
    if S_ph is not None:
        burstsph_sim['state'] = S_ph
    return burstsph_sim


#
# - FOLLOWING FUNCTIONS ARE SLOW PYTHON VERSIONS FOR TESTING ONLY - - - -
#

def sim_DA_from_timestamps2_p2(timestamps, dt, k_D, R0, R_mean, R_sigma,
                               tau_relax, rg, chunk_size=1000,
                               alpha=0.05, ndt=10):
    """
    Recoloring using fixed dt, emission uses exponential CDF.
    For efficiency, random numbers are computed in blocks.

    The assumption here is that since tau_relax >> dt, distance and FRET
    will not change during dt, so the transition probability can be computed
    from the exponential CDF.
    """
    if tau_relax < ndt * dt:
        dt = tau_relax / ndt
        print(f'WARNING: Reducing dt to {dt:g} '
              f'[tau_relax = {tau_relax}]')
    R = rg.randn() * R_sigma + R_mean
    t0 = 0
    nanotime = 0
    # Array flagging photons as A (1) or D (0) emitted
    A_ph = np.zeros(timestamps.size, dtype=bool)
    # Instantaneous D-A distance at D de-excitation time
    R_ph = np.zeros(timestamps.size, dtype=np.float64)
    # Time of D de-excitation relative to the last timestamp
    T_ph = np.zeros(timestamps.size, dtype=np.float64)
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    for iph, t in enumerate(timestamps):
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t = t - t0
        if delta_t < 0:
            # avoid negative delta_t possible when when two photons have
            # the same macrotime
            delta_t = 0
            t = t0
        # Compute the D-A distance at the "excitation time"
        iN += 1
        if iN == chunk_size:
            Na = memoryview(rg.randn(chunk_size))
            Pa = memoryview(rg.rand(chunk_size))
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R = ou_single_step_cy(R, delta_t, N, R_mean, R_sigma, tau_relax)
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            d_prob_ph_em = dt * k_emission
            if d_prob_ph_em > alpha:
                d_prob_ph_em = 1 - exp(-d_prob_ph_em)
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += dt
            iN += 1
            if iN == chunk_size:
                Na = memoryview(rg.randn(chunk_size))
                Pa = memoryview(rg.rand(chunk_size))
                iN = 0
            N = Na[iN]
            p = Pa[iN]
            # Update R following the OU process
            R = ou_single_step_cy(R, dt, N, R_mean, R_sigma, tau_relax)
        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / k_emission
        if prob_A_em >= p_DA:
            A_ph[iph] = True
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t + nanotime
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
    return A_ph, R_ph, T_ph


def sim_DA_from_timestamps2_p2_2states(timestamps, dt_ref, k_D, R0, R_mean,
                                       R_sigma, tau_relax, k_s, rg,
                                       chunk_size=1000, alpha=0.05, ndt=10):
    """
    2-states recoloring using CDF in dt and with random number caching
    """
    dt = np.array([dt_ref] * 2, dtype=np.float64)
    for state in [0, 1]:
        if tau_relax[state] < ndt * dt[state]:
            dt[state] = tau_relax[state] / ndt
            print(f'WARNING: Reducing dt[{state}] to {dt[state]:g} '
                  f'[tau_relax[{state}] = {tau_relax[state]}]')
    # Array flagging photons as A (1) or D (0) emitted
    A_ph = np.zeros(timestamps.size, dtype=np.uint8)
    # Instantaneous D-A distance at D de-excitation time
    R_ph = np.zeros(timestamps.size, dtype=np.float64)
    # Time of D de-excitation relative to the last timestamp
    T_ph = np.zeros(timestamps.size, dtype=np.float64)
    # State for each photon
    S_ph = np.zeros(timestamps.size, dtype=np.uint8)
    peq = [k_s[1] / (k_s[0] + k_s[1]),
           k_s[0] / (k_s[0] + k_s[1])]
    k_s_sum = np.sum(k_s)
    t0 = 0
    nanotime = 0
    state = 0  # the two states are 0 and 1
    R = rg.randn() * R_sigma[state] + R_mean[state]
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    for iph, t in enumerate(timestamps):
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t0 = t - t0
        delta_t = delta_t0 - nanotime
        if delta_t < 0:
            # avoid negative delta_t possible when when two photons have
            # the same macrotime
            delta_t = 0
            t = t0
        p_state = (1 - peq[state]) * np.exp(-(delta_t0 * k_s_sum)) + peq[state]
        u = rg.rand()
        #print(f'iph={iph}, state={state}, p_state={p_state}, u={u}, delta_t0={delta_t0}')
        # Inversion of u is for compatibility with N-state version
        if state == 1:
            u = 1 - u
        if p_state <= u:
            #print('   * state change')
            state = 0 if state == 1 else 1
            R = rg.randn() * R_sigma[state] + R_mean[state]
        # Compute the D-A distance at the "excitation time"
        iN += 1
        if iN == chunk_size:
            Na = memoryview(rg.randn(chunk_size))
            Pa = memoryview(rg.rand(chunk_size))
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R = ou_single_step_cy(R, delta_t, N, R_mean[state], R_sigma[state],
                              tau_relax[state])
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            d_prob_ph_em = k_emission * dt[state]  # prob. of emission in dt
            if d_prob_ph_em > alpha:
                d_prob_ph_em = 1 - exp(-d_prob_ph_em)
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += dt[state]
            iN += 1
            if iN == chunk_size:
                Na = memoryview(rg.randn(chunk_size))
                Pa = memoryview(rg.rand(chunk_size))
                iN = 0
            N = Na[iN]
            p = Pa[iN]
            R = ou_single_step_cy(R, dt[state], N, R_mean[state], R_sigma[state],
                                  tau_relax[state])
        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / k_emission
        if prob_A_em >= p_DA:
            A_ph[iph] = 1
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
        # Save state for current photon
        S_ph[iph] = state
    return A_ph, R_ph, T_ph, S_ph


def sim_DA_from_timestamps2_p2_Nstates(timestamps, dt_ref, k_D, R0, R_mean,
                                       R_sigma, tau_relax, K_s, rg,
                                       chunk_size=1000, alpha=0.05, ndt=10):
    """
    2-states recoloring using CDF in dt and with random number caching
    """
    assert K_s.shape[0] == K_s.shape[1], 'K_s needs to be a square matrix.'
    num_states = K_s.shape[0]
    dt = np.array([dt_ref] * num_states, dtype=np.float64)
    for state in range(num_states):
        if tau_relax[state] < ndt * dt[state]:
            dt[state] = tau_relax[state] / ndt
            print(f'WARNING: Reducing dt[{state}] to {dt[state]:g} '
                  f'[tau_relax[{state}] = {tau_relax[state]}]')
    # Array flagging photons as A (1) or D (0) emitted
    A_ph = np.zeros(timestamps.size, dtype=np.uint8)
    # Instantaneous D-A distance at D de-excitation time
    R_ph = np.zeros(timestamps.size, dtype=np.float64)
    # Time of D de-excitation relative to the last timestamp
    T_ph = np.zeros(timestamps.size, dtype=np.float64)
    # State for each photon
    S_ph = np.zeros(timestamps.size, dtype=np.uint8)
    eigenval, V, V_inv = ctmc.decompose(K_s)
    t0 = 0
    nanotime = 0
    state = 0
    state_vector = np.zeros(num_states)
    state_vector[state] = 1
    R = rg.randn() * R_sigma[state] + R_mean[state]
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    for iph, t in enumerate(timestamps):
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t0 = t - t0
        delta_t = delta_t0 - nanotime
        if delta_t < 0:
            # avoid negative delta_t possible when when two photons have
            # the same macrotime
            delta_t = 0
            t = t0
        p_states = ctmc.occupancy(
            delta_t0, state_vector, K_s, eigenval, V, V_inv)
        u = rg.rand()
        #print(f'iph={iph}, state={state}, p_states={p_states}, u={u}, delta_t0={delta_t0}')
        for s, p_state in enumerate(p_states):
            if u <= p_state:
                newstate = s
                break
            u -= p_state
        if newstate != state:
            #print('   * state change')
            state_vector[state] = 0
            state = newstate
            state_vector[state] = 1
            R = rg.randn() * R_sigma[state] + R_mean[state]
        # Compute the D-A distance at the "excitation time"
        iN += 1
        if iN == chunk_size:
            Na = memoryview(rg.randn(chunk_size))
            Pa = memoryview(rg.rand(chunk_size))
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R = ou_single_step_cy(R, delta_t, N, R_mean[state], R_sigma[state],
                              tau_relax[state])
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            d_prob_ph_em = k_emission * dt[state]  # prob. of emission in dt
            if d_prob_ph_em > alpha:
                d_prob_ph_em = 1 - exp(-d_prob_ph_em)
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += dt[state]
            iN += 1
            if iN == chunk_size:
                Na = memoryview(rg.randn(chunk_size))
                Pa = memoryview(rg.rand(chunk_size))
                iN = 0
            N = Na[iN]
            p = Pa[iN]
            R = ou_single_step_cy(R, dt[state], N, R_mean[state], R_sigma[state],
                                  tau_relax[state])
        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / k_emission
        if prob_A_em >= p_DA:
            A_ph[iph] = 1
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
        # Save state for current photon
        S_ph[iph] = state
    return A_ph, R_ph, T_ph, S_ph
