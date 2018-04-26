from math import exp
import numpy as np
import pandas as pd
import numba
import depi_cy
from depi_cy import ou_single_step_cy
import ctmc


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
    A_em = burstsph.stream.cat.codes  # get stream as 1 (DexDem) and 0 (DexAem)
    E = [A_em[istart:istop].sum() / bsize
         for istart, istop, bsize in
         zip(bursts.istart, bursts.istop, bursts.size_raw)]
    return E


def recolor_burstsph_OU_gauss_R(timestamps, R0, R_mean, R_sigma,
                                τ_relax, τ_D, τ_A, δt, k_s=None, rg=None,
                                chunk_size=1000, α=0.05, ndt=10, cdf=True):
    """Recolor burst photons with Ornstein–Uhlenbeck D-A distance diffusion.

    Simulate Gaussian-distributed D-A distances diffusing according
    to an Ornstein–Uhlenbeck (OU) process with relaxation time `τ_relax`.
    Each input timestamp is the D excitation time. The D de-excitation
    happens via D radiative emission (`τ_D`) or energy transfer to A
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
            adaptively choosen so that `p <= α`.

    Returns:
        burstsph (pandas.DataFrame): DataFrame with 3 columns: 'timestamp'
            (same as input timestamps), 'nanotime' (simulated TCSPC nanotime)
            and 'stream' (color or the photon).
    """
    if rg is None:
        rg = np.random.RandomState()
    if all(np.atleast_1d(np.asarray(τ_relax)) == 0) and ndt > 0:
        raise ValueError('When τ_relax = 0 also ndt needs to be 0 '
                         'in order to avoid a 0 time-step size.')
    if α <= 0:
        raise ValueError(f'α needs to be strictly positive. It is {α}.')
    k_D = 1 / τ_D
    ts = timestamps.values
    # Use the size of R_mean to infer the number of states
    if np.size(R_mean) == 1:
        if cdf:
            func = depi_cy.sim_DA_from_timestamps2_p2_cy
        else:
            func = depi_cy.sim_DA_from_timestamps2_p_cy
        params = R_mean[0], R_sigma[0], τ_relax[0]
        A_em, R_ph, T_ph = func(ts, δt, k_D, R0, *params,
                                rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
    else:
        # Check that all parameters have length 2
        num_states = len(R_mean)
        p = dict(R_mean=R_mean, R_sigma=R_sigma, τ_relax=τ_relax)
        for name, val in p.items():
            m = f'Argument "{name}" (={val}) should be of length {num_states}.'
            assert np.size(val) == num_states, m
        k_s = np.asarray(k_s)
        if num_states == 1:
            m = f'"k_s" (={k_s}) should be a 1d array or length 1 (# states = 1).'
            assert np.size(k_s) == 1, m
        elif k_s.ndim == 1:
            m = f'"k_s" (={k_s}) should an {num_states}x{num_states} array.'
            assert num_states == 2, m
            m = f'"k_s" (={k_s}) should be a 2-element 1d array or an 2x2 array.'
            assert np.size(k_s) == 2, m
            func = depi_cy.sim_DA_from_timestamps2_p2_2states_cy
        else:
            m = f'"k_s" (={k_s}) should be an {num_states}x{num_states} array.'
            assert k_s.ndim == 2 and k_s.shape[0] == k_s.shape[1], m
            func = depi_cy.sim_DA_from_timestamps2_p2_Nstates_cy
        params = (np.asarray(R_mean), np.asarray(R_sigma),
                  np.asarray(τ_relax), np.asarray(k_s))
        A_em, R_ph, T_ph, S_ph = func(
            ts, δt, k_D, R0, *params,
            rg=rg, chunk_size=chunk_size, alpha=α, ndt=ndt)
    A_mask = A_em.view(bool)
    T_ph = np.asarray(T_ph)
    # Add exponentially distributed lifetimes to A nanotimes
    T_ph[A_mask] += rg.exponential(scale=τ_A, size=A_mask.sum())
    burstsph_sim = pd.DataFrame(timestamps)
    burstsph_sim['nanotime'] = T_ph
    burstsph_sim['stream'] = (
        pd.Categorical.from_codes(A_em, categories=["DexDem", "DexAem"]))
    burstsph_sim['R_ph'] = R_ph
    if np.size(R_mean) == 2:
        burstsph_sim['state'] = S_ph
    return burstsph_sim


def recolor_burstsph_OU_gauss_R_approx(timestamps, R0, R_mean, R_sigma, τ_relax,
                                       τ_D, τ_A, δt, rg=None):
    """Recolor burst photons with Ornstein–Uhlenbeck D-A distance diffusion.

    This version is a more approximate version of
    :func:`recolor_burstsph_OU_gauss_R` using a fixed `δt`.

    Arguments:
        timestamps (pandas.Series): macrotimes of photons to be recolored.
            The index needs to have two levels: ('burst', 'ph').
        R0 (float): Förster radious of the D-A pair
        R_mean (float): mean D-A distance
        R_sigma (float): standard deviation of the distance distribution
        τ_relax (float): relaxation time of the OU process
        δt (float): time step for the diffusion-mediated D de-excitation.
            Same units as `timestamps`.
        rg (None or RandomGenerator): random number generator object,
            usually `numpy.random.RandomState()`. If None, use
            `numpy.random.RandomState()` with no seed. Use this to pass
            an RNG initialized with a specific seed or to choose a
            RNG other than numpy's default Mersen Twister MT19937.
    Returns:
        burstsph (pandas.DataFrame): DataFrame with 3 columns: 'timestamp'
            (same as input timestamps), 'nanotime' (simulated TCSPC nanotime)
            and 'stream' (color or the photon).
    """
    if rg is None:
        rg = np.random.RandomState()
    k_D = 1 / τ_D
    ts = timestamps.values
    A_em, R_ph, T_ph = depi_cy.sim_DA_from_timestamps2_cy(
        ts, δt, k_D, R0, R_mean, R_sigma, τ_relax, rg)
    A_mask = A_em.view(bool)
    T_ph = np.asarray(T_ph)
    T_ph[A_mask] += rg.exponential(scale=τ_A, size=A_mask.sum())
    burstsph_sim = pd.DataFrame(timestamps)
    burstsph_sim['nanotime'] = T_ph
    burstsph_sim['stream'] = (
        pd.Categorical.from_codes(A_em, categories=["DexDem", "DexAem"]))
    burstsph_sim['R_ph'] = R_ph
    return burstsph_sim


#
# - FOLLOWING FUNCTIONS ARE SLOW PYTHON VERSIONS FOR TESTING ONLY - - - -
#


def sim_DA_from_timestamps(timestamps, δt, k_D, R0,
                           R_mean, R_sigma, τ_relax, rg):
    """
    Recoloring using a fixed δt and no random number caching
    """
    A_ph = np.zeros(timestamps.size, dtype=bool)
    R_ph = np.zeros(timestamps.size, dtype=np.float64)
    T_ph = np.zeros(timestamps.size, dtype=np.float64)
    R = rg.randn() * R_sigma + R_mean
    t0 = 0
    for iph, t in enumerate(timestamps):
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t = t - t0  # current time - previous emission time
        if delta_t < 0:
            # avoid negative delta_t possible when when two photons have
            # the same macrotime
            delta_t = 0
            t = t0
        # computed `R_prev`, the D-A distance at the excitation time
        N = rg.randn()
        R = ou_single_step_cy(R, delta_t, N, R_mean, R_sigma, τ_relax)
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step δt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            # with current `R`, we compute the transition rates
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            # random trial to emit in the current time bin δt
            d_prob_ph_em = k_emission * δt  # prob. of emission in δt
            p = rg.rand()
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += δt
            N = rg.randn()
            R = ou_single_step_cy(R, δt, N, R_mean, R_sigma, τ_relax)

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


def sim_DA_from_timestamps_p(timestamps, δt_max, k_D, R0, R_mean, R_sigma,
                             τ_relax, rg, α=0.05, ndt=10):
    """
    Recoloring using adaptive δt and no random number caching.

    The adaptive δt is chosen so that k_emission * δt < 0.05,
    so that k_emission * δt is a good approximation for the exponential CDF
    (1 - exp(-k_emission δt)).
    """
    if τ_relax < ndt * δt_max:
        δt_max = τ_relax / ndt
        print(f'WARNING: Reducing δt_max to {δt_max:g}')
    A_ph = np.zeros(timestamps.size, dtype=bool)
    R_ph = np.zeros(timestamps.size, dtype=np.float64)
    T_ph = np.zeros(timestamps.size, dtype=np.float64)
    R = rg.randn() * R_sigma + R_mean
    t0 = 0
    for iph, t in enumerate(timestamps):
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t = t - t0
        if delta_t < 0:
            # avoid negative delta_t possible when when two photons have
            # the same macrotime
            delta_t = 0
            t = t0
        # computed `R_prev`, the D-A distance at the excitation time
        N = rg.randn()
        R = ou_single_step_cy(R, delta_t, N, R_mean, R_sigma, τ_relax)
        nanotime = 0
        # loop through D-A diffusion steps with a time-step δt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            δt = min(α / k_emission, δt_max)
            d_prob_ph_em = k_emission * δt  # prob. of emission in δt
            p = rg.rand()
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += δt
            N = rg.randn()
            R = ou_single_step_cy(R, δt, N, R_mean, R_sigma, τ_relax)

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


def sim_DA_from_timestamps_p2(timestamps, δt, k_D, R0, R_mean, R_sigma,
                              τ_relax, rg, α=0.05, ndt=10):
    """
    Recoloring using fixed δt, emission are using exponential CDF and
    no random number caching.

    The assumption here is that since τ_relax >> δt, distance and FRET
    will not change during δt, so the transition probability can be computed
    from the exponential CDF.
    """
    A_ph = np.zeros(timestamps.size, dtype=bool)
    R_ph = np.zeros(timestamps.size, dtype=np.float64)
    T_ph = np.zeros(timestamps.size, dtype=np.float64)
    if τ_relax < ndt * δt:
        δt = τ_relax / ndt
        print(f'WARNING: Reducing δt to {δt:g}')
    R = rg.randn() * R_sigma + R_mean
    t0 = 0
    for iph, t in enumerate(timestamps):
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t = t - t0
        if delta_t < 0:
            # avoid negative delta_t possible when when two photons have
            # the same macrotime
            delta_t = 0
            t = t0
        # `R` here is the D-A distance at the excitation time
        N = rg.randn()
        R = ou_single_step_cy(R, delta_t, N, R_mean, R_sigma, τ_relax)
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step δt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            d_prob_ph_em = k_emission * δt  # prob. of emission in δt
            if d_prob_ph_em > α:
                d_prob_ph_em = 1 - exp(-d_prob_ph_em)
            p = rg.rand()
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += δt
            # Update R following the OU process
            N = rg.randn()
            R = ou_single_step_cy(R, δt, N, R_mean, R_sigma, τ_relax)

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


def sim_DA_from_timestamps2(timestamps, dt, k_D, R0, R_mean, R_sigma,
                            τ_relax, rg, chunk_size=1000):
    """
    Recoloring using a fixed δt and with random number caching
    """
    R = rg.randn() * R_sigma + R_mean
    t0 = 0
    nanotime = 0
    # Array flagging photons as A (1) or D (0) emitted
    A_ph = np.zeros(timestamps.size, dtype=bool)
    # Instantaneous D-A distance at D de-excitation time
    R_ph = np.zeros(timestamps.size, dtype=np.float64)
    # Time of D de-excitation relative to the last timestamp
    T_ph = np.zeros(timestamps.size, dtype=np.float64)
    iN = chunk_size - 1
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
            #print(f'Outer rand iph:{iph}, iN:{iN}', flush=True)
            Na = memoryview(rg.randn(chunk_size))
            Pa = memoryview(rg.rand(chunk_size))
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R = ou_single_step_cy(R, delta_t, N, R_mean, R_sigma, τ_relax)
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            d_prob_ph_em = k_emission * dt  # prob. of emission in dt
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
            R = ou_single_step_cy(R, dt, N, R_mean, R_sigma, τ_relax)

        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / (k_ET + k_D)
        if prob_A_em >= p_DA:
            A_ph[iph] = True
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t + nanotime
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
    return A_ph, R_ph, T_ph


def sim_DA_from_timestamps2_p(timestamps, dt_max, k_D, R0, R_mean, R_sigma,
                              tau_relax, rg, chunk_size=1000,
                              alpha=0.05, ndt=10):
    """
    Recoloring using adaptive dt and with random number caching

    The adaptive dt is chosen so that `k_emission * dt < 0.05`,
    so that k_emission * dt is a good approximation for the exponential CDF
    (1 - exp(-k_emission dt)).
    """
    if tau_relax < ndt * dt_max:
        dt_max = tau_relax / ndt
        print(f'WARNING: Reducing dt_max to {dt_max:g} '
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
        #print(f'R: {R}, iph:{iph}, N:{N}, p:{p}')
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            dt = min(alpha / k_emission, dt_max)
            d_prob_ph_em = k_emission * dt  # prob. of emission in dt
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
            #print(f' - R: {R}, iph:{iph}, N:{N}, p:{p}, dt:{dt}')

        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / (k_ET + k_D)
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
    dt = np.array([dt_ref] * 2, dtype=np.float64)
    assert K_s.shape[0] == K_s.shape[1], 'K_s needs to be a square matrix.'
    num_states = K_s.shape[0]
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


#
# - FOLLOWING FUNCTIONS ARE NOT USED FOR RECOLORING - - - - - - - - - - -
#


@numba.jit(nopython=True)
def ou_process_core_numba(X, N, delta_t, ou_tau, ou_sigma):
    """Low-level function computing values of the OU process
    """
    for i in range(1, X.size):
        dt = delta_t[i - 1]
        dt_over_tau = dt / ou_tau
        relax = np.exp(-dt_over_tau)
        diffuse = ou_sigma * np.sqrt(1 - relax**2)
        X[i] = X[i - 1] * relax + diffuse * N[i]


def ou_process(delta_t, N, ou_mean, ou_sigma, ou_tau, x0=None):
    """Compute an Ornstein–Uhlenbeck (OU) process from a series of delta_t.

    Arguments:
        delta_t (array): intervals between time points where the OU process
            will be evaluated
        N (array): array of white Gaussian noise with sigma=1 of size
            size(delta_t) + 1.
        ou_mean (float): mean of the OU process
        ou_sigma (float): standard deviation of the relaxed OU process
        ou_tau (float): relaxation time of the OU process
        x0 (float): initial value for the OU process. If None,
            the initial value is `N[0] * ou_sigma`. If not None,
            `N[0]` is not used.
    Returns:
        Array of OU process values. Output size is `size(delta_t) + 1`
    """
    X = np.zeros(delta_t.size + 1, dtype=float)
    if x0 is None:
        X[0] = N[0] * ou_sigma
    else:
        X[0] = x0
    ou_process_core_numba(X, N, delta_t, ou_tau, ou_sigma)
    X += ou_mean
    return X


def E_sim_gauss_R_burst(burst_size_series, R0, R_mean, τ_relax=0,
                        R_sigma=0, oversample=100, dithering_sigma=0.02):
    """Simulate a E for each burst using a fixed D-A distance per burst.

    Distances are drawn from a Gaussian distribution.

    Arguments:
        burst_size_series (pandas.Series): series of burst sizes
    """
    size_counts = burst_size_series.value_counts()
    E_sim = []
    for burst_size, counts in size_counts.iteritems():
        num_bursts = oversample * counts
        R = np.random.randn(num_bursts) * R_sigma + R_mean
        assert (R >= 0).all()
        E = E_from_dist(R, R0)
        assert (E >= 0).all() and (E <= 1).all()
        na_sim = np.random.binomial(n=burst_size, p=E, size=num_bursts)
        E_sim.append(na_sim / burst_size)
    E_sim = np.hstack(E_sim)
    assert E_sim.size == size_counts.sum() * oversample
    E_sim += np.random.randn(E_sim.size) * dithering_sigma
    return E_sim


def E_sim_gauss_R_ph(burst_size_series, R0, R_mean, τ_relax=0,
                     R_sigma=0, oversample=100, dithering_sigma=0.02):
    """Simulate E for each burst using a new Gaussian D-A distance per photon

    - R drawn from Gaussian distribution for each photon in each burst
    - Group by burst size

    Arguments:
        burst_size_series (pandas.Series): series of burst sizes
    """
    size_counts = burst_size_series.value_counts()
    E_sim = []
    for burst_size, counts in size_counts.iteritems():
        num_bursts = oversample * counts
        R = np.random.randn(num_bursts, burst_size) * R_sigma + R_mean
        assert (R >= 0).all()
        E = E_from_dist(R, R0)
        assert (E >= 0).all() and (E <= 1).all()
        na_sim = np.random.binomial(n=1, p=E).sum(axis=1)
        E_sim.append(na_sim / burst_size)
    E_sim = np.hstack(E_sim)
    assert E_sim.size == size_counts.sum() * oversample
    E_sim += np.random.randn(E_sim.size) * dithering_sigma
    return E_sim


def E_sim(E_ph, oversample=100, dithering_sigma=0.02):
    """Simulate E for each burst using the passed `E_ph` for each photon.
    """
    num_bursts = oversample
    E_sim = []
    for ib, E in E_ph.groupby('burst'):
        burst_size = E.shape[0]
        assert (E >= 0).all() and (E <= 1).all()
        na_sim = (np.random.binomial(n=1, p=E, size=(num_bursts, burst_size))
                  .sum(axis=1))
        E_sim.append(na_sim / burst_size)
    E_sim = np.hstack(E_sim)
    E_sim += np.random.randn(E_sim.size) * dithering_sigma
    return E_sim
