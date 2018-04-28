from math import exp
import numpy as np
import pandas as pd
import depi_cy
from depi_cy import ou_single_step_cy


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
