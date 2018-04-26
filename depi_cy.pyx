"""
Build this module with:

python setup.py build_ext --inplace

"""

import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt

# Uncomment this block to enable profiling with line_profiler
# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()
# directive_defaults['linetrace'] = False
# directive_defaults['binding'] = False


@cython.cdivision(True)
cpdef inline double ou_single_step_cy(double x0, double delta_t, double N,
                                      double ou_mean, double ou_sigma,
                                      double ou_tau) nogil:
    """Compute the next value of an OU process starting from `x0`
    """
    cdef double dt_over_tau = delta_t / ou_tau
    cdef double relax = exp(-dt_over_tau)
    cdef double diffuse = ou_sigma * sqrt(1 - relax**2)
    return (x0 - ou_mean) * relax + diffuse * N + ou_mean


@cython.cdivision(True)
cdef inline double ou_single_step_cy0(double x0, double delta_t, double N,
                                      double ou_sigma, double ou_tau) nogil:
    """Compute the next value of an OU process starting from `x0`
    This version uses a zero-mean OU process as micro-optimization.
    """
    cdef double dt_over_tau = delta_t / ou_tau
    cdef double relax = exp(-dt_over_tau)
    cdef double diffuse = ou_sigma * sqrt(1 - relax**2)
    return x0 * relax + diffuse * N


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sim_DA_from_timestamps2_p_cy(np.int64_t[:] timestamps, double dt_max,
                                 double k_D, double R0, double R_mean,
                                 double R_sigma, double tau_relax, rg,
                                 int chunk_size=1000,
                                 double alpha=0.05, double ndt=10):
    cdef double R_ou = rg.randn() * R_sigma
    cdef double R, delta_t, dt, nanotime, k_ET, d_prob_ph_em, k_emission
    cdef np.int64_t t, t0#, ni=0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN
    # Generate random number in chunks for efficiency
    cdef np.float64_t[:] Na, Pa
    # Array flagging photons as A (1) or D (0) emitted
    cdef np.uint8_t[:] A_ph = np.zeros(len(timestamps), dtype=np.uint8)
    # Istantaneous D-A distance at D de-excitation time
    cdef np.float64_t[:] R_ph = np.zeros(timestamps.size, dtype=np.float64)
    # Time of D de-excitation relative to the last timestamp
    cdef np.float64_t[:] T_ph = np.zeros(timestamps.size, dtype=np.float64)
    if tau_relax < ndt * dt_max:
        dt_max = tau_relax / ndt
        print(f'WARNING: Reducing dt_max to {dt_max:g} '
              f'[tau_relax = {tau_relax}]')
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    t0 = 0
    nanotime = 0
    for iph in range(len(timestamps)):
        t = timestamps[iph]
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t = (t - t0) - nanotime
        if delta_t < 0:
            delta_t = 0
            t = t0
        # Compute the D-A distance at the "excitation time"
        iN += 1
        if iN == chunk_size:
            Na = rg.randn(chunk_size)
            Pa = rg.rand(chunk_size)
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R_ou = ou_single_step_cy0(R_ou, delta_t, N, R_sigma, tau_relax)
        R = R_ou + R_mean
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation (by photon emission or energy transfer to A)
        while True:
            # if R <= 0:
            #     raise ValueError(f'timestamps2_p_cy: R = {R}')
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            dt = dt_max if dt_max <= alpha/k_emission else alpha/k_emission
            # if dt == alpha/k_emission:
            #     ni += 1
            d_prob_ph_em = dt * k_emission
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += dt
            iN += 1
            if iN == chunk_size:
                Na = rg.randn(chunk_size)
                Pa = rg.rand(chunk_size)
                iN = 0
            N = Na[iN]
            p = Pa[iN]
            # Update R following the OU process
            R_ou = ou_single_step_cy0(R_ou, dt, N, R_sigma, tau_relax)
            R = R_ou + R_mean

        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / k_emission
        if prob_A_em >= p_DA:
            A_ph[iph] = True
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
    #print(ni)
    return np.asarray(A_ph), np.asarray(R_ph), np.asarray(T_ph)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sim_DA_from_timestamps2_p2_cy(np.int64_t[:] timestamps, double dt,
                                  double k_D, double R0, double R_mean,
                                  double R_sigma, double tau_relax, rg,
                                  int chunk_size=1000,
                                  double alpha=0.05, double ndt=10):
    cdef double R_ou = rg.randn() * R_sigma
    cdef double R, R_prev, delta_t, nanotime, k_ET, d_prob_ph_em, k_emission
    cdef np.int64_t t, t0#, ni=0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN
    # Generate random number in chunks for efficiency
    cdef np.float64_t[:] Na, Pa
    # Array flagging photons as A (1) or D (0) emitted
    cdef np.uint8_t[:] A_ph = np.zeros(len(timestamps), dtype=np.uint8)
    # Istantaneous D-A distance at D de-excitation time
    cdef np.float64_t[:] R_ph = np.zeros(timestamps.size, dtype=np.float64)
    # Time of D de-excitation relative to the last timestamp
    cdef np.float64_t[:] T_ph = np.zeros(timestamps.size, dtype=np.float64)
    if tau_relax < ndt * dt:
        dt = tau_relax / ndt
        print(f'WARNING: Reducing dt to {dt:g} '
              f'[tau_relax = {tau_relax}]')
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    t0 = 0
    nanotime = 0
    for iph in range(len(timestamps)):
        t = timestamps[iph]
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t = (t - t0) - nanotime
        if delta_t < 0:
            delta_t = 0
            t = t0
        # Compute the D-A distance at the "excitation time"
        iN += 1
        if iN == chunk_size:
            Na = rg.randn(chunk_size)
            Pa = rg.rand(chunk_size)
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R_prev = R
        R_ou = ou_single_step_cy0(R_ou, delta_t, N, R_sigma, tau_relax)
        R = R_ou + R_mean
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation (by photon emission or energy transfer to A)
        while True:
            # if R <= 0:
            #     raise ValueError(f'timestamps2_p2_cy: iph = {iph}; nanotime = {nanotime}; R = {R}; R_prev = {R_prev}; delta_t = {delta_t}; dt = {dt}; N = {N}; R_sigma = {R_sigma}; tau_relax = {tau_relax}')
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            d_prob_ph_em = dt * k_emission
            if d_prob_ph_em > alpha:
                d_prob_ph_em = 1 - exp(-d_prob_ph_em)
                #ni += 1
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += dt
            iN += 1
            if iN == chunk_size:
                Na = rg.randn(chunk_size)
                Pa = rg.rand(chunk_size)
                iN = 0
            N = Na[iN]
            p = Pa[iN]
            # Update R following the OU process
            R_prev = R
            R_ou = ou_single_step_cy0(R_ou, dt, N, R_sigma, tau_relax)
            R = R_ou + R_mean

        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / k_emission
        if prob_A_em >= p_DA:
            A_ph[iph] = True
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
    return np.asarray(A_ph), np.asarray(R_ph), np.asarray(T_ph)


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
def sim_DA_from_timestamps2_p2_2states_cy(np.int64_t[:] timestamps,
        double dt_ref, double k_D, double R0, double[:] R_mean,
        double[:] R_sigma, double[:] tau_relax, double[:] k_s, *, rg,
        int chunk_size=1000, double alpha=0.05, double ndt=10):
    cdef double[:] dt = np.array([dt_ref]*2, dtype=np.float64)
    cdef double R_ou
    cdef double R, R_prev, delta_t, delta_t0, nanotime, k_ET, d_prob_ph_em, k_emission
    cdef double k_s_sum, p_eq
    #cdef double R_mean_i, R_sigma_i, tau_relax_i, dt_i
    cdef np.int64_t t, t0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN
    cdef np.uint8_t state = 0
    # Generate random number in chunks for efficiency
    cdef np.float64_t[:] Na, Pa
    # Array flagging photons as A (1) or D (0) emitted
    cdef np.uint8_t[:] A_ph = np.zeros(len(timestamps), dtype=np.uint8)
    # Istantaneous D-A distance at D de-excitation time
    cdef np.float64_t[:] R_ph = np.zeros(len(timestamps), dtype=np.float64)
    # Time of D de-excitation relative to the last timestamp
    cdef np.float64_t[:] T_ph = np.zeros(len(timestamps), dtype=np.float64)
    # State for each photon
    cdef np.uint8_t[:] S_ph = np.zeros(len(timestamps), dtype=np.uint8)
    for state in [0, 1]:
        if tau_relax[state] < ndt * dt[state]:
            dt[state] = tau_relax[state] / ndt
            print(f'WARNING: Reducing dt[{state}] to {dt[state]:g} '
                  f'[tau_relax[{state}] = {tau_relax[state]}]')
    k_s_sum = np.sum(k_s)
    peq = [k_s[1] / (k_s[0] + k_s[1]),
           k_s[0] / (k_s[0] + k_s[1])]
    state = 0  # the two states are 0 and 1
    R_ou = rg.randn() * R_sigma[state]
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    t0 = 0
    nanotime = 0
    for iph in range(len(timestamps)):
        t = timestamps[iph]
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
        if p_state <= u:
            state = 0 if state == 1 else 1
            R_ou = rg.randn() * R_sigma[state]
            # R_mean_i = R_mean[state]
            # R_sigma_i = R_sigma[state]
            # tau_relax_i = tau_relax[state]
            # dt_i = dt[state]
        # Compute the D-A distance at the "excitation time"
        iN += 1
        if iN == chunk_size:
            Na = rg.randn(chunk_size)
            Pa = rg.rand(chunk_size)
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R_ou = ou_single_step_cy0(R_ou, delta_t, N, R_sigma[state], tau_relax[state])
        R = R_ou + R_mean[state]
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation (by photon emission or energy transfer to A)
        while True:
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            d_prob_ph_em = k_emission * dt[state]
            if d_prob_ph_em > alpha:
                d_prob_ph_em = 1 - exp(-d_prob_ph_em)
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += dt[state]
            iN += 1
            if iN == chunk_size:
                Na = rg.randn(chunk_size)
                Pa = rg.rand(chunk_size)
                iN = 0
            N = Na[iN]
            p = Pa[iN]
            # Update R following the OU process
            R_ou = ou_single_step_cy0(R_ou, dt[state], N, R_sigma[state], tau_relax[state])
            R = R_ou + R_mean[state]
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
    return np.asarray(A_ph), np.asarray(R_ph), np.asarray(T_ph), np.asarray(S_ph)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef sim_DA_from_timestamps_cy(np.int64_t[:] timestamps, double dt,
                                double k_D, double R0, double R_mean,
                                double R_sigma, double tau_relax, rg):
    cdef double R_ou = rg.randn() * R_sigma
    cdef np.uint8_t[:] A_em = np.zeros(len(timestamps), dtype=np.uint8)
    cdef double[:] R_ph = np.zeros(timestamps.size, dtype=np.float64)
    cdef double[:] T_ph = np.zeros(timestamps.size, dtype=np.float64)
    cdef double t, t0, delta_t, N, R, nanotime, p, k_ET, k_emission, d_prob_ph_em
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph
    t0 = 0
    for iph in range(len(timestamps)):
        t = timestamps[iph]
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t = t - t0
        if delta_t < 0:
            delta_t = 0
            t = t0
        # `R` here is the D-A distance at the excitation time
        N = rg.randn()
        R_ou = ou_single_step_cy0(R_ou, delta_t, N, R_sigma, tau_relax)
        R = R_ou + R_mean
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation by photon emission or energy transfer to A
        while True:
            # with current `R`, we compute the rate of energy transfer k_ET
            k_ET = k_D * (R0 / R)**6
            k_emission = k_ET + k_D
            # random trial to emit in the current time bin dt
            d_prob_ph_em = k_emission * dt  # prob. of emission in dt
            p = rg.rand()
            if d_prob_ph_em >= p:
                break   # break out of the loop when the photon is emitted
            nanotime += dt
            N = rg.randn()
            R_ou = ou_single_step_cy0(R_ou, dt, N, R_sigma, tau_relax)
            R = R_ou + R_mean
        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / k_emission
        if prob_A_em >= p_DA:
            A_em[iph] = True
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t + nanotime
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
    return np.asarray(A_em), np.asarray(R_ph), np.asarray(T_ph)


#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.cdivision(True)
def sim_DA_from_timestamps2_cy(np.int64_t[:] timestamps, double dt,
                               double k_D, double R0, double R_mean,
                               double R_sigma, double tau_relax, rg):
    cdef double R_prev = rg.randn() * R_sigma
    cdef double delta_t, R, nanotime, k_ET, d_prob_ph_em
    cdef double dt_kD = dt * k_D
    cdef np.int64_t t, t0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN, iNp
    # Generate random number in chunks for efficiency
    cdef int chunk_size = 1000
    cdef np.float64_t[:] Na, Pa, Np
    # Array flagging photons as A (1) or D (0) emitted
    cdef np.uint8_t[:] A_ph = np.zeros(len(timestamps), dtype=np.uint8)
    # Istantaneous D-A distance at D de-excitation time
    cdef np.float64_t[:] R_ph = np.zeros(timestamps.size, dtype=np.float64)
    # Time of D de-excitation relative to the last timestamp
    cdef np.float64_t[:] T_ph = np.zeros(timestamps.size, dtype=np.float64)

    t0 = 0
    nanotime = 0
    Na = rg.randn(chunk_size)
    Pa = rg.rand(chunk_size)
    Np = rg.randn(len(timestamps))
    iN = 0
    for iph in range(len(timestamps)):
        t = timestamps[iph]
        # each cycle starts with a new photon timestamp `t`
        # excitation time is `t`, emission time is `t + nanotime`
        delta_t = (t - t0) - nanotime
        if delta_t < 0:
            delta_t = 0
            t = t0
        # `R_prev` here is the D-A distance at the excitation time `t`
        R_prev = ou_single_step_cy0(R_prev, delta_t, Np[iph], R_sigma, tau_relax)
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation (by photon emission or energy transfer to A)
        while True:
            if iN == chunk_size:
                Na = rg.randn(chunk_size)
                Pa = rg.rand(chunk_size)
                iN = 0
            R_ou = ou_single_step_cy0(R_prev, dt, Na[iN], R_sigma, tau_relax)
            R = R_ou + R_mean
            k_ET = k_D * (R0 / R)**6
            d_prob_ph_em = dt * (k_ET + k_D)
            if d_prob_ph_em >= Pa[iN]:
                break   # break out of the loop when the photon is emitted
            iN += 1
            R_prev = R_ou
            nanotime += dt

        # photon emitted, let's decide if it is from D or A
        p_DA = Pa[iN] / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / (k_ET + k_D)
        if prob_A_em >= p_DA:
            A_ph[iph] = True
        # time of D de-excitation by photon emission or energy transfer to A
        iN += 1
        t0 = t
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
    return np.asarray(A_ph), np.asarray(R_ph), np.asarray(T_ph)
