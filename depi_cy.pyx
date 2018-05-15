"""
Build this module with:

python setup.py build_ext --inplace

"""

import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
import ctmc


@cython.boundscheck(False)
cdef inline np.complex128_t[:,:] matmul(
        np.complex128_t[:,:] A,
        np.complex128_t[:,:] B,
        np.complex128_t[:,:] out,
        Py_ssize_t n):
    cdef temp
    for i in range(n):
        for j in range(n):
            temp = 0
            for k in range(n):
                temp += A[i,k] * B[k,j]
            out[i,j] = temp
    return out


@cython.cdivision(True)
cpdef inline double ou_single_step_cy(double x0, double delta_t, double N,
                                      double ou_mean, double ou_sigma,
                                      double ou_tau):
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
def sim_DA_from_timestamps2_p2_cy(
        np.int64_t[:] timestamps, double dt, double[:] K_D, double[:] D_fract,
        double R0,
        double R_mean, double R_sigma, double tau_relax,
        *, double gamma=1.0, rg=np.random.RandomState(),
        int chunk_size=1000, double alpha=0.05, double ndt=10):
    cdef double R_ou = rg.randn() * R_sigma
    cdef double R, R_prev, delta_t, nanotime, k_ET, d_prob_ph_em, k_emission
    cdef double p, N
    cdef np.int64_t t, t0, num_D_lifetimes = len(K_D)
    cdef double p_DA, prob_A_em, k_D
    cdef Py_ssize_t iph, iN, iD_comp
    cdef Py_ssize_t[:] D_comps = np.arange(num_D_lifetimes, dtype='int64')
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
    if num_D_lifetimes == 1:
        k_D = K_D[0]
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
        if num_D_lifetimes > 1:
            iD_comp = rg.choice(D_comps, size=1, p=D_fract)[0]
            k_D = K_D[iD_comp]
        R_prev = R
        R_ou = ou_single_step_cy0(R_ou, delta_t, N, R_sigma, tau_relax)
        R = R_ou + R_mean
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation (by photon emission or energy transfer to A)
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
        prob_A_em = k_ET / (k_ET + k_D / gamma)
        if prob_A_em >= p_DA:
            A_ph[iph] = True
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
    return np.asarray(A_ph), np.asarray(R_ph), np.asarray(T_ph)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sim_DA_from_timestamps2_p2_dist_cy(
        np.int64_t[:] timestamps, double dt, double[:] K_D, double[:] D_fract,
        double R0, double tau_relax,
        double[:] r_dd, Py_ssize_t idx_offset_dd, double du_norm,
        *, double gamma=1.0, rg=np.random.RandomState(),
        int chunk_size=1000, double alpha=0.05, double ndt=10):
    cdef double R_ou = rg.randn()
    cdef double R, delta_t, nanotime, k_ET, d_prob_ph_em, k_emission
    cdef double p, N
    cdef np.int64_t t, t0, num_D_lifetimes = len(K_D)
    cdef double p_DA, prob_A_em, k_D
    cdef Py_ssize_t iph, iN, ix, iD_comp
    cdef Py_ssize_t[:] D_comps = np.arange(num_D_lifetimes, dtype='int64')
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
    if num_D_lifetimes == 1:
        k_D = K_D[0]
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
        if num_D_lifetimes > 1:
            iD_comp = rg.choice(D_comps, size=1, p=D_fract)[0]
            k_D = K_D[iD_comp]
        R_ou = ou_single_step_cy0(R_ou, delta_t, N, 1, tau_relax)
        ix = int(round(R_ou / du_norm)) + idx_offset_dd
        R = r_dd[ix]
        nanotime = 0
        # loop through D-A diffusion steps with a fixed time-step dt
        # until D de-excitation (by photon emission or energy transfer to A)
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
                Na = rg.randn(chunk_size)
                Pa = rg.rand(chunk_size)
                iN = 0
            N = Na[iN]
            p = Pa[iN]
            # Update R following the OU process
            R_ou = ou_single_step_cy0(R_ou, dt, N, 1, tau_relax)
            ix = int(round(R_ou / du_norm)) + idx_offset_dd
            R = r_dd[ix]

        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / (k_ET + k_D / gamma)
        if prob_A_em >= p_DA:
            A_ph[iph] = True
        # time of D de-excitation by photon emission or energy transfer to A
        t0 = t
        # save D-A distance at emission time
        R_ph[iph] = R
        # save time of emission relative to the excitation time `t`
        T_ph[iph] = nanotime
    return np.asarray(A_ph), np.asarray(R_ph), np.asarray(T_ph)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sim_DA_from_timestamps2_p2_2states_cy(
        np.int64_t[:] timestamps, double dt_ref, double k_D, double R0,
        double[:] R_mean, double[:] R_sigma,
        double[:] tau_relax, double[:] k_s,
        *, double gamma=1.0, rg=np.random.RandomState(),
        int chunk_size=1000, double alpha=0.05, double ndt=10):
    cdef double[:] dt = np.array([dt_ref]*2, dtype=np.float64)
    cdef double R_ou
    cdef double R, R_prev, delta_t, delta_t0, nanotime, k_ET, d_prob_ph_em, k_emission
    cdef double k_s_sum, p, N, p_state, u
    #cdef double R_mean_i, R_sigma_i, tau_relax_i, dt_i
    cdef np.int64_t t, t0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN
    cdef np.uint8_t state = 0
    cdef np.float64_t[:] peq = np.zeros(2, dtype=np.float64)
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
    peq[0] = k_s[1] / (k_s[0] + k_s[1])
    peq[1] = k_s[0] / (k_s[0] + k_s[1])
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
        p_state = (1 - peq[state]) * exp(-(delta_t0 * k_s_sum)) + peq[state]
        u = rg.rand()
        # Inversion of u is for compatibility with N-state version
        if state == 1:
            u = 1 - u
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
def sim_DA_from_timestamps2_p2_2states_dist_cy(
        np.int64_t[:] timestamps, double dt_ref, double k_D, double R0,
        double[:] tau_relax, double[:] k_s,
        double[:,:] r_dd, Py_ssize_t[:] idx_offset_dd, double du_norm,
        *, double gamma=1.0, rg=np.random.RandomState(),
        int chunk_size=1000, double alpha=0.05, double ndt=10):
    cdef double[:] dt = np.array([dt_ref]*2, dtype=np.float64)
    cdef double R_ou
    cdef double R, R_prev, delta_t, delta_t0, nanotime, k_ET, d_prob_ph_em, k_emission
    cdef double k_s_sum, p, N, p_state, u
    cdef np.int64_t t, t0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN, ix
    cdef np.uint8_t state = 0
    cdef np.float64_t[:] peq = np.zeros(2, dtype=np.float64)
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
    peq[0] = k_s[1] / (k_s[0] + k_s[1])
    peq[1] = k_s[0] / (k_s[0] + k_s[1])
    state = 0  # the two states are 0 and 1
    R_ou = rg.randn()
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
        p_state = (1 - peq[state]) * exp(-(delta_t0 * k_s_sum)) + peq[state]
        u = rg.rand()
        # Inversion of u is for compatibility with N-state version
        if state == 1:
            u = 1 - u
        if p_state <= u:
            state = 0 if state == 1 else 1
            R_ou = rg.randn()
        # Compute the D-A distance at the "excitation time"
        iN += 1
        if iN == chunk_size:
            Na = rg.randn(chunk_size)
            Pa = rg.rand(chunk_size)
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R_ou = ou_single_step_cy0(R_ou, delta_t, N, 1, tau_relax[state])
        ix = int(round(R_ou / du_norm)) + idx_offset_dd[state]
        R = r_dd[state, ix]
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
            R_ou = ou_single_step_cy0(R_ou, dt[state], N, 1, tau_relax[state])
            ix = int(round(R_ou / du_norm)) + idx_offset_dd[state]
            R = r_dd[state, ix]
        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / (k_ET + k_D / gamma)
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
def sim_DA_from_timestamps2_p2_Nstates_dist_cy(
        np.int64_t[:] timestamps, double dt_ref, double k_D, double R0,
        double[:] tau_relax, double[:,:] K_s,
        double[:,:] r_dd, Py_ssize_t[:] idx_offset_dd, double du_norm,
        *, double gamma=1.0, rg=np.random.RandomState(),
        int chunk_size=1000, double alpha=0.05, double ndt=10):
    cdef double[:] dt
    cdef double R_ou
    cdef double R, R_prev, delta_t, delta_t0, nanotime, k_ET, d_prob_ph_em, k_emission
    cdef double p, N, p_state, u
    cdef np.int64_t t, t0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN, s, i, ix
    cdef Py_ssize_t state, newstate
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
    assert K_s.shape[0] == K_s.shape[1], 'K_s needs to be a square matrix.'
    cdef np.uint8_t num_states = K_s.shape[0]
    cdef np.float64_t[:] state_vector, p_states = np.zeros(num_states, dtype=np.float64)
    cdef np.complex128_t[:,:] D = np.zeros((num_states, num_states), dtype=np.complex128)

    dt = np.array([dt_ref]*num_states, dtype=np.float64)
    for state in range(num_states):
        if tau_relax[state] < ndt * dt[state]:
            dt[state] = tau_relax[state] / ndt
            print(f'WARNING: Reducing dt[{state}] to {dt[state]:g} '
                  f'[tau_relax[{state}] = {tau_relax[state]}]')
    state = 0
    newstate = 0
    state_vector = np.zeros(num_states)
    state_vector[state] = 1
    p_states[0] = 0.5
    p_states[1] = 0.5
    R_ou = rg.randn()
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    eigenval, V, V_inv = ctmc.decompose(K_s)
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
        # Select the new state
        for i in range(num_states):
            D[i,i] = eigenval[i] * t
        P_t_matrix = (V @ D @ V_inv).real
        p_states = state_vector @ P_t_matrix
        u = rg.rand()
        for s in range(num_states):
            p_state = p_states[s]
            if u <= p_state:
                newstate = s
                break
            u -= p_state
        if newstate != state:
            state_vector[state] = 0
            state = newstate
            state_vector[state] = 1
            R_ou = rg.randn()
        # Compute the D-A distance at the "excitation time"
        iN += 1
        if iN == chunk_size:
            Na = rg.randn(chunk_size)
            Pa = rg.rand(chunk_size)
            iN = 0
        N = Na[iN]
        p = Pa[iN]
        R_ou = ou_single_step_cy0(R_ou, delta_t, N, 1, tau_relax[state])
        ix = int(round(R_ou / du_norm)) + idx_offset_dd[state]
        R = r_dd[state, ix]
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
            R_ou = ou_single_step_cy0(R_ou, dt[state], N, 1, tau_relax[state])
            ix = int(round(R_ou / du_norm)) + idx_offset_dd[state]
            R = r_dd[state, ix]
        # photon emitted, let's decide if it is from D or A
        p_DA = p / d_prob_ph_em  # equivalent to rand(), but faster
        prob_A_em = k_ET / (k_ET + k_D / gamma)
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


def _state_sel(R_ou, num_states, state, t, eigenval, V, V_inv, D, P_t_matrix,
               state_vector, rg):
        for i in range(num_states):
            D[i,i] = eigenval[i] * t
        P_t_matrix = (V @ D @ V_inv).real
        p_states = state_vector @ P_t_matrix
        u = rg.rand()
        for s in range(num_states):
            p_state = p_states[s]
            if u <= p_state:
                newstate = s
                break
            u -= p_state
        if newstate != state:
            state_vector[state] = 0
            state_vector[newstate] = 1
            R_ou = rg.randn()
        return newstate, R_ou


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sim_DA_from_timestamps2_p2_Nstates_cy(
        np.int64_t[:] timestamps, double dt_ref, double k_D, double R0,
        double[:] R_mean, double[:] R_sigma,
        double[:] tau_relax, double[:,:] K_s,
        *, double gamma=1.0, rg=np.random.RandomState(),
        int chunk_size=1000, double alpha=0.05, double ndt=10):
    cdef double[:] dt
    cdef double R_ou
    cdef double R, R_prev, delta_t, delta_t0, nanotime, k_ET, d_prob_ph_em, k_emission
    #cdef double R_mean_i, R_sigma_i, tau_relax_i, dt_i
    cdef double p, N, p_state, u
    cdef np.int64_t t, t0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN, s, i
    cdef Py_ssize_t state, newstate
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
    assert K_s.shape[0] == K_s.shape[1], 'K_s needs to be a square matrix.'
    cdef np.uint8_t num_states = K_s.shape[0]
    cdef np.float64_t[:] state_vector, p_states = np.zeros(num_states, dtype=np.float64)
    #cdef np.complex128_t[:] eigenval
    #cdef np.float64_t[:,:] V, V_inv
    cdef np.complex128_t[:,:] D = np.zeros((num_states, num_states), dtype=np.complex128)
    dt = np.array([dt_ref]*num_states, dtype=np.float64)
    for state in range(num_states):
        if tau_relax[state] < ndt * dt[state]:
            dt[state] = tau_relax[state] / ndt
            print(f'WARNING: Reducing dt[{state}] to {dt[state]:g} '
                  f'[tau_relax[{state}] = {tau_relax[state]}]')
    state = 0
    newstate = 0
    state_vector = np.zeros(num_states)
    state_vector[state] = 1
    p_states[0] = 0.5
    p_states[1] = 0.5
    R_ou = rg.randn() * R_sigma[state]
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    eigenval, V, V_inv = ctmc.decompose(K_s)
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
        for i in range(num_states):
            D[i,i] = eigenval[i] * t
        P_t_matrix = (V @ D @ V_inv).real
        p_states = state_vector @ P_t_matrix
        u = rg.rand()
        for s in range(num_states):
            p_state = p_states[s]
            if u <= p_state:
                newstate = s
                break
            u -= p_state
        if newstate != state:
            state_vector[state] = 0
            state = newstate
            state_vector[state] = 1
            R_ou = rg.randn() * R_sigma[state]
            # R_mean_i = R_mean[state]
            # R_sigma_i = R_sigma[state]
            # tau_relax_i = tau_relax[state]
            # dt_i = dt[state]
        #print(f'iph={iph}; delta_t0={delta_t0}, state={state}, R={R_ou + R_mean[state]}, u={u}, p_states={p_states}')
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
        prob_A_em = k_ET / (k_ET + k_D / gamma)
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
def sim_DA_from_timestamps2_p2_Nstates_cym(
        np.int64_t[:] timestamps, double dt_ref, double k_D, double R0,
        double[:] R_mean, double[:] R_sigma,
        double[:] tau_relax, double[:,:] K_s,
        *, double gamma=1.0, rg=np.random.RandomState(),
        int chunk_size=1000, double alpha=0.05, double ndt=10):
    cdef double[:] dt
    cdef double R_ou
    cdef double R, R_prev, delta_t, delta_t0, nanotime, k_ET, d_prob_ph_em, k_emission
    #cdef double R_mean_i, R_sigma_i, tau_relax_i, dt_i
    cdef double p, N, p_state, u
    cdef np.int64_t t, t0
    cdef double p_DA, prob_A_em
    cdef Py_ssize_t iph, iN, s, i
    cdef Py_ssize_t state, newstate
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
    assert K_s.shape[0] == K_s.shape[1], 'K_s needs to be a square matrix.'
    cdef np.uint8_t num_states = K_s.shape[0]
    cdef np.float64_t[:] state_vector, p_states = np.zeros(num_states, dtype=np.float64)
    #cdef np.complex128_t[:] eigenval
    #cdef np.float64_t[:,:] V, V_inv
    cdef np.complex128_t[:,:] D = np.zeros((num_states, num_states), dtype=np.complex128)
    cdef np.complex128_t[:,:] P_t_matrix = np.zeros((num_states, num_states), dtype=np.complex128)
    dt = np.array([dt_ref]*num_states, dtype=np.float64)
    for state in range(num_states):
        if tau_relax[state] < ndt * dt[state]:
            dt[state] = tau_relax[state] / ndt
            print(f'WARNING: Reducing dt[{state}] to {dt[state]:g} '
                  f'[tau_relax[{state}] = {tau_relax[state]}]')
    state = 0
    newstate = 0
    state_vector = np.zeros(num_states)
    state_vector[state] = 1
    p_states[0] = 0.5
    p_states[1] = 0.5
    R_ou = rg.randn() * R_sigma[state]
    iN = chunk_size - 1  # value to get the first chunk of random numbers
    eigenval, V, V_inv = ctmc.decompose(K_s)
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
        for i in range(num_states):
            D[i,i] = eigenval[i] * t

        matmul(D, V_inv, P_t_matrix, num_states)
        matmul(V, P_t_matrix, P_t_matrix, num_states)
        p_states = state_vector @ np.real(P_t_matrix)
        #p_states = ctmc.occupancy(
        #    delta_t0, state_vector, K_s, eigenval, V, V_inv)
        u = rg.rand()
        for s in range(num_states):
            p_state = p_states[s]
            if u <= p_state:
                newstate = s
                break
            u -= p_state
        if newstate != state:
            state_vector[state] = 0
            state = newstate
            state_vector[state] = 1
            R_ou = rg.randn() * R_sigma[state]
            # R_mean_i = R_mean[state]
            # R_sigma_i = R_sigma[state]
            # tau_relax_i = tau_relax[state]
            # dt_i = dt[state]
        #print(f'iph={iph}; delta_t0={delta_t0}, state={state}, R={R_ou + R_mean[state]}, u={u}, p_states={p_states}')
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
        prob_A_em = k_ET / (k_ET + k_D / gamma)
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


#
# Function implementing different simulation approaches, used for testing.
#

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
    return np.asarray(A_ph), np.asarray(R_ph), np.asarray(T_ph)


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
