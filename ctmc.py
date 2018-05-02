import numpy as np
import scipy.linalg as la


def G_from_Q(Q):
    G = Q.copy()
    G[np.diag_indices_from(G)] = -leaving_rates(Q)
    return G


def Q_from_G(G):
    Q = G.copy()
    Q[np.diag_indices_from(Q)] = 0
    return Q


def leaving_rates(G):
    Q = Q_from_G(G)
    return Q.sum(1)


def entering_rates(G):
    Q = Q_from_G(G)
    return Q.sum(0)


def P_from_G(G):
    Q = Q_from_G(G)
    return Q / leaving_rates(Q)[np.newaxis, :].T


def CT_stationary_distribution(G):
    """
    Continuous-time stationary distribution computed from
    the left eigenvector of the zero eigen value.
    """
    λ, V = la.eig(G.T)
    i_zero_left_eig = np.where(np.abs(λ) < 1e-12)[0]
    π = (V[:, i_zero_left_eig].T[0] / V[:, i_zero_left_eig].sum()).real
    return π


def CT_stationary_distribution_from_psi(G, ψ):
    """
    Continuous-time stationary distribution computed from
    the stationary distribution ψ of the embedded chain.
    """
    v_inv = 1 / leaving_rates(G)
    π = (ψ * v_inv) / np.dot(ψ, v_inv)
    return π


def DT_embedded_stationary_distribution_from_pi(G, π):
    """
    Discrete-time stationary distribution of the embedded chain
    computed from the continous-time stationary distribution π.
    """
    v = leaving_rates(G)
    ψ = π * v / np.dot(π, v)
    return ψ


def DT_embedded_stationary_distribution(G, n=50):
    """
    Discrete-time stationary distribution of the embedded chain
    computed by iterative transition matrix multiplication.
    """
    P = P_from_G(G)
    ψ0 = np.ones((1, P.shape[0]))
    ψ = ψ0 @ np.linalg.matrix_power(P, n)
    ψ /= ψ.sum()
    return ψ


def transition_matrix_t(t, G):
    λ, V = la.eig(G)
    P_t = (V @ np.diag(np.exp(λ * t)) @ la.inv(V)).real
    return P_t


def decompose(G):
    λ, V = la.eig(G)
    V_inv = la.inv(V)
    return λ, V, V_inv


def occupancy(t, α, G, λ=None, V=None, V_inv=None):
    if λ is None:
        λ, V = la.eig(G)
    if V_inv is None:
        V_inv = la.inv(V)
    P_t_matrix = (V @ np.diag(np.exp(λ * t)) @ V_inv).real
    P_t = α @ P_t_matrix
    return P_t


def occupancy_vs_t(t, α, G, λ=None, V=None, V_inv=None):
    if λ is None:
        λ, V = la.eig(G)
    if V_inv is None:
        V_inv = la.inv(V)
    t = np.asarray(t)
    P_t = np.zeros((t.size, G.shape[0]))
    for i, ti in enumerate(t):
        P_ti = (V @ np.diag(np.exp(λ * ti)) @ V_inv).real
        P_t[i] = α @ P_ti
    return P_t
