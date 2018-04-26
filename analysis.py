from collections import defaultdict
import numpy as np
import pandas as pd
import numba


def E_from_dist(x, R0):
    return 1 / (1 + (x / R0)**6)


def bva(burstsph_Dex, n):
    """Compute BVA quantity for each burst.

    Arguments:
        burstsph_Dex (pandas.DataFrame): dataframe with one row per photon
            and a column `stream` indicating the stream of each photon.
            The index need to have two levels (burst, ph) indicating the
            burst and photon ID respectively.
        n (int): the number of photon used to compute E in sub-bursts
    """
    E_sub_std = []
    DexAem = burstsph_Dex.stream == 'DexAem'
    for i in np.unique(burstsph_Dex.index.get_level_values('burst')):
        DexAem_burst = DexAem.loc[i].values
        E_sub_bursts = []
        for istart in range(0, DexAem_burst.shape[0], n):
            A_D = DexAem_burst[istart:istart + n].sum()
            E = A_D / n
            E_sub_bursts.append(E)
        E_sub_std.append(np.std(E_sub_bursts))
    E_sub_std = np.array(E_sub_std)
    return E_sub_std


def bva_bin(burstsph_Dex, n, num_sub_bursts_th=60,
            E_bins=np.arange(0, 1.01, 0.05)):
    """Compute the mean BVA on E bins.

    Arguments:
        burstsph_Dex (pandas.DataFrame): dataframe with one row per photon
            and a column `stream` indicating the stream of each photon.
            The index need to have two levels (burst, ph) indicating the
            burst and photon ID respectively.
        n (int): the number of photon used to compute E in sub-bursts
        num_sub_bursts_th (int): minimum number of sub-bursts in a bin
            for BVA to be reported.
        E_bins (array): array of E bin edges, size in number of bins + 1.

    Returns:
        2-tuple of two arrays:
        - `E_sub_std_mean_bin`: the mean BVA in each E bins
        - `E_centers`: bin centers for each value in `E_sub_std_mean_bin`.
    """

    E_centers = E_bins[:-1] + 0.5 * (E_bins[1] - E_bins[0])
    burstsph_Dex_grp = burstsph_Dex.groupby('burst')
    Nd_plus_Na = burstsph_Dex_grp['stream'].count()
    Na = burstsph_Dex_grp['stream'].agg(lambda x: (x == 'DexAem').sum())
    E_raw = Na / Nd_plus_Na
    E_raw_bins = pd.cut(E_raw, E_bins, labels=np.arange(E_bins.size - 1),
                        include_lowest=True)
    E_sub_bursts_bin = defaultdict(list)
    DexAem = burstsph_Dex.stream == 'DexAem'
    for i in np.unique(burstsph_Dex.index.get_level_values('burst')):
        DexAem_burst = DexAem.loc[i].values
        E_raw_bin = E_raw_bins.loc[i]
        for istart in range(0, DexAem_burst.shape[0], n):
            A_D = DexAem_burst[istart:istart + n].sum()
            E = A_D / n
            E_sub_bursts_bin[E_raw_bin].append(E)
    E_sub_std_mean_bin = [np.std(E) if len(E) > num_sub_bursts_th else -1
                          for i, E in sorted(E_sub_bursts_bin.items())]
    assert len(E_sub_std_mean_bin) == len(E_bins) - 1
    return E_sub_std_mean_bin, E_centers


def bva_bin_combo(burstsph_Dex, n, num_sub_bursts_th=60,
                  E_bins=np.arange(0, 1.01, 0.05)):
    """Compute the BVA per each burst and mean BVA on E bins.

    This function combines :func:`bva` and :func:`bva_bin` and is much
    faster than calling these two functions separately.

    Arguments:
        burstsph_Dex (pandas.DataFrame): dataframe with one row per photon
            and a column `stream` indicating the stream of each photon.
            The index need to have two levels (burst, ph) indicating the
            burst and photon ID respectively.
        n (int): the number of photon used to compute E in sub-bursts
        num_sub_bursts_th (int): minimum number of sub-bursts in a bin
            for BVA to be reported.
        E_bins (array): array of E bin edges, size in number of bins + 1.

    Returns:
        Tuple of 3 arrays:
        - `E_sub_std`: the BVA computed on each burst
        - `E_sub_std_mean_bin`: the mean BVA in each E bins
        - `E_centers`: bin centers for each value in `E_sub_std_mean_bin`.
    """

    E_centers = E_bins[:-1] + 0.5 * (E_bins[1] - E_bins[0])

    burstsph_Dex_grp = burstsph_Dex.groupby('burst')
    Nd_plus_Na = burstsph_Dex_grp['stream'].count()
    Na = burstsph_Dex_grp['stream'].agg(lambda x: sum(x == 'DexAem'))
    E_raw = Na / Nd_plus_Na
    E_raw_bins = pd.cut(E_raw, E_bins, labels=np.arange(E_bins.size - 1),
                        include_lowest=True)

    # standard deviation of sub-bursts in each burst
    E_sub_std = []
    # dict with bin index as key and values that are a list of lists.
    # The innermost list contains the E values of sub-bursts in a burst.
    E_sub_bursts_bin = defaultdict(list)
    DexAem = burstsph_Dex.stream == 'DexAem'
    for i in np.unique(burstsph_Dex.index.get_level_values('burst')):
        DexAem_burst = DexAem.loc[i].values
        E_raw_bin = E_raw_bins.loc[i]
        # We initialize a list of sub-bursts for each burst
        E_sub_bursts = []
        for istart in range(0, DexAem_burst.shape[0], n):
            A_D = DexAem_burst[istart:istart + n].sum()
            E = A_D / n
            E_sub_bursts.append(E)
        # Append list of current burst as an element in the list of current bin
        E_sub_bursts_bin[E_raw_bin].append(E_sub_bursts)
        E_sub_std.append(np.std(E_sub_bursts))
    E_bin_len = {k: sum([len(x) for x in v])
                 for k, v in E_sub_bursts_bin.items()}
    E_sub_std_mean_bin = [np.std(np.hstack(E_lists))
                          if E_bin_len[i] > num_sub_bursts_th else -1
                          for i, E_lists in sorted(E_sub_bursts_bin.items())]
    assert len(E_sub_std_mean_bin) == len(E_bins) - 1
    E_sub_std = np.array(E_sub_std)
    return E_sub_std, E_sub_std_mean_bin, E_centers


@numba.jit(nopython=True)
def ou_process_core_numba(X, N, delta_t, ou_tau, ou_sigma):
    """Low-level function with computes the OU process
    """
    X[0] = N[0] * ou_sigma
    for i in range(1, X.size):
        dt = delta_t[i - 1]
        dt_over_tau = dt / ou_tau
        relax = np.exp(-dt_over_tau)
        diffuse = ou_sigma * np.sqrt(1 - relax**2)
        X[i] = X[i - 1] * relax + diffuse * N[i]


def ou_process(delta_t, N, ou_mean, ou_sigma, ou_tau):
    """Compute an Ornstein–Uhlenbeck (OU) process from a series of delta_t.

    Arguments:
        delta_t (array): intervals between time points where the OU process
            will be evaluated
        N (array): array of white Gaussian noise with sigma=1 of size
            size(delta_t) + 1.
        ou_mean (float): mean of the OU process
        ou_sigma (float): standard deviation of the relaxed OU process
        ou_tau (float): relaxation time of the OU process

    Returns:
        Array of OU process values. Output size is `size(delta_t) + 1`
    """
    X = np.zeros(delta_t.size + 1, dtype=float)
    ou_process_core_numba(X, N, delta_t, ou_tau, ou_sigma)
    X += ou_mean
    return X


def pda_gauss_R_static(burst_size_series, R0, R_mean, τ_relax=0,
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


def pda_gauss_R_relaxed(burst_size_series, R0, R_mean, τ_relax=0,
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


def pda_from_E(E_ph, oversample=100, dithering_sigma=0.02):
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
