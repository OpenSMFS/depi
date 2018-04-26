import numpy as np
from scipy.stats import expon


def get_ecdf(s, offset=0.5):
    """Return arrays (x, y) for the empirical CDF curve of sample `s`.

    Arguments:
        s (array of floats): sample
        offset (float, default 0.5): Offset to add to the y values of the CDF

    Returns:
        (x, y) (tuple of arrays): the x and y values of the empirical CDF
    """
    return np.sort(s), (np.arange(s.size) + offset) / s.size


def get_residuals(s, tau_fit, offset=0.5):
    """Returns residuals of sample `s` CDF vs an exponential CDF.

    Arguments:
        s (array of floats): sample
        tau_fit (float): mean waiting-time of the exponential distribution
            to use as reference
        offset (float): Default 0.5. Offset to add to the empirical CDF.
            See :func:`get_ecdf` for details.

    Returns:
        residuals (array): residuals of empirical CDF compared with analytical
        CDF with time constant `tau_fit`.
    """
    x, y = get_ecdf(s, offset=offset)
    ye = expon.cdf(x, scale=tau_fit)
    residuals = y - ye
    return x, residuals


def compute_score(residuals, x_residuals, metric):
    """Compute score to test if a ECDF is from a given distribution.

    Arguments:
        residuals (array): difference between the empirical CDF and the model
            CDF.
        x_residuals (array): values where the ECF is computed. Same size as
            `residuals`.
        metric (string): type of test statistics. Valid values are
            'KS' for Kolgomorov-Smirnov, 'CM' for Cramer von Mises.
    """
    if metric not in ['KS', 'CM']:
        raise ValueError(f"Invalid `metric` argument ({metric}).\n"
                          "Valid values are 'KS' or 'CM'.")
    error = None
    if metric == 'KS':
        error = np.abs(residuals).max() * 100
    elif metric == 'CM':
        error = np.trapz(residuals**2, x=x_residuals)
    return error


def get_score(samples, th):
    """Compute the exponentiality score of sample, for valuels > than `th`.
    """
    tail = samples[samples > th] - th
    tau_fit = tail.mean()
    x_residuals, residuals = get_residuals(tail, tau_fit)
    score = compute_score(residuals, x_residuals, 'CM')
    return score, tau_fit, tail.size


def calc_scores(values, th_list):
    """Compute exp score for all the threshold values in `th_list`.
    """
    scores = np.zeros_like(th_list, dtype=float)
    taus = np.zeros_like(th_list, dtype=float)
    for i, th in enumerate(th_list):
        scores[i], taus[i], _ = get_score(values, th)
    return scores, taus


def exp_from_hist(counts, bins, tau_th, tau_fit):
    """Compute an exponential curve fitting the tail of the input histogram.

    Arguments:
        counts (array): histograms counts
        bins (array): histogram bins, `bins` size is `counts.size + 1`.
        tau_th (float): threshold value where the exponential tail of the
            histogram starts.
        tau_fit (float): decay constant of the exponentail function.
    """
    t = bins[:-1] + 0.5 * (bins[1] - bins[0])
    i_max = np.nonzero(counts > 0)[0][-1]  # index of last non-zero element
    i_tau_th = np.searchsorted(bins, tau_th)
    y_fit = np.exp(- t / tau_fit)
    y_fit *= counts[i_tau_th:].sum() / y_fit[i_tau_th:].sum()
    return t, y_fit
