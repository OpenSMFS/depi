import numpy as np
from scipy.special import erf
import ctmc
from IPython.display import Math, HTML, display


valid_model_names = ('gaussian', 'wlc', 'gaussian_chain', 'radial_gaussian')


def assert_valid_model_name(name):
    name = name.lower()
    if name not in valid_model_names:
        raise TypeError(f'Distance model name "{name}" not recognized.'
                        f'Valid model names are:\n {valid_model_names})')


def distribution(params):
    name = params['name'].lower()
    assert_valid_model_name(name)
    model = {'gaussian': Gaussian, 'radial_gaussian': RadialGaussian,
             'gaussian_chain': GaussianChain, 'wlc': WormLikeChain}
    return model[name](params)


class BaseDistribution:
    # Redefine these attributes in sub-classes:
    param_names = tuple()
    _pdf_latex = (r'P(r) = {\rm\; to \; be \; implemented} \\')

    def __init__(self, params):
        self.name = params['name']
        self.params = {k: np.asfarray(v) for k, v in params.items()
                       if k in self.param_names}
        self.num_states = self._get_num_states()
        if self.num_states > 1:
            self.k_s = k_s = np.asfarray(params['k_s'])
            if k_s.ndim == 1:
                self.fract = np.array([k_s[1] / (k_s.sum()), k_s[0] / (k_s.sum())])
            else:
                self.fract = ctmc.CT_stationary_distribution(k_s)

    def _get_num_states(self):
        for k, v in self.params.items():
            num_states = np.size(v)
            break
        return num_states

    def get_r_axis(self, dr, n=5):
        r_max = self.r_max(n=n)
        start = np.min(self.params.get('offset', 0))
        return np.arange(start, r_max + dr, dr)

    def get_pdf(self, dr=0.01, n=5):
        r = self.get_r_axis(dr, n=n)
        if self.num_states == 1:
            y = self.pdf(r, **self.params)
        else:
            y = np.zeros_like(r)
            for i in range(self.num_states):
                p = {k: v[i] for k, v in self.params.items()}
                y += self.fract[i] * self.pdf(r, **p)
        return r, y

    def _ipython_display_(self):
        state = '1-state' if self.num_states == 1 else f'{self.num_states}-states'
        display(HTML(f'<h3>Distance Model: "{self.name} {state}" </h3>'),
                Math(self._latex_pdf), HTML(self._html_params()))


class Gaussian(BaseDistribution):
    param_names = ('R_mean', 'R_sigma')
    _latex_pdf = (r'P(r) = \frac{1}{\sqrt{2 \pi \sigma^2}}'
                  r'\exp\left(-\frac{(r - \mu)^2}{2\sigma^2}\right) \\')

    @staticmethod
    def pdf(r, R_mean, R_sigma):
        u = (r - R_mean) / R_sigma
        area = R_sigma * np.sqrt(2 * np.pi)
        return np.exp(-(u**2) / 2) / area

    def r_max(self, n=5):
        return np.max(self.params['R_mean']) + n * np.max(self.params['R_sigma'])

    def _html_params(self):
        p = self.params
        return (f'<i>μ</i> = <code>R_mean</code> = {p["R_mean"]} <br>'
                f'<i>σ</i> = <code>R_sigma</code> = {p["R_sigma"]}')


class RadialGaussian(BaseDistribution):
    param_names = ('sigma', 'offset', 'mu')
    _latex_pdf = (r'P(r) = a^{-1}\; (r - r_0)^2 \;'
                  r'\exp\left[-\frac{[(r - r_0)- \mu]^2}{2\sigma^2}\right] \\'
                  r'a = \sqrt{\frac{\pi}{2}}\sigma(\sigma^2 + \mu^2) '
                  r'    + \mu \sigma^2 exp\left(-\frac{\mu^2}{2\sigma^2} \right)\\')

    @staticmethod
    def pdf(r, sigma, offset, mu):
        sigma_sq = sigma * sigma
        mu_sq = mu * mu
        area = (np.sqrt(np.pi / 2) * sigma * (sigma_sq + mu_sq)
                * (1 - erf(-mu / (np.sqrt(2) * sigma)))
                + mu * sigma_sq * np.exp(- 0.5 * mu_sq / sigma_sq))
        r0 = (r - offset)
        valid = r0 > 0
        res = np.zeros_like(r)
        res[~valid] = 0
        r0_squared = r0[valid]**2
        r0_minus_mu_squared = (r0[valid] - mu)**2
        res[valid] = r0_squared * np.exp(-r0_minus_mu_squared / (2 * sigma_sq)) / area
        return res

    def _mean_peaks(self):
        mu, sigma = self.params['mu'], self.params['sigma']
        return (np.sqrt(np.pi / 2) * mu * sigma * (mu**2 + 3 * sigma**2)
                * (1 - erf(-mu / (np.sqrt(2) * sigma)))
                + sigma**2 * (mu**2 + 2 * sigma**2) * np.exp(-0.5 * (mu / sigma)**2))

    def _std_dev_peaks(self):
        mu, sigma = self.params['mu'], self.params['sigma']
        M2 = (np.sqrt(np.pi / 2) * sigma * (mu**4 + 6 * mu**2 * sigma**2 + 3 * sigma**4)
              * (1 - erf(-mu / (np.sqrt(2) * sigma)))
              + sigma**2 * (mu**3 + 5 * mu * sigma**2))
        return M2 - self._mean_peaks()**2

    def r_max(self, n=5):
        return 20  # np.max(self._mean_peaks()) + n * np.max(self._std_dev_peaks())

    def _html_params(self):
        p = self.params
        return (f'<i>μ</i> = <code>mu</code> = {p["mu"]} <br>'
                f'<i>r₀</i> = <code>offset</code> = {p["offset"]} <br>'
                f'<i>σ</i> = <code>sigma</code> = {p["sigma"]}')


class GaussianChain(BaseDistribution):
    param_names = ('sigma', 'offset')
    _latex_pdf = (r'P(r) = a^{-1}\; (r - r_0)^2 \;'
                  r'\exp\left[-\frac{(r - r_0)^2}{2\sigma^2}\right] \\'
                  r'a = \sqrt{\frac{\pi}{2}}\sigma^3 \\')

    @staticmethod
    def pdf(r, sigma, offset):
        return RadialGaussian.pdf(r, sigma, offset, mu=0)

    def r_max(self, n=5):
        std_dev = np.max(self.params['sigma'])**2 * (3 * np.pi - 8) / np.pi
        return np.max(self.params['offset']) + n * std_dev

    def _html_params(self):
        p = self.params
        return (f'<i>r₀</i> = <code>offset</code> = {p["offset"]} <br>'
                f'<i>σ</i> = <code>sigma</code> = {p["sigma"]}')


class WormLikeChain(BaseDistribution):
    param_names = ('L', 'lp', 'offset')
    _latex_pdf = (r'P(r) = c\,r^2 \, '
                  r'\left[ 1 - \left(\frac{r}{L}\right)^2 \right]^{-\frac{9}{2}} \,'
                  r'\exp\left[-\frac{9}{8}\frac{L}{l_p}'
                  r'\left( 1 - \left(\frac{r}{L}\right)^2 \right)^{-1} \right] \\')

    @staticmethod
    def pdf(r, L, lp, offset):
        res = np.zeros_like(r)
        ro = r - offset
        valid = (ro < L) * (ro > 0)
        res[~valid] = 0
        ro_valid = ro[valid]
        F = 1 - (ro_valid / L)**2
        res[valid] = np.exp(- (9 / 8) * (L / lp) / F) * ro_valid**2 / F**(9 / 2)
        area = np.trapz(res, r)
        return res / area

    def r_max(self, n=5):
        return np.max(self.params['L']) + np.max(self.params['offset']) + n / 10

    def _html_params(self):
        p = self.params
        return (f'<i>L</i> = <code>L</code> = {p["L"]} <br>'
                f'<i>lp</i> = <code>lp</code> = {p["lp"]} <br>'
                f'<i>r₀</i> = <code>offset</code> = {p["offset"]}')


def _get_norm_cdf(du, u_max):
    temp = np.arange(0, u_max + du, du)
    u = np.hstack([-temp[::-1][:-1], temp])
    assert len(u) % 2 == 1, 'Size of `u` has to be an odd number.'
    idx_offset = (u.size - 1) // 2
    norm_pdf = Gaussian.pdf(u, R_mean=0, R_sigma=1)
    norm_cdf = np.cumsum(norm_pdf) * du
    return norm_cdf, idx_offset


def get_r_dist_distrib(du, u_max, dr, dd_params):
    """Computes R axis and index offset for mapping a distance distribution.

    Arguments:
        du (float): step-size for the x-axis of the unitary Gaussian PDF
        u_max (float): max range of x-axis for the unitary Gaussian PDF.
            The PFD is evaluated between -u_max and u_max.
        dr (float): step-size of the R axis of the new distance distribution.
            `dr` should smaller than `du`, for example `dr = 0.1 * du`.
        dd_params (dict): parameters of the new distance distribution.
            `dd_params['name']` must be a string with the name of the
            distance distribution (i.e. 'WLC' or 'radial_gaussian').
            The other elements need to match the parameter names of the
            chosen distance distribution.

    Returns:
        - r_dist (array): array of distances with same CDF values as
          a unitary Normal CDF evaluated on an array `u` going from
          `-u_max` to `u_max` with step `du`.
        - idx_offset (int): size of the positive side of the `u` array,
          not including 0.

    Note:
        Given a unitary (sigma = 1) normal random number `x`,
        a random number `R` from an arbitrary distribution can
        be obtained as::

            ix = int(round(x / du)) + idx_offset
            R = r_dist[ix]

        where `r_dist` and `idx_offset` are returned by this function,
        and `du` is the input argument to this function.
    """
    # CDF of the standard normal distribution with sigma = 1
    norm_cdf, idx_offset = _get_norm_cdf(du, u_max)
    # PDF and CDF of the new distance distribution
    r, dd_pdf = distribution(dd_params).get_pdf(dr)
    dd_cdf = np.cumsum(dd_pdf) * dr
    assert dd_cdf[-1] <= 1 + 1e-10, 'CDF is larger than 1!'
    assert 1 - dd_cdf[-1] < 1e-3, 'CDF axis range too small. '
    # Build the R-axis at positions matching the normal CDF values
    r_dist = np.interp(norm_cdf, dd_cdf, r)
    return r_dist, idx_offset
