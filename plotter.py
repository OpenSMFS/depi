from collections.abc import Sequence
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import depi
import dist_distrib as dd
import ctmc


def mpl_text_1state(params, E_μ, show_τ_relax=True, space='nm', time='s'):
    text = rf'R ~ OU{{μ={params["R_mean"][0]:.1f} {space}, σ={params["R_sigma"][0]:.1f} {space}'
    if show_τ_relax:
        text += ',\n' + rf'$\qquad\qquad\;\tau_{{relax}}$={params["τ_relax"][0]:,} {time}'
    text += ('}\n'
             rf'$R_0$ = {params["R0"]} {space},  $\langle E \rangle$ = {E_μ:.2f}' + '\n'
             rf'$\tau_D$ = {params["τ_D"]} {time},  $\tau_A$ = {params["τ_A"]} {time}' + '\n'
             rf'$\delta t$ = {params["δt"]:,} {time}' + '\n'
             )
    return text


def plot_text_1state(ax, x, y, params, show_τ_relax=True, **text_kws):
    p = params
    E_μ = depi.mean_E_from_gauss_PoR(p['R_mean'][0], p['R_sigma'][0], p['R0'])
    text_kws_used = dict(transform=ax.transAxes, va='top')
    text_kws_used.update(**text_kws)
    text = mpl_text_1state(params, E_μ, show_τ_relax=show_τ_relax)
    ax.text(x, y, text, **text_kws_used)


def plot_R_distrib(params, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    name = params.copy().pop('name').lower()
    if name.startswith('gauss'):
        print('Gaussian model', flush=True)
        func = plot_R_distrib_gauss
    elif name.startswith('wlc'):
        print('WLC model', flush=True)
        func = plot_R_distrib_wlc
    else:
        raise ValueError(f'Distance model "{name}" not recognized.')
    ax = func(params, ax)
    ax.axvline(params['R0'], color='r', ls='--')
    return ax


def plot_R_distrib_gauss(params, ax):
    R_mean, R_sigma = params['R_mean'], params['R_sigma']
    num_states = np.size(params['R_mean'])
    if num_states == 1:
        r = np.arange(0, R_mean + 5 * R_sigma, R_sigma / 20)
        y = dd.gaussian(r, R_mean, R_sigma)
    else:
        k_s = np.array(params['k_s'])
        if k_s.ndim == 1:
            fract = np.array([k_s[1] / (k_s.sum()), k_s[0] / (k_s.sum())])
        else:
            fract = ctmc.CT_stationary_distribution(np.asarray(params['k_s']))
        r = np.arange(0, np.max(R_mean) + 5 * np.max(R_sigma),
                      np.min(R_sigma) / 20)
        y = np.zeros_like(r)
        for i in range(num_states):
            y += fract[i] * dd.gaussian(r, R_mean[i], R_sigma[i])
    ax.plot(r, y, 'k')
    return ax


def plot_R_distrib_wlc(params, ax):
    L, lp, offset = params['L'], params['lp'], params['offset']
    num_states = np.size(L)
    if num_states == 1:
        r = np.arange(0, L + offset, L / 100)
        y = dd.wormlike_chain(r, L, lp, offset=offset)
        y /= np.trapz(y, r)
    else:
        k_s = np.array(params['k_s'])
        if k_s.ndim == 1:
            fract = np.array([k_s[1] / (k_s.sum()), k_s[0] / (k_s.sum())])
        else:
            fract = ctmc.CT_stationary_distribution(np.asarray(params['k_s']))
        r = np.arange(0, np.max(L) + np.max(offset) , np.min(L) / 100)
        y = np.zeros_like(r)
        for i in range(num_states):
            yi = dd.wormlike_chain(r, L[i], lp[i], offset=offset[i])
            y += fract[i] * yi / np.trapz(yi, r)
    ax.plot(r, y, 'k')
    return ax


def plot_E_sim(burstsph_sim, params, E=None, ax=None, legend=True):
    if ax is None:
        _, ax = plt.subplots()
    if E is None:
        E = depi.calc_E_burst(burstsph_sim)
    bins = np.arange(-0.1, 1.1, 0.02)
    R0 = params['R0']
    E_ph = depi.E_from_dist(burstsph_sim.R_ph, R0)
    E_burst = E_ph.groupby('burst').mean()
    ax.hist([], histtype='step', lw=1.2, color='k',
            label='E per ph (no bursts, y-axis A.U.)')
    ax.hist(E, bins=bins, alpha=0.8, label='E per burst (w/ shot-noise)')
    ax2 = ax.twinx()
    ax2.hist(E_ph, histtype='step', weights=np.repeat(0.5e-1, E_ph.size),
             lw=1.2, color='k', bins=100)
    ax.hist(E_burst, bins=bins, alpha=0.6, color='C1',
            label='Burst avg. of E from R @ t_emission')
    if params['name'].lower().startswith('gauss'):
        R_mean = np.atleast_1d(params['R_mean'])
        R_sigma = np.atleast_1d(params['R_sigma'])
        for i, (R_m, R_s) in enumerate(zip(R_mean, R_sigma)):
            E_μ = depi.mean_E_from_gauss_PoR(R_m, R_s, R0)
            ax.axvline(E_μ, ls='--', color='k', label=f'state{i}')
    #ax.set_xlabel('E')
    if legend:
        ax.legend(loc=(0.9, 0.0), frameon=False, ncol=1)
        #ax2.legend(loc=(1, 0), frameon=False);
    sns.despine(ax=ax, trim=True)
    sns.despine(ax=ax2, left=True, trim=True)
    ax2.yaxis.set_visible(False)
    return ax


def plot_nanotimes(burstsph, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    nt_d = burstsph.loc[burstsph.stream == 'DexDem', 'nanotime']
    nt_a = burstsph.loc[burstsph.stream == 'DexAem', 'nanotime']
    kws = dict(bins=np.arange(0, 50, 0.2), log=True, histtype='step', lw=1.3)
    ax.hist(nt_d, color='C2', label='D', **kws)
    ax.hist(nt_a, color='C3', label='A', **kws)
    sns.despine(ax=ax)
    return ax


def plot_nanotime_tau_rel_series(relax_dict, cmap='Spectral', axes=None):
    colors = sns.color_palette(cmap, len(relax_dict))
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(8, 8))
    for color, (τ_relax, p) in zip(cycle(colors), relax_dict.items()):
        burstsph = p['bph']
        nt_d = burstsph.loc[burstsph.stream == 'DexDem', 'nanotime']
        nt_a = burstsph.loc[burstsph.stream == 'DexAem', 'nanotime']
        kws = dict(label=f'{τ_relax:,} ns', bins=np.arange(0, 50, 0.2),
                   log=True, histtype='step', lw=1.3, color=color)
        axes[0].hist(nt_d, **kws)
        axes[1].hist(nt_a, **kws)
    for a, label in zip(axes, ['Donor', 'Acceptor']):
        sns.despine(ax=a)
        a.set_xlabel('ns')
        a.set_title(label, y=0.9, va='top')
        a.set_ylim(0.5)
    return axes, colors


def plot_E_tau_rel_series(relax_dict, cmap=None, axes=None, colors=None,
                          legend=True, **kws):
    if colors is None:
        if cmap is not None:
            colors = sns.color_palette(cmap, len(relax_dict))
        else:
            colors = [f'C{i}' for i in range(10)]
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True)
    bins = np.arange(-0.1, 1.1, 0.0333)
    for color, (τ_relax, p) in zip(cycle(colors), relax_dict.items()):
        bph = p['bph']
        params = p['params']
        R0 = params['R0']
        E = depi.calc_E_burst(bph)
        E_μ = depi.mean_E_from_gauss_PoR(params['R_mean'][0], params['R_sigma'][0], R0)
        #E_mode = depi.E_from_dist(params['R_mean'][0], R0)
        E_ph = depi.E_from_dist(bph.R_ph, R0)
        #E_burst = E_ph.groupby('burst').mean()
        hist_kws = dict(histtype='stepfilled', color=color,
                        label=f'{τ_relax:,} ns', bins=bins)
        hist_kws.update(kws)
        axes[0].hist(E_ph, **hist_kws)
        axes[1].hist(E_ph.groupby('burst').mean(), **hist_kws)
        axes[2].hist(E, **hist_kws)
    labels = ['E from R, per ph', 'E from R, burst average', 'E burst (incl. shot noise)']
    for a, t in zip(axes, labels):
        sns.despine(ax=a)
        a.set_title(t)
        a.axvline(E_μ, color='k', ls='-', alpha=0.3)
        #a.axvline(E_mode, color='k', ls=':')
    if legend:
        axes[-1].legend(bbox_to_anchor=(1.1, 1), loc='upper right',
                        title=r'$\tau_{relax}$', frameon=True)
    return axes, colors


def plot_colorbar(relax_dict, colors, vertical=False, ax=None):
    if all(c in [f'C{i}' for i in range(10)] for c in colors):
        #idx_c  = [int(c[1]) for c in colors]
        hex_c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [hex_c[i] for i in idx_i]
    if isinstance(colors[0], str):
        colors = [mpl.colors.hex2color(c) for c in colors]
    if vertical:
        if ax is None:
            ax = plt.axes([1, 0.25, 0.02, 0.5])
        ax.yaxis.tick_right()
        ax.imshow(np.asarray(colors)[:, np.newaxis, :], aspect=3)
        ax.xaxis.set_visible(False)
        ax.set_yticks(range(len(relax_dict)))
        ax.set_yticklabels(['%g' % r for r in relax_dict])
    else:
        if ax is None:
            ax = plt.axes([0.25, 0, 0.5, 0.02])
        ax.imshow(np.asarray(colors)[np.newaxis, ...], aspect=0.3)
        ax.yaxis.set_visible(False)
        ax.set_xticks(range(len(relax_dict)))
        ax.set_xticklabels(['%g' % r for r in relax_dict])
    ax.set_title(r'$\tau_{relax}$ (ns)')
    return ax


def plot_E_R_pdf(params, ax=None, which='ER', **kws):
    assert len(which) <= 2, '`which` can only be "E", "R" or "ER"'
    R0 = params['R0']
    R_mean = params['R_mean']
    R_sigma = params['R_sigma']
    R_samples = np.random.randn(500000) * R_sigma + R_mean
    E_samples = depi.E_from_dist(R_samples, R0)
    style = dict(color='k', lw=2, histtype='step', density=True)
    style.update(kws)
    if ax is None:
        _, ax = plt.subplots(1, len(which), figsize=(6 * len(which), 4))
    if isinstance(ax, Sequence) or hasattr(ax, '__array__'):
        axR, axE = ax
        which = 'ER'
    else:
        assert len(which) == 1, 'One axis passed but two chars in `which`'
        if which == 'E':
            axE = ax
        elif which == 'R':
            axR = ax
    if 'R' in which:
        axR.hist(R_samples, bins=200, **style)
    if 'E' in which:
        axE.hist(E_samples, bins=np.arange(0, 1.01, 0.01), **style)
        axE.set_ylim(0)
    return ax
