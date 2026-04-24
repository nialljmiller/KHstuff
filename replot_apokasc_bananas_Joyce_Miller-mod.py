#!/usr/bin/env python3
"""
replot_apokasc_bananas.py

Regenerates banana plots from existing chain .pkl files.
Does NO MCMC sampling.

Usage
-----
Remake all plots:
    python replot_apokasc_bananas.py

Remake a single star:
    python replot_apokasc_bananas.py "2M12345678+1234567"
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

CHAIN_DIR = 'results/apokasc/chains'            #/home/mjoyce/Jamie/age_posteriors/make_my_own_figures/'
PLOTS_DIR = 'results/apokasc/plots'            #/home/mjoyce/Jamie/age_posteriors/make_my_own_figures/'

BESTFIT_FEH_WINDOW = 0.01

# ── Stone-Martinez 2025: load full catalogue age distribution ─────────────────
# We load ALL StarFlow ages and build a KDE over them. This is plotted as a
# distribution on the right-hand histogram panel for shape comparison —
# no per-star cross-matching involved.
_sm_ages_all = None

def _load_starflow(path='StarFlow_summary_v1_0_0.fits'):
    global _sm_ages_all
    try:
        from astropy.io import fits
        with fits.open(path) as f:
            sf = f[1].data
        ages = np.array(sf['age'], dtype=float)
        _sm_ages_all = ages[np.isfinite(ages) & (ages > 0)]
        print(f"Loaded StarFlow: {len(_sm_ages_all)} stars with valid ages from {path}")
    except FileNotFoundError:
        print(f"WARNING: {path} not found — SM distribution will not be shown.")
    except Exception as e:
        print(f"WARNING: could not load StarFlow ({e}) — SM distribution will not be shown.")


def save_banana_plot(star_id, flat_samples, blobs_df,
                     teff_obs, lum_obs, logg_obs, mh_obs,
                     aux_value, stellar_class,
                     alpha_fe=0.0, int_mass=float('nan'), e_teff=float('nan'),
                     acc=float('nan'),
                     e_mh_obs=0.1, e_int_age_hi=float('nan'), e_int_age_lo=float('nan'),
                     out_dir=PLOTS_DIR):

    safe_id = star_id.replace('/', '_')
    os.makedirs(out_dir, exist_ok=True)

    class_color = {'RGB': 'steelblue', 'clump': 'seagreen', 'unknown': 'grey'}
    c = class_color.get(stellar_class, 'grey')

    age_col = 'age' if 'age' in blobs_df.columns else 'Age(Gyr)'

    feh = flat_samples['initial_met'].values
    age = blobs_df[age_col].values
    mask = np.isfinite(feh) & np.isfinite(age)
    feh, age = feh[mask], age[mask]

    age_med = np.median(age)

    feh_min, feh_max = np.nanmin(feh), np.nanmax(feh)
    feh_range = np.ptp(feh) if len(feh) > 1 else 0.0
    age_range = np.ptp(age) if len(age) > 1 else 0.0
    feh_pad = 0.04 * max(feh_range, 0.25)
    age_pad = 0.05 * max(age_range, 1.0)

    x_lo = feh_min - feh_pad
    x_hi = feh_max + feh_pad
    y_lo = max(0.0, np.nanmin(age) - age_pad)
    y_hi = np.nanmax(age) + age_pad

    fig = plt.figure(figsize=(12.5, 6.1))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4.8, 2.4],
        height_ratios=[1.5, 4.5],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top   = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_hist  = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis('off')

    # ── Main banana: median relation with 16th–84th percentile band ──────────
    n_feh_bins = int(np.clip(np.sqrt(len(feh)), 18, 36))
    feh_bins = np.linspace(feh_min, feh_max, n_feh_bins)
    med_ages, lo_ages, hi_ages, feh_mids = [], [], [], []
    for j in range(len(feh_bins) - 1):
        mask_bin = (feh >= feh_bins[j]) & (feh < feh_bins[j + 1])
        ages_bin = age[mask_bin]
        med_ages.append(np.median(ages_bin))
        lo_ages.append(np.percentile(ages_bin, 16))
        hi_ages.append(np.percentile(ages_bin, 84))
        feh_mids.append((feh_bins[j] + feh_bins[j + 1]) / 2.0)

    feh_mids = np.asarray(feh_mids)
    med_ages = np.asarray(med_ages)
    lo_ages  = np.asarray(lo_ages)
    hi_ages  = np.asarray(hi_ages)
    ax_main.fill_between(feh_mids, lo_ages, hi_ages, color=c, alpha=0.30, lw=0)
    ax_main.plot(feh_mids, med_ages, color=c, lw=2.2)

    obs_label = f'obs [M/H] = {mh_obs:.2f}'
    ax_main.axvline(mh_obs, color='0.25', lw=1.4, ls='--', label=obs_label)
    ax_top.axvline(mh_obs, color='0.25', lw=1.4, ls='--')

    # ── Inferred point at obs [M/H] via KDE MAP ───────────────────────────────
    _feh_mask = (feh >= mh_obs - BESTFIT_FEH_WINDOW) & (feh <= mh_obs + BESTFIT_FEH_WINDOW)
    _inf_med = age_med
    _age_at_obs = age[_feh_mask]
    _inf_med = float(np.median(_age_at_obs))
    _inf_lo  = float(np.percentile(_age_at_obs, 16))
    _inf_hi  = float(np.percentile(_age_at_obs, 84))

    _kde = gaussian_kde(_age_at_obs, bw_method='scott')
    _age_grid = np.linspace(y_lo, y_hi, 500)
    _kde_vals = _kde(_age_grid)

    _inf_map = float(_age_grid[np.argmax(_kde_vals)])
    _inf_lo  = float(np.percentile(_age_at_obs, 16))
    _inf_hi  = float(np.percentile(_age_at_obs, 84))
    _inf_med = _inf_map

    ax_main.errorbar(
        mh_obs, _inf_med,
        yerr=[[_inf_med - _inf_lo], [_inf_hi - _inf_med]],
        fmt='o', color='steelblue', ms=7, lw=1.8,
        capsize=4, zorder=5, label='Inferred',
    )

    # ── APOKASC reference point ───────────────────────────────────────────────
    if np.isfinite(aux_value):
        _has_yerr = np.isfinite(e_int_age_hi) and np.isfinite(e_int_age_lo)
        if _has_yerr:
            ax_main.errorbar(
                mh_obs, aux_value,
                yerr=[[e_int_age_lo], [e_int_age_hi]],
                fmt='*', color='k', ms=14, lw=1.8,
                capsize=4, zorder=5, label='APOKASC',
            )
        else:
            ax_main.plot(mh_obs, aux_value,
                         '*', color='k', ms=14, zorder=5, label='APOKASC')

    ax_main.set_xlabel('[Fe/H] assumed', fontsize=11)
    ax_main.set_ylabel('Age inferred (Gyr)', fontsize=11)
    ax_main.set_xlim(x_lo, x_hi)
    ax_main.set_ylim(y_lo, y_hi)
    ax_main.legend(loc='upper right', fontsize=8, frameon=True)

    # ── Top marginal: p([Fe/H]) ───────────────────────────────────────────────
    x_grid = np.linspace(x_lo, x_hi, 400)
    kde = gaussian_kde(feh)
    pdf = kde(x_grid)
    ax_top.fill_between(x_grid, 0.0, pdf, color='lightsteelblue', alpha=0.85)
    ax_top.plot(x_grid, pdf, color=c, lw=2.0)
    ax_top.set_ylabel(r'$p([\mathrm{Fe/H}])$', fontsize=11)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Right panel: age histogram at obs [M/H] + SM age distribution ────────
    age_for_hist = age[_feh_mask]

    n_age_bins = int(np.clip(np.sqrt(len(age_for_hist)) / 3.0, 18, 32))
    age_bins = np.linspace(y_lo, y_hi, n_age_bins + 1)
    age_counts, _, _ = ax_hist.hist(
        age_for_hist,
        bins=age_bins,
        orientation='horizontal',
        color='salmon',
        alpha=0.90,
        edgecolor='white',
        linewidth=0.6,
        label='This work',
    )

    positive_counts = age_counts[age_counts > 0]
    x_max = float(positive_counts.max()) if positive_counts.size else 1.0

    # ── SM age distribution as a scaled KDE on the same panel ────────────────
    # Scaled to match the peak of our histogram so the shapes are comparable.
    if _sm_ages_all is not None and len(_sm_ages_all) > 5:
        sm_in_range = _sm_ages_all[(_sm_ages_all >= y_lo) & (_sm_ages_all <= y_hi)]
        if len(sm_in_range) > 5:
            sm_kde = gaussian_kde(sm_in_range, bw_method='scott')
            sm_pdf = sm_kde(_age_grid)  # _age_grid runs y_lo to y_hi
            # Scale so the SM KDE peak matches our histogram peak
            if sm_pdf.max() > 0:
                sm_pdf_scaled = sm_pdf / sm_pdf.max() * x_max
                ax_hist.plot(sm_pdf_scaled, _age_grid,
                             color='tomato', lw=2.0, ls='-',
                             label='Stone-Martinez (2025)', zorder=4)

    ax_hist.set_xlim(0.0, x_max * 1.25)  # extra room for SM curve label
    ax_hist.axhline(_inf_med, color='steelblue', lw=1.5, ls='--', zorder=2)
    if np.isfinite(aux_value):
        ax_hist.axhline(aux_value, color='k', lw=1.5, ls='-', zorder=2)

    ax_hist.set_xlabel(r'$N$ samples / scaled density', fontsize=10)
    ax_hist.set_ylabel(f'Age at [M/H] = {mh_obs:.2f} (Gyr)', fontsize=11)
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position('right')
    ax_hist.legend(loc='upper right', fontsize=7, frameon=True)

    n_eff = mask.sum()
    fig.subplots_adjust(top=0.93)

    png_path = os.path.join(out_dir, f'{safe_id}_revised.png')
    fig.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return png_path


# ── Load one chain pkl and call save_banana_plot ──────────────────────────────
def plot_from_pkl(pkl_path, out_dir=PLOTS_DIR):

    with open(pkl_path, 'rb') as f:
        res = pickle.load(f)

    flat_samples = res.get('flat_samples', res.get('output'))
    blobs_df     = res.get('blobs_df',     res.get('output'))

    int_age   = float(res.get('int_age', np.nan))
    aux_value = int_age if np.isfinite(int_age) else np.nan

    return save_banana_plot(
        res['star_id'], flat_samples, blobs_df,
        res['teff_obs'], res['lum_obs'], res['logg_obs'], res['mh_obs'],
        aux_value, res['stellar_class'],
        alpha_fe=res.get('alpha_fe', 0.0),
        int_mass=res.get('int_mass', float('nan')),
        e_teff=res.get('e_teff', float('nan')),
        acc=res.get('acceptance_fraction', float('nan')),
        e_mh_obs=res.get('e_mh_obs', 0.1),
        e_int_age_hi=res.get('e_int_age_hi', float('nan')),
        e_int_age_lo=res.get('e_int_age_lo', float('nan')),
        out_dir=out_dir,
    )


if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    _load_starflow()

    if len(sys.argv) > 1:
        star_id = sys.argv[1]
        safe_id = star_id.replace('/', '_')
        pkl_path = os.path.join(CHAIN_DIR, f'{safe_id}.pkl')
        png = plot_from_pkl(pkl_path)
        print(f"Saved: {png}")

    else:
        chain_files = sorted(f for f in os.listdir(CHAIN_DIR) if f.endswith('.pkl'))
        print(f"Replotting {len(chain_files)} chains...")
        for i, fname in enumerate(chain_files):
            pkl_path = os.path.join(CHAIN_DIR, fname)
            png = plot_from_pkl(pkl_path)
            star_id = fname.replace('.pkl', '')
            status = png if png else 'SKIPPED'
            print(f"  [{i+1}/{len(chain_files)}] {star_id}  →  {status}")
        print("Done.")
