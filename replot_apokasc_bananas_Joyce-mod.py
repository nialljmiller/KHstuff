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

CHAIN_DIR = '/home/mjoyce/Jamie/age_posteriors/make_my_own_figures/' #'results/apokasc/chains'
PLOTS_DIR = '/home/mjoyce/Jamie/age_posteriors/make_my_own_figures/' #'results/apokasc/plots'


BESTFIT_FEH_WINDOW = 0.01



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

    #  Main banana: median relation with 16th–84th percentile band 
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

    #  Inferred point at obs [M/H] 
    _feh_mask = (feh >= mh_obs - BESTFIT_FEH_WINDOW) & (feh <= mh_obs + BESTFIT_FEH_WINDOW)
    _inf_med = age_med  # fallback to global median if window is empty
    _age_at_obs = age[_feh_mask]
    _inf_med = float(np.median(_age_at_obs))
    _inf_lo  = float(np.percentile(_age_at_obs, 16))
    _inf_hi  = float(np.percentile(_age_at_obs, 84))


    _age_at_obs = age[_feh_mask]

    # KDE over the marginal age posterior at obs [M/H]
    _kde = gaussian_kde(_age_at_obs, bw_method='scott')
    _age_grid = np.linspace(y_lo, y_hi, 500)
    _kde_vals = _kde(_age_grid)

    _inf_map = float(_age_grid[np.argmax(_kde_vals)])   # MAP
    _inf_lo  = float(np.percentile(_age_at_obs, 16))
    _inf_hi  = float(np.percentile(_age_at_obs, 84))
    _inf_med = _inf_map  # use MAP as the reported point estimate

    ax_main.errorbar(
        mh_obs, _inf_med,
        yerr=[[_inf_med - _inf_lo], [_inf_hi - _inf_med]],
        fmt='o', color='steelblue', ms=7, lw=1.8,
        capsize=4, zorder=5, label='Inferred',
    )

    #  APOKASC reference point 
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

    #  Top marginal: p([Fe/H]) 
    x_grid = np.linspace(x_lo, x_hi, 400)
    pdf = None

    kde = gaussian_kde(feh)
    pdf = kde(x_grid)

    ax_top.fill_between(x_grid, 0.0, pdf, color='lightsteelblue', alpha=0.85)
    ax_top.plot(x_grid, pdf, color=c, lw=2.0)

    ax_top.set_ylabel(r'$p([\mathrm{Fe/H}])$', fontsize=11)
    plt.setp(ax_top.get_xticklabels(), visible=False)


    #HERE!
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
    )

    positive_counts = age_counts[age_counts > 0]
    x_max = float(positive_counts.max())
    ax_hist.set_xlim(0.0, x_max * 1.08)

    ax_hist.axhline(_inf_med, color='steelblue', lw=1.5, ls='--', zorder=2)
    ax_hist.axhline(aux_value, color='k', lw=1.5, ls='-', zorder=2)

    ax_hist.set_xlabel(r'$N$ samples', fontsize=11)
    ax_hist.set_ylabel(f'Age at [M/H] = {mh_obs:.2f} (Gyr)', fontsize=11)
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position('right')

    n_eff = mask.sum()
    fig.subplots_adjust(top=0.93)


    #fig.savefig(os.path.join(out_dir, f'{safe_id}.pdf'), dpi=130, bbox_inches='tight')

    '''fig.suptitle(
        f"{star_id}  [{stellar_class}]   N_samples={n_eff:,}",
        fontsize=10, fontweight='bold', y=0.99,
    )'''



    # #  Top-right info box only for image. 
    # e_teff_str = f"{e_teff:.0f} K"      if np.isfinite(e_teff)   else "—"
    # mass_str   = f"{int_mass:.3f} Msun" if np.isfinite(int_mass) else "—"
    # acc_str    = f"{acc:.3f}"           if np.isfinite(acc)       else "—"
    # if np.isfinite(aux_value) and np.isfinite(e_int_age_hi) and np.isfinite(e_int_age_lo):
    #     intage_str = f"{aux_value:.2f} +{e_int_age_hi:.2f} / -{e_int_age_lo:.2f} Gyr"
    # elif np.isfinite(aux_value):
    #     intage_str = f"{aux_value:.2f} Gyr"
    # else:
    #     intage_str = "—"

    # info_lines = [
    #     f"ID:       {star_id}",
    #     f"Class:    {stellar_class}",
    #     f"Teff:     {teff_obs:.0f} ± {e_teff_str}",
    #     f"Logg:     {logg_obs:.3f} cgs",
    #     f"Lum:      {lum_obs:.3f} log L/Lsun",
    #     f"[Fe/H]:   {mh_obs:.3f}",
    #     f"[a/Fe]:   {alpha_fe:.2f}",
    #     f"Mass:     {mass_str}",
    #     f"APOKASC AGE:   {intage_str}",
    #     f"Acc frac: {acc_str}",
    # ]
    # ax_empty.text(
    #     0.98, 0.95, '\n'.join(info_lines),
    #     transform=ax_empty.transAxes,
    #     ha='right', va='top',
    #     fontsize=8.5, family='monospace',
    #     bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
    #               edgecolor='#aaaaaa', alpha=0.9),
    # )

    # fig.suptitle(
    #     f"{star_id}  [{stellar_class}]   N_samples={n_eff:,}",
    #     fontsize=10, fontweight='bold', y=0.99,)

    png_path = os.path.join(out_dir, f'{safe_id}_revised.png')
    fig.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return png_path




#  Load one chain pkl and call save_banana_plot 
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

    if len(sys.argv) > 1:
        # Single star by ID
        star_id = sys.argv[1]
        safe_id = star_id.replace('/', '_')
        pkl_path = os.path.join(CHAIN_DIR, f'{safe_id}.pkl')
        png = plot_from_pkl(pkl_path)
        print(f"Saved: {png}")

    else:
        # All chains
        chain_files = sorted(f for f in os.listdir(CHAIN_DIR) if f.endswith('.pkl'))
        print(f"Replotting {len(chain_files)} chains...")
        for i, fname in enumerate(chain_files):
            pkl_path = os.path.join(CHAIN_DIR, fname)
            png = plot_from_pkl(pkl_path)
            star_id = fname.replace('.pkl', '')
            status = png if png else 'SKIPPED'
            print(f"  [{i+1}/{len(chain_files)}] {star_id}  →  {status}")
        print("Done.")
