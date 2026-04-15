'''
replot_bananas.py

Post-processes existing MCMC banana chains from results/bananas/chains/.
Filters to physically meaningful ages (< age of the universe),
regenerates per-star banana plots in Figure 8 style, and regenerates
the summary plot.

Run this without re-running MCMC:
    python replot_bananas.py
'''

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

os.makedirs('results/bananas/plots', exist_ok=True)

# ── Physical constants ────────────────────────────────────────────────────────
AGE_UNIVERSE = 13.8   # Gyr — hard physical upper limit

# ── Load chains ───────────────────────────────────────────────────────────────
chain_dir   = 'results/bananas/chains'
chain_files = sorted(f for f in os.listdir(chain_dir) if f.endswith('.pkl'))
print(f"Found {len(chain_files)} chain files.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_age_col(df):
    for c in ['age', 'Age(Gyr)']:
        if c in df.columns:
            return c
    return None

def median_banana(feh_samples, age_samples, feh_min=-1.0, feh_max=0.4, n_bins=30):
    '''Bin samples and compute median + 16/84th percentile age at each [Fe/H].'''
    bins = np.linspace(feh_min, feh_max, n_bins + 1)
    mids, meds, lo, hi, counts = [], [], [], [], []
    for j in range(len(bins) - 1):
        mask = (feh_samples >= bins[j]) & (feh_samples < bins[j+1])
        if mask.sum() >= 10:
            ages_bin = age_samples[mask]
            mids.append((bins[j] + bins[j+1]) / 2)
            meds.append(np.median(ages_bin))
            lo.append(np.percentile(ages_bin, 16))
            hi.append(np.percentile(ages_bin, 84))
            counts.append(mask.sum())
    return (np.array(mids), np.array(meds),
            np.array(lo), np.array(hi), np.array(counts))

# ── Per-star Figure 8-style banana plot ──────────────────────────────────────
def plot_banana_fig8(res, age_col):
    star_id       = res['star_id']
    stellar_class = res.get('stellar_class', 'unknown')
    teff_obs      = res['teff_obs']
    lum_obs       = res['lum_obs']
    logg_obs      = res['logg_obs']
    mh_obs        = res['mh_obs']
    orig_loss     = res.get('orig_loss', np.nan)
    safe_id       = star_id.replace('/', '_')

    output = res['output']

    feh_all = output['initial_met'].values
    age_all = output[age_col].values

    # Apply physical age filter
    physical = np.isfinite(feh_all) & np.isfinite(age_all) & (age_all <= AGE_UNIVERSE) & (age_all >= 0)
    feh  = feh_all[physical]
    age  = age_all[physical]

    n_total    = len(feh_all)
    n_physical = physical.sum()
    pct_physical = 100 * n_physical / max(n_total, 1)

    # Compute median banana curve
    feh_mids, age_meds, age_lo, age_hi, counts = median_banana(feh, age)

    # ── Figure: two panels, mimicking Figure 8 of the paper ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={'width_ratios': [2, 1]})

    class_color = {'RGB': 'steelblue', 'clump': 'seagreen', 'unknown': 'grey'}
    c = class_color.get(stellar_class, 'grey')

    # ── Left: banana (age vs assumed [Fe/H]) — the degeneracy map ────────────
    ax = axes[0]

    # 2D density of physical samples
    if n_physical > 50:
        h = ax.hexbin(feh, age, gridsize=25, cmap='YlOrRd',
                      mincnt=1, linewidths=0.2, alpha=0.6)
        plt.colorbar(h, ax=ax, label='Sample count', pad=0.02)

    # Median banana curve with 1σ band
    if len(feh_mids) >= 3:
        ax.fill_between(feh_mids, age_lo, age_hi,
                        color=c, alpha=0.25, label='16–84th pct')
        ax.plot(feh_mids, age_meds, '-', color=c, lw=2.5,
                label='Median banana')

    # Physical limit line
    ax.axhline(AGE_UNIVERSE, color='dimgrey', lw=1.2, ls=':',
               alpha=0.7, label=f'Age of universe ({AGE_UNIVERSE} Gyr)')

    # Observed metallicity
    ax.axvline(mh_obs, color='k', lw=1.5, ls='--',
               label=f'obs [M/H] = {mh_obs:.2f}')

    ax.set_xlabel('[Fe/H] Assumed', fontsize=12)
    ax.set_ylabel('Age Inferred (Gyr)', fontsize=12)
    ax.set_title('Age–[Fe/H] Degeneracy Map', fontsize=12)
    ax.set_xlim(-1.05, 0.45)
    ax.set_ylim(0, AGE_UNIVERSE + 0.5)
    ax.legend(fontsize=8, loc='upper right')

    # ── Right: marginal age distribution at observed [Fe/H] ──────────────────
    # This is a preview of what step 4 will produce properly with the MDF
    ax2 = axes[1]
    feh_window = 0.15   # ±0.15 dex around observed metallicity
    mask_obs = (feh >= mh_obs - feh_window) & (feh <= mh_obs + feh_window)

    if mask_obs.sum() >= 10:
        ages_at_obs = age[mask_obs]
        ax2.hist(ages_at_obs, bins=20, orientation='horizontal',
                 color=c, alpha=0.8, edgecolor='white')
        med_age = np.median(ages_at_obs)
        ax2.axhline(med_age, color='k', lw=2,
                    label=f'Median: {med_age:.1f} Gyr')
        ax2.axhline(np.percentile(ages_at_obs, 16),
                    color='k', lw=1, ls='--')
        ax2.axhline(np.percentile(ages_at_obs, 84),
                    color='k', lw=1, ls='--')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Insufficient\nsamples near\nobs [M/H]',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=9)

    ax2.set_xlabel('N samples', fontsize=12)
    ax2.set_ylabel('Age Inferred (Gyr)', fontsize=12)
    ax2.set_title(f'Age at obs [M/H]±{feh_window}', fontsize=11)
    ax2.set_ylim(0, AGE_UNIVERSE + 0.5)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()

    fig.suptitle(
        f"{star_id}  [{stellar_class}]\n"
        f"Teff={teff_obs:.0f} K   logg={logg_obs:.2f}   lum={lum_obs:.2f}   "
        f"obs[M/H]={mh_obs:.2f}   "
        f"Physical samples: {n_physical:,}/{n_total:,} ({pct_physical:.0f}%)",
        fontsize=10, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(f'results/bananas/plots/{safe_id}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

# ── Main loop ─────────────────────────────────────────────────────────────────
all_bananas = {}   # for summary plot

for fname in chain_files:
    with open(os.path.join(chain_dir, fname), 'rb') as f:
        res = pickle.load(f)

    star_id = res['star_id']
    output  = res.get('output')
    if output is None:
        print(f"  {star_id}: no output DataFrame, skipping")
        continue

    age_col = get_age_col(output)
    if age_col is None:
        print(f"  {star_id}: no age column found, skipping")
        continue

    # Apply physical filter
    feh = output['initial_met'].values
    age = output[age_col].values
    physical = np.isfinite(feh) & np.isfinite(age) & (age <= AGE_UNIVERSE) & (age >= 0)
    n_physical = physical.sum()
    pct = 100 * n_physical / max(len(feh), 1)

    stellar_class = res.get('stellar_class', 'unknown')
    mh_obs        = res.get('mh_obs', np.nan)

    print(f"  {star_id}  [{stellar_class}]  physical={n_physical:,} ({pct:.0f}%)")

    if n_physical < 50:
        print(f"    → too few physical samples, skipping plot")
        continue

    all_bananas[star_id] = {
        'feh':     feh[physical],
        'age':     age[physical],
        'mh_obs':  mh_obs,
        'stellar_class': stellar_class,
    }

    plot_banana_fig8(res, age_col)
    print(f"    → plot saved")

print(f"\n{len(all_bananas)} stars with valid physical bananas.\n")

# ── Summary plot: Figure 8-style overview, RGB vs clump ──────────────────────
print("Making summary plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle(
    "Platinum Sample — Age–[Fe/H] Degeneracy Maps (MCMC, age < 13.8 Gyr)\n"
    "Left: RGB (logg < 2.2 or > 3.0)   Right: Clump (2.2 ≤ logg ≤ 3.0)",
    fontsize=12, fontweight='bold'
)

mh_vals = np.array([v['mh_obs'] for v in all_bananas.values()])
vmin    = np.nanpercentile(mh_vals, 5)
vmax    = np.nanpercentile(mh_vals, 95)
cmap    = plt.cm.plasma

for ax, target_class in zip(axes, ['RGB', 'clump']):
    ax.set_title(target_class, fontsize=13)
    n_plotted = 0

    for sid, data in all_bananas.items():
        if data['stellar_class'] != target_class:
            continue

        color = cmap((data['mh_obs'] - vmin) / max(vmax - vmin, 1e-6))

        feh_mids, age_meds, age_lo, age_hi, _ = median_banana(
            data['feh'], data['age'])

        if len(feh_mids) < 3:
            continue

        ax.fill_between(feh_mids, age_lo, age_hi,
                        color=color, alpha=0.12)
        ax.plot(feh_mids, age_meds, '-', color=color, lw=1.5, alpha=0.85)
        n_plotted += 1

    ax.axhline(AGE_UNIVERSE, color='dimgrey', lw=1.2, ls=':',
               alpha=0.6, label=f'Age of universe ({AGE_UNIVERSE} Gyr)')
    ax.set_xlabel('[Fe/H] Assumed', fontsize=12)
    ax.set_ylabel('Age Inferred (Gyr)', fontsize=12)
    ax.set_xlim(-1.05, 0.45)
    ax.set_ylim(0, AGE_UNIVERSE + 0.5)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.97, f'N={n_plotted} stars',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
fig.colorbar(sm, ax=axes, label='obs [M/H]', shrink=0.8)

fig.savefig('results/bananas/banana_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Summary plot saved to results/bananas/banana_summary.png")
print("Done.")
