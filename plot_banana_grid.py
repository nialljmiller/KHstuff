'''
plot_banana_grid.py

Publication-quality multi-panel grid of all valid RGB banana curves.
One panel per star, sorted by observed [M/H].

Output: results/bananas/banana_grid.png
        results/bananas/banana_grid.pdf
'''

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import AutoMinorLocator
import warnings

warnings.filterwarnings('ignore')

# ── ApJ plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':         'serif',
    'font.size':           8,
    'axes.labelsize':      8,
    'xtick.labelsize':     7,
    'ytick.labelsize':     7,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.facecolor':   'white',
})

# ── Load chains ───────────────────────────────────────────────────────────────
chain_dir   = 'results/bananas/chains'
chain_files = sorted(f for f in os.listdir(chain_dir) if f.endswith('.pkl'))

def get_age_col(df):
    for c in ['age', 'Age(Gyr)']:
        if c in df.columns:
            return c
    return None

def classify_star(logg):
    return 'RGB' if (float(logg) < 2.2 or float(logg) > 3.0) else 'clump'

def median_banana(feh, age, n_bins=25):
    bins = np.linspace(-1.0, 0.4, n_bins + 1)
    mids, meds, lo, hi = [], [], [], []
    for j in range(len(bins) - 1):
        m = (feh >= bins[j]) & (feh < bins[j+1])
        if m.sum() >= 15:
            ab = age[m]
            mids.append((bins[j] + bins[j+1]) / 2)
            meds.append(np.median(ab))
            lo.append(np.percentile(ab, 16))
            hi.append(np.percentile(ab, 84))
    return np.array(mids), np.array(meds), np.array(lo), np.array(hi)

# ── Collect valid RGB stars ───────────────────────────────────────────────────
TEFF_MAX = 5500.0
MIN_BINS = 5

stars = []
for fname in chain_files:
    with open(os.path.join(chain_dir, fname), 'rb') as f:
        res = pickle.load(f)

    teff_obs      = float(res['teff_obs'])
    logg_obs      = float(res['logg_obs'])
    stellar_class = res.get('stellar_class') or classify_star(logg_obs)

    if stellar_class != 'RGB' or teff_obs >= TEFF_MAX:
        continue

    output  = res.get('output')
    if output is None:
        continue
    age_col = get_age_col(output)
    if age_col is None:
        continue

    feh_all = output['initial_met'].values
    age_all = output[age_col].values
    ok      = np.isfinite(feh_all) & np.isfinite(age_all) & (age_all > 0)
    feh, age = feh_all[ok], age_all[ok]

    if len(feh) < 100:
        continue

    feh_mids, age_meds, age_lo, age_hi = median_banana(feh, age)
    if len(feh_mids) < MIN_BINS:
        continue

    mh_obs = float(res['mh_obs'])
    w = 0.15
    mask_obs = (feh >= mh_obs - w) & (feh <= mh_obs + w)
    age_at_obs    = np.median(age[mask_obs])    if mask_obs.sum() >= 10 else np.nan
    age_at_obs_lo = np.percentile(age[mask_obs], 16) if mask_obs.sum() >= 10 else np.nan
    age_at_obs_hi = np.percentile(age[mask_obs], 84) if mask_obs.sum() >= 10 else np.nan

    stars.append({
        'star_id':       res['star_id'],
        'teff_obs':      teff_obs,
        'logg_obs':      logg_obs,
        'lum_obs':       float(res['lum_obs']),
        'mh_obs':        mh_obs,
        'feh':           feh,
        'age':           age,
        'feh_mids':      feh_mids,
        'age_meds':      age_meds,
        'age_lo':        age_lo,
        'age_hi':        age_hi,
        'age_at_obs':    age_at_obs,
        'age_at_obs_lo': age_at_obs_lo,
        'age_at_obs_hi': age_at_obs_hi,
    })

stars.sort(key=lambda s: s['mh_obs'])
N = len(stars)
print(f"{N} valid RGB stars for grid plot.")

# ── Layout ────────────────────────────────────────────────────────────────────
NCOLS = 5
NROWS = int(np.ceil(N / NCOLS))

fig = plt.figure(figsize=(NCOLS * 3.0, NROWS * 2.8))
fig.patch.set_facecolor('white')

outer_gs = gridspec.GridSpec(
    NROWS, NCOLS, figure=fig,
    hspace=0.60, wspace=0.38,
    left=0.06, right=0.88, top=0.97, bottom=0.06
)

vmin_mh = -1.1
vmax_mh =  0.45
cmap_mh = plt.cm.RdYlBu_r

for idx, s in enumerate(stars):
    row, col = divmod(idx, NCOLS)
    ax = fig.add_subplot(outer_gs[row, col])

    c = cmap_mh((s['mh_obs'] - vmin_mh) / (vmax_mh - vmin_mh))

    # Dynamic y ceiling: 99th percentile of samples
    age_ceil = np.nanpercentile(s['age'], 99) * 1.08

    if len(s['feh_mids']) >= 3:
        ax.fill_between(s['feh_mids'], s['age_lo'], s['age_hi'],
                        color=c, alpha=0.25, linewidth=0)
        ax.plot(s['feh_mids'], s['age_meds'], '-', color=c, lw=1.6)

    ax.axvline(s['mh_obs'], color='k', lw=0.9, ls='--', alpha=0.7)

    if np.isfinite(s['age_at_obs']):
        ax.errorbar(
            s['mh_obs'], s['age_at_obs'],
            yerr=[[s['age_at_obs'] - s['age_at_obs_lo']],
                  [s['age_at_obs_hi'] - s['age_at_obs']]],
            fmt='o', color='k', ms=3.5, lw=1.0, capsize=1.5, zorder=5
        )

    ax.set_xlim(-1.05, 0.45)
    ax.set_ylim(0, age_ceil)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelsize=6)

    ax.set_title(
        f"{s['star_id']}\n"
        f"$T$={s['teff_obs']:.0f} K  "
        f"$\\log g$={s['logg_obs']:.1f}  "
        f"[M/H]={s['mh_obs']:.2f}",
        fontsize=5.0, pad=2
    )

    if np.isfinite(s['age_at_obs']):
        ax.text(0.97, 0.97,
                f"${s['age_at_obs']:.1f}"
                f"^{{+{s['age_at_obs_hi']-s['age_at_obs']:.1f}}}"
                f"_{{-{s['age_at_obs']-s['age_at_obs_lo']:.1f}}}$ Gyr",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=6.0, color='k',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.8))

    if col == 0:
        ax.set_ylabel('Age (Gyr)', fontsize=7)
    if row == NROWS - 1 or idx == N - 1:
        ax.set_xlabel('[Fe/H] assumed', fontsize=7)

# Hide unused panels
for idx in range(N, NROWS * NCOLS):
    row, col = divmod(idx, NCOLS)
    fig.add_subplot(outer_gs[row, col]).set_visible(False)

# Colourbar
cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.78])
sm = ScalarMappable(cmap=cmap_mh, norm=Normalize(vmin=vmin_mh, vmax=vmax_mh))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('obs [M/H]', fontsize=9)
cb.ax.tick_params(labelsize=7)

os.makedirs('results/bananas', exist_ok=True)
fig.savefig('results/bananas/banana_grid.png')
fig.savefig('results/bananas/banana_grid.pdf')
plt.close(fig)
print(f"Saved results/bananas/banana_grid.png and .pdf  ({N} panels)")
