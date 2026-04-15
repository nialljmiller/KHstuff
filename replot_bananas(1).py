'''
replot_bananas.py

Post-processes existing MCMC banana chains. Applies science-target selection
(RGB stars, Teff < 5500 K, age < 13.8 Gyr) and produces:

Per-star plots (results/bananas/plots/<star_id>.png):
    Figure 8-style: banana + marginal age at obs [M/H]

Summary plots (results/bananas/):
    banana_summary.png          — all RGB bananas overlaid, coloured by obs [M/H]
    hr_diagram.png              — HR diagram coloured by banana quality
    kiel_diagram.png            — Kiel diagram coloured by banana quality
    banana_width.png            — age uncertainty (84th-16th pct) vs [M/H] and logg
    individual/<star_id>.png    — each banana at full size, standalone

Science selection:
    stellar_class == 'RGB'  (logg < 2.2 or logg > 3.0)
    teff_obs < 5500 K
    age < 13.8 Gyr (age of universe)
'''

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import warnings

warnings.filterwarnings('ignore')

os.makedirs('results/bananas/plots',      exist_ok=True)
os.makedirs('results/bananas/individual', exist_ok=True)

# ── Science selection cuts ────────────────────────────────────────────────────
AGE_UNIVERSE  = 13.8   # Gyr
TEFF_MAX      = 5500.0 # K  — exclude hot non-RGB outliers

# ── Load and filter chains ────────────────────────────────────────────────────
chain_dir   = 'results/bananas/chains'
chain_files = sorted(f for f in os.listdir(chain_dir) if f.endswith('.pkl'))
print(f"Found {len(chain_files)} chains.\n")

def get_age_col(df):
    for c in ['age', 'Age(Gyr)']:
        if c in df.columns:
            return c
    return None

def classify_star(logg):
    if np.isnan(float(logg)):
        return 'unknown'
    return 'RGB' if (float(logg) < 2.2 or float(logg) > 3.0) else 'clump'

def median_banana(feh, age, feh_min=-1.0, feh_max=0.4, n_bins=30):
    bins = np.linspace(feh_min, feh_max, n_bins + 1)
    mids, meds, lo, hi = [], [], [], []
    for j in range(len(bins) - 1):
        m = (feh >= bins[j]) & (feh < bins[j+1])
        if m.sum() >= 10:
            ab = age[m]
            mids.append((bins[j] + bins[j+1]) / 2)
            meds.append(np.median(ab))
            lo.append(np.percentile(ab, 16))
            hi.append(np.percentile(ab, 84))
    return (np.array(mids), np.array(meds),
            np.array(lo),   np.array(hi))

# ── Collect valid stars ───────────────────────────────────────────────────────
stars = {}   # star_id → dict with metadata + filtered samples

for fname in chain_files:
    with open(os.path.join(chain_dir, fname), 'rb') as f:
        res = pickle.load(f)

    star_id       = res['star_id']
    teff_obs      = float(res['teff_obs'])
    logg_obs      = float(res['logg_obs'])
    lum_obs       = float(res['lum_obs'])
    mh_obs        = float(res['mh_obs'])
    orig_loss     = float(res.get('orig_loss', np.nan))
    stellar_class = res.get('stellar_class') or classify_star(logg_obs)

    # ── Science selection ─────────────────────────────────────────────────────
    if stellar_class != 'RGB':
        print(f"  SKIP {star_id}: not RGB (class={stellar_class})")
        continue
    if teff_obs >= TEFF_MAX:
        print(f"  SKIP {star_id}: Teff={teff_obs:.0f} K >= {TEFF_MAX:.0f} K")
        continue

    output  = res.get('output')
    if output is None:
        print(f"  SKIP {star_id}: no output DataFrame")
        continue
    age_col = get_age_col(output)
    if age_col is None:
        print(f"  SKIP {star_id}: no age column")
        continue

    feh_all = output['initial_met'].values
    age_all = output[age_col].values
    ok      = np.isfinite(feh_all) & np.isfinite(age_all) & (age_all > 0) & (age_all <= AGE_UNIVERSE)
    feh     = feh_all[ok]
    age     = age_all[ok]

    if len(feh) < 100:
        print(f"  SKIP {star_id}: only {len(feh)} physical samples")
        continue

    print(f"  OK   {star_id}  Teff={teff_obs:.0f}K  logg={logg_obs:.2f}  "
          f"[M/H]={mh_obs:.2f}  N_phys={len(feh):,}")

    feh_mids, age_meds, age_lo, age_hi = median_banana(feh, age)
    age_width = age_hi - age_lo   # 1σ width of banana at each [Fe/H] bin

    # Banana quality: fraction of samples that are physical
    pct_physical = 100 * ok.sum() / max(len(feh_all), 1)

    # Age at observed metallicity (±0.15 dex window)
    w = 0.15
    mask_obs = (feh >= mh_obs - w) & (feh <= mh_obs + w)
    age_at_obs_med = np.median(age[mask_obs]) if mask_obs.sum() >= 10 else np.nan
    age_at_obs_lo  = np.percentile(age[mask_obs], 16) if mask_obs.sum() >= 10 else np.nan
    age_at_obs_hi  = np.percentile(age[mask_obs], 84) if mask_obs.sum() >= 10 else np.nan

    stars[star_id] = {
        'star_id':       star_id,
        'teff_obs':      teff_obs,
        'logg_obs':      logg_obs,
        'lum_obs':       lum_obs,
        'mh_obs':        mh_obs,
        'orig_loss':     orig_loss,
        'feh':           feh,
        'age':           age,
        'feh_mids':      feh_mids,
        'age_meds':      age_meds,
        'age_lo':        age_lo,
        'age_hi':        age_hi,
        'age_width':     age_width,
        'pct_physical':  pct_physical,
        'age_at_obs_med': age_at_obs_med,
        'age_at_obs_lo':  age_at_obs_lo,
        'age_at_obs_hi':  age_at_obs_hi,
        'n_physical':    len(feh),
    }

print(f"\n{len(stars)} RGB stars with Teff < {TEFF_MAX:.0f} K and valid physical bananas.\n")

if len(stars) == 0:
    print("No stars passed selection. Exiting.")
    exit(0)

# ── Colour scale for [M/H] ───────────────────────────────────────────────────
mh_vals = np.array([s['mh_obs'] for s in stars.values()])
vmin    = np.nanpercentile(mh_vals, 5)
vmax    = np.nanpercentile(mh_vals, 95)
cmap_mh = plt.cm.plasma

def mh_color(mh):
    return cmap_mh((mh - vmin) / max(vmax - vmin, 1e-6))

# ── 1. Per-star Figure 8-style plots ─────────────────────────────────────────
print("Plotting per-star banana plots...")
for sid, s in stars.items():
    safe_id = sid.replace('/', '_')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={'width_ratios': [2, 1]})
    c = mh_color(s['mh_obs'])

    ax = axes[0]
    if len(s['feh']) > 50:
        h = ax.hexbin(s['feh'], s['age'], gridsize=25, cmap='YlOrRd',
                      mincnt=1, linewidths=0.2, alpha=0.5)
        plt.colorbar(h, ax=ax, label='Sample count', pad=0.02)
    if len(s['feh_mids']) >= 3:
        ax.fill_between(s['feh_mids'], s['age_lo'], s['age_hi'],
                        color=c, alpha=0.3, label='16–84th pct')
        ax.plot(s['feh_mids'], s['age_meds'], '-', color=c, lw=2.5,
                label='Median')
    ax.axhline(AGE_UNIVERSE, color='dimgrey', lw=1.2, ls=':',
               label=f'Age of universe')
    ax.axvline(s['mh_obs'], color='k', lw=1.5, ls='--',
               label=f"obs [M/H]={s['mh_obs']:.2f}")
    ax.set_xlabel('[Fe/H] Assumed', fontsize=12)
    ax.set_ylabel('Age Inferred (Gyr)', fontsize=12)
    ax.set_title('Age–[Fe/H] Degeneracy Map', fontsize=12)
    ax.set_xlim(-1.05, 0.45)
    ax.set_ylim(0, AGE_UNIVERSE + 0.5)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    w = 0.15
    mask = (s['feh'] >= s['mh_obs'] - w) & (s['feh'] <= s['mh_obs'] + w)
    if mask.sum() >= 10:
        ax2.hist(s['age'][mask], bins=20, orientation='horizontal',
                 color=c, alpha=0.85, edgecolor='white')
        ax2.axhline(s['age_at_obs_med'], color='k', lw=2,
                    label=f"Median: {s['age_at_obs_med']:.1f} Gyr")
        ax2.axhline(s['age_at_obs_lo'], color='k', lw=1, ls='--')
        ax2.axhline(s['age_at_obs_hi'], color='k', lw=1, ls='--')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Insufficient\nsamples', ha='center',
                 va='center', transform=ax2.transAxes, fontsize=9)
    ax2.set_xlabel('N samples', fontsize=12)
    ax2.set_ylabel('Age Inferred (Gyr)', fontsize=12)
    ax2.set_title(f'Age at obs [M/H]±{w} dex', fontsize=11)
    ax2.set_ylim(0, AGE_UNIVERSE + 0.5)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()

    fig.suptitle(
        f"{sid}\n"
        f"Teff={s['teff_obs']:.0f} K   logg={s['logg_obs']:.2f}   "
        f"lum={s['lum_obs']:.2f}   obs[M/H]={s['mh_obs']:.2f}   "
        f"physical={s['pct_physical']:.0f}%",
        fontsize=10, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(f"results/bananas/plots/{safe_id}.png", dpi=120, bbox_inches='tight')
    plt.close(fig)

print("  Done.\n")

# ── 2. Summary banana plot — all RGB stars overlaid ───────────────────────────
print("Plotting banana summary...")
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle(
    f"RGB Platinum Sample — Age–[Fe/H] Degeneracy Maps (MCMC)\n"
    f"N={len(stars)} stars  |  Teff < {TEFF_MAX:.0f} K  |  logg < 2.2 or > 3.0  |  age < {AGE_UNIVERSE} Gyr",
    fontsize=12, fontweight='bold'
)
for s in stars.values():
    if len(s['feh_mids']) < 3:
        continue
    c = mh_color(s['mh_obs'])
    ax.fill_between(s['feh_mids'], s['age_lo'], s['age_hi'],
                    color=c, alpha=0.10)
    ax.plot(s['feh_mids'], s['age_meds'], '-', color=c, lw=1.5, alpha=0.85)

ax.axhline(AGE_UNIVERSE, color='dimgrey', lw=1.2, ls=':',
           label=f'Age of universe ({AGE_UNIVERSE} Gyr)')
ax.set_xlabel('[Fe/H] Assumed', fontsize=13)
ax.set_ylabel('Age Inferred (Gyr)', fontsize=13)
ax.set_xlim(-1.05, 0.45)
ax.set_ylim(0, AGE_UNIVERSE + 0.5)
ax.legend(fontsize=10)
sm = plt.cm.ScalarMappable(cmap=cmap_mh, norm=Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='obs [M/H]')
fig.savefig('results/bananas/banana_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: results/bananas/banana_summary.png\n")

# ── 3. HR diagram coloured by banana quality ──────────────────────────────────
print("Plotting HR diagram...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, color_key, label, cm in [
    (axes[0], 'pct_physical', 'Physical sample %', 'viridis'),
    (axes[1], 'age_at_obs_med', 'Age at obs [M/H] (Gyr)', 'plasma'),
]:
    vals = np.array([s[color_key] for s in stars.values()])
    teffs = np.array([s['teff_obs'] for s in stars.values()])
    lums  = np.array([s['lum_obs']  for s in stars.values()])
    finite = np.isfinite(vals)

    sc = ax.scatter(teffs[finite], lums[finite], c=vals[finite],
                    cmap=cm, s=80, zorder=3, edgecolors='k', lw=0.5)
    plt.colorbar(sc, ax=ax, label=label)
    ax.invert_xaxis()
    ax.set_xlabel('Teff (K)', fontsize=12)
    ax.set_ylabel('log(L/L☉)', fontsize=12)
    ax.set_title(f'H-R Diagram — colour = {label}', fontsize=11)

fig.suptitle('RGB Platinum Sample — HR Diagram', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/bananas/hr_diagram.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: results/bananas/hr_diagram.png\n")

# ── 4. Kiel diagram coloured by banana quality ────────────────────────────────
print("Plotting Kiel diagram...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, color_key, label, cm in [
    (axes[0], 'pct_physical', 'Physical sample %', 'viridis'),
    (axes[1], 'age_at_obs_med', 'Age at obs [M/H] (Gyr)', 'plasma'),
]:
    vals  = np.array([s[color_key] for s in stars.values()])
    teffs = np.array([s['teff_obs'] for s in stars.values()])
    loggs = np.array([s['logg_obs'] for s in stars.values()])
    finite = np.isfinite(vals)

    sc = ax.scatter(teffs[finite], loggs[finite], c=vals[finite],
                    cmap=cm, s=80, zorder=3, edgecolors='k', lw=0.5)
    plt.colorbar(sc, ax=ax, label=label)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('Teff (K)', fontsize=12)
    ax.set_ylabel('log g', fontsize=12)
    ax.set_title(f'Kiel Diagram — colour = {label}', fontsize=11)

fig.suptitle('RGB Platinum Sample — Kiel Diagram', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('results/bananas/kiel_diagram.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: results/bananas/kiel_diagram.png\n")

# ── 5. Banana width vs [Fe/H] and vs logg ────────────────────────────────────
print("Plotting banana width...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, x_key, xlabel in [
    (axes[0], 'mh_obs',  'obs [M/H]'),
    (axes[1], 'logg_obs', 'logg'),
]:
    for s in stars.values():
        if len(s['feh_mids']) < 3:
            continue
        c    = mh_color(s['mh_obs'])
        xval = s[x_key]
        # Mean banana width across all [Fe/H] bins
        mean_width = np.mean(s['age_width']) if len(s['age_width']) > 0 else np.nan
        if np.isfinite(mean_width):
            ax.scatter(xval, mean_width, color=c, s=60,
                       edgecolors='k', lw=0.5, zorder=3)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Mean banana width\n(84th–16th pct age, Gyr)', fontsize=12)
    ax.set_title(f'Banana Width vs {xlabel}', fontsize=11)
    ax.set_ylim(bottom=0)

fig.suptitle('RGB Platinum Sample — Age Uncertainty Width of Banana',
             fontsize=13, fontweight='bold')
sm = plt.cm.ScalarMappable(cmap=cmap_mh, norm=Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
fig.colorbar(sm, ax=axes, label='obs [M/H]', shrink=0.8)
fig.tight_layout()
fig.savefig('results/bananas/banana_width.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: results/bananas/banana_width.png\n")

# ── 6. Individual full-size banana per star ───────────────────────────────────
print("Plotting individual full-size bananas...")
for sid, s in stars.items():
    safe_id = sid.replace('/', '_')
    c = mh_color(s['mh_obs'])

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(s['feh']) > 50:
        h = ax.hexbin(s['feh'], s['age'], gridsize=30, cmap='YlOrRd',
                      mincnt=1, linewidths=0.2, alpha=0.5)
        plt.colorbar(h, ax=ax, label='Sample count')
    if len(s['feh_mids']) >= 3:
        ax.fill_between(s['feh_mids'], s['age_lo'], s['age_hi'],
                        color=c, alpha=0.3, label='16–84th pct')
        ax.plot(s['feh_mids'], s['age_meds'], '-', color=c, lw=2.5,
                label='Median banana')

    ax.axhline(AGE_UNIVERSE, color='dimgrey', lw=1.2, ls=':',
               label=f'Age of universe ({AGE_UNIVERSE} Gyr)')
    ax.axvline(s['mh_obs'], color='k', lw=1.5, ls='--',
               label=f"obs [M/H]={s['mh_obs']:.2f}")

    if not np.isnan(s['age_at_obs_med']):
        ax.axhline(s['age_at_obs_med'], color=c, lw=1.5, ls='-.',
                   label=f"Age at obs [M/H]: {s['age_at_obs_med']:.1f} Gyr")
        ax.axhspan(s['age_at_obs_lo'], s['age_at_obs_hi'],
                   color=c, alpha=0.12)

    ax.set_xlabel('[Fe/H] Assumed', fontsize=13)
    ax.set_ylabel('Age Inferred (Gyr)', fontsize=13)
    ax.set_xlim(-1.05, 0.45)
    ax.set_ylim(0, AGE_UNIVERSE + 0.5)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(
        f"{sid}\n"
        f"Teff={s['teff_obs']:.0f} K   logg={s['logg_obs']:.2f}   "
        f"obs[M/H]={s['mh_obs']:.2f}",
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(f"results/bananas/individual/{safe_id}.png",
                dpi=130, bbox_inches='tight')
    plt.close(fig)

print(f"  Saved {len(stars)} individual plots to results/bananas/individual/\n")

# ── Summary table ─────────────────────────────────────────────────────────────
rows = []
for s in stars.values():
    rows.append({
        'star_id':         s['star_id'],
        'teff_obs':        s['teff_obs'],
        'logg_obs':        s['logg_obs'],
        'lum_obs':         s['lum_obs'],
        'mh_obs':          s['mh_obs'],
        'n_physical':      s['n_physical'],
        'pct_physical':    s['pct_physical'],
        'age_at_obs_med':  s['age_at_obs_med'],
        'age_at_obs_lo':   s['age_at_obs_lo'],
        'age_at_obs_hi':   s['age_at_obs_hi'],
        'mean_banana_width': np.mean(s['age_width']) if len(s['age_width']) > 0 else np.nan,
    })
table = pd.DataFrame(rows).sort_values('mh_obs')
table.to_csv('results/bananas/rgb_banana_table.csv', index=False)
print(f"Summary table saved to results/bananas/rgb_banana_table.csv")
print("\nDone.")
