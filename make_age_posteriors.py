'''
make_age_posteriors.py  —  Step 4

For each valid RGB banana star, weight the MCMC samples by the Zoccali et al.
(2017) bulge MDF and produce an age posterior.

Inputs
------
zocalli.dat                    Zoccali et al. (2017). Only LRp0m1 rows used.
results/bananas/bananas.pkl

Outputs
-------
results/posteriors/
    <star_id>_posterior.png/.pdf
    age_metallicity_relation.png/.pdf
    age_posterior_table.csv
'''

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings('ignore')

os.makedirs('results/posteriors', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCIENCE SELECTION CUTS
# Set to None to disable a cut entirely.
# ══════════════════════════════════════════════════════════════════════════════

TEFF_MIN    = 3700     # K  — no lower Teff cut
TEFF_MAX    = 5500.0   # K  — exclude hot non-RGB outliers
LOGG_MIN    = 1.0      # dex — exclude near-TRGB (uninformative bananas)
LOGG_MAX    = None     # dex — no upper logg cut
MH_MIN      = None     # dex — no lower metallicity cut
MH_MAX      = None     # dex — no upper metallicity cut
LUM_MIN     = None     # log(L/Lsun) — no lower luminosity cut
LUM_MAX     = None     # log(L/Lsun) — no upper luminosity cut
AGE_MIN     = None     # Gyr — no lower age cut on samples
AGE_MAX     = 20       # Gyr — no upper age cut on samples
MIN_SAMPLES = 100      # minimum valid MCMC samples to process a star

# ══════════════════════════════════════════════════════════════════════════════

# ── ApJ plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':         'serif',
    'font.size':           10,
    'axes.labelsize':      11,
    'xtick.labelsize':     9,
    'ytick.labelsize':     9,
    'xtick.direction':     'in',
    'ytick.direction':     'in',
    'xtick.top':           True,
    'ytick.right':         True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.fontsize':     9,
    'legend.framealpha':   0.8,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.facecolor':   'white',
})

# ── 1. Load Zoccali MDF ───────────────────────────────────────────────────────
print("Loading Zoccali MDF...")
zoc_rows = []
with open('Zoccali_MDF.dat', 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) < 8:
            continue
        if not parts[0].startswith('LRp0m1'):
            continue
        try:
            zoc_rows.append(float(parts[7]))
        except ValueError:
            continue

zoc_feh = np.array(zoc_rows)
print(f"  N={len(zoc_feh)}  range [{zoc_feh.min():.2f}, {zoc_feh.max():.2f}]"
      f"  median={np.median(zoc_feh):.2f}\n")

mdf_kde  = gaussian_kde(zoc_feh, bw_method=0.15)
FEH_GRID = np.linspace(-1.1, 0.5, 200)

# ── 2. Load bananas ───────────────────────────────────────────────────────────
print("Loading bananas.pkl...")
with open('results/bananas/bananas.pkl', 'rb') as f:
    bananas = pickle.load(f)
print(f"  {len(bananas)} stars\n")

# ── Load Stone-Martinez 2025 ages ─────────────────────────────────────────────
from astropy.io import fits
print("Loading StarFlow summary...")
with fits.open('StarFlow_summary_v1_0_0.fits') as f:
    sf = f[1].data
sf_ids  = np.array([x.strip() for x in sf['sdss4_apogee_id']])
sf_age  = sf['age']
sf_ep   = sf['e_p_age']
sf_en   = sf['e_n_age']

def get_sm_age(star_id):
    '''Return (age, e_p, e_n) or (nan, nan, nan) if not in StarFlow.'''
    m = np.where(sf_ids == star_id.strip())[0]
    if len(m) == 0:
        return np.nan, np.nan, np.nan
    return float(sf_age[m[0]]), float(sf_ep[m[0]]), float(sf_en[m[0]])


def get_age_col(df):
    for c in ['age', 'Age(Gyr)']:
        if c in df.columns:
            return c
    return None

def classify_star(logg):
    return 'RGB' if (float(logg) < 2.2 or float(logg) > 3.0) else 'clump'

def passes_cuts(teff, logg, mh, lum):
    if TEFF_MIN  is not None and teff < TEFF_MIN:   return False
    if TEFF_MAX  is not None and teff >= TEFF_MAX:  return False
    if LOGG_MIN  is not None and logg < LOGG_MIN:   return False
    if LOGG_MAX  is not None and logg > LOGG_MAX:   return False
    if MH_MIN    is not None and mh   < MH_MIN:     return False
    if MH_MAX    is not None and mh   > MH_MAX:     return False
    if LUM_MIN   is not None and lum  < LUM_MIN:    return False
    if LUM_MAX   is not None and lum  > LUM_MAX:    return False
    return True

def median_banana(feh, age, n_bins=25):
    bins = np.linspace(-1.05, 0.45, n_bins + 1)
    mids, meds, lo, hi = [], [], [], []
    for j in range(len(bins) - 1):
        m = (feh >= bins[j]) & (feh < bins[j+1])
        if m.sum() >= 10:
            ab = age[m]
            mids.append((bins[j] + bins[j+1]) / 2)
            meds.append(np.median(ab))
            lo.append(np.percentile(ab, 16))
            hi.append(np.percentile(ab, 84))
    return np.array(mids), np.array(meds), np.array(lo), np.array(hi)

# ── 3. Compute posteriors ─────────────────────────────────────────────────────
results = []

for star_id, output in bananas.items():
    teff_obs = float(output['teff_obs'].iloc[0])
    logg_obs = float(output['logg_obs'].iloc[0])
    mh_obs   = float(output['mh_obs'].iloc[0])
    lum_obs  = float(output['lum_obs'].iloc[0])
    stellar_class = (output['stellar_class'].iloc[0]
                     if 'stellar_class' in output.columns
                     else classify_star(logg_obs))

    if stellar_class != 'RGB':
        continue
    if not passes_cuts(teff_obs, logg_obs, mh_obs, lum_obs):
        continue

    age_col = get_age_col(output)
    if age_col is None:
        continue

    feh_s = output['initial_met'].values
    age_s = output[age_col].values
    ok    = np.isfinite(feh_s) & np.isfinite(age_s) & (age_s > 0)
    if AGE_MIN is not None: ok &= (age_s >= AGE_MIN)
    if AGE_MAX is not None: ok &= (age_s <= AGE_MAX)
    feh = feh_s[ok]
    age = age_s[ok]

    if len(feh) < MIN_SAMPLES:
        continue

    weights = np.maximum(mdf_kde(feh), 0)
    if weights.sum() == 0:
        print(f"  SKIP {star_id}: zero MDF weight")
        continue
    weights /= weights.sum()

    post    = np.random.choice(age, size=10000, p=weights, replace=True)
    med_age = np.median(post)
    lo_age  = np.percentile(post, 16)
    hi_age  = np.percentile(post, 84)

    print(f"  {star_id}  [M/H]={mh_obs:.2f}  "
          f"{med_age:.1f} +{hi_age-med_age:.1f} -{med_age-lo_age:.1f} Gyr")

    results.append({
        'star_id':  star_id, 'teff_obs': teff_obs, 'logg_obs': logg_obs,
        'lum_obs':  lum_obs, 'mh_obs':   mh_obs,
        'age_median': med_age, 'age_lo': lo_age, 'age_hi': hi_age,
        'age_posterior': post, 'feh': feh, 'age': age, 'weights': weights,
    })

print(f"\n{len(results)} valid posteriors.\n")

# ── 4. Per-star Figure 8-style plots ─────────────────────────────────────────
print("Plotting per-star posteriors...")

cmap_mh = plt.cm.RdYlBu_r
vmin_mh = -1.1
vmax_mh =  0.45

for r in results:
    safe_id  = r['star_id'].replace('/', '_')
    c        = cmap_mh((r['mh_obs'] - vmin_mh) / (vmax_mh - vmin_mh))
    age_max  = np.nanpercentile(r['age'], 99) * 1.1
    age_bins = np.linspace(0, age_max, 30)

    fig = plt.figure(figsize=(10, 4.2))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1, 3],
                             width_ratios=[0.01, 2, 1],
                             hspace=0.06, wspace=0.40)

    # Top: MDF
    ax_mdf = fig.add_subplot(gs[0, 1])
    ax_mdf.fill_between(FEH_GRID, mdf_kde(FEH_GRID),
                         color='steelblue', alpha=0.35)
    ax_mdf.plot(FEH_GRID, mdf_kde(FEH_GRID), color='steelblue', lw=1.5)
    ax_mdf.axvline(r['mh_obs'], color='k', lw=1.0, ls='--', alpha=0.7)
    ax_mdf.set_xlim(-1.05, 0.45)
    ax_mdf.set_ylabel(r'$p$([Fe/H])', fontsize=9)
    ax_mdf.set_xticklabels([])
    ax_mdf.xaxis.set_minor_locator(AutoMinorLocator())
    ax_mdf.yaxis.set_minor_locator(AutoMinorLocator())

    # Centre: banana
    ax_ban = fig.add_subplot(gs[1, 1])
    feh_mids, age_meds, age_lo, age_hi = median_banana(r['feh'], r['age'])
    if len(feh_mids) >= 3:
        ax_ban.fill_between(feh_mids, age_lo, age_hi,
                             color=c, alpha=0.25, linewidth=0)
        ax_ban.plot(feh_mids, age_meds, '-', color=c, lw=2.0)
    ax_ban.axvline(r['mh_obs'], color='k', lw=1.0, ls='--', alpha=0.7,
                    label=r'obs [M/H] $=' + f'{r["mh_obs"]:.2f}$')
    ax_ban.set_xlabel('[Fe/H] assumed', fontsize=11)
    ax_ban.set_ylabel('Age inferred (Gyr)', fontsize=11)
    ax_ban.set_xlim(-1.05, 0.45)
    ax_ban.set_ylim(0, age_max)
    ax_ban.xaxis.set_minor_locator(AutoMinorLocator())
    ax_ban.yaxis.set_minor_locator(AutoMinorLocator())
    ax_ban.legend(loc='upper right')

    # Right: age posterior
    ax_post = fig.add_subplot(gs[1, 2], sharey=ax_ban)
    ax_post.hist(r['age_posterior'], bins=age_bins,
                  orientation='horizontal',
                  color=c, alpha=0.8, edgecolor='white', lw=0.3)
    lbl = (f"${r['age_median']:.1f}"
           f"^{{+{r['age_hi']-r['age_median']:.1f}}}"
           f"_{{-{r['age_median']-r['age_lo']:.1f}}}$ Gyr")
    ax_post.axhline(r['age_median'], color='k', lw=1.8, label=lbl)
    ax_post.axhline(r['age_lo'],     color='k', lw=0.9, ls='--')
    ax_post.axhline(r['age_hi'],     color='k', lw=0.9, ls='--')
    ax_post.set_xlabel('$N$ samples', fontsize=11)
    ax_post.set_ylabel('Age (Gyr)', fontsize=11)
    ax_post.yaxis.set_label_position('right')
    ax_post.yaxis.tick_right()
    ax_post.xaxis.set_minor_locator(AutoMinorLocator())
    ax_post.yaxis.set_minor_locator(AutoMinorLocator())
    ax_post.legend(loc='upper right')
    
    sm_age_val, sm_ep, sm_en = get_sm_age(r['star_id'])
    if np.isfinite(sm_age_val):
        ax_post.axhline(sm_age_val, color='tomato', lw=1.5, ls=':',
                        label=f'Stone-Martinez: ${sm_age_val:.1f}^{{+{sm_ep:.1f}}}_{{-{sm_en:.1f}}}$ Gyr')
        ax_post.legend(loc='upper right')
    
    fig.add_subplot(gs[0, 0]).set_visible(False)
    fig.add_subplot(gs[0, 2]).set_visible(False)

    fig.savefig(f'results/posteriors/{safe_id}_posterior.png')
    fig.savefig(f'results/posteriors/{safe_id}_posterior.pdf')
    plt.close(fig)

print(f"  Saved {len(results)} plots.\n")

# ── 5. Age-metallicity relation ───────────────────────────────────────────────
print("Plotting AMR...")
fig, ax = plt.subplots(figsize=(5.5, 4.5))

mh_arr  = np.array([r['mh_obs']     for r in results])
med_arr = np.array([r['age_median'] for r in results])
lo_arr  = np.array([r['age_lo']     for r in results])
hi_arr  = np.array([r['age_hi']     for r in results])
cols    = [cmap_mh((m - vmin_mh) / (vmax_mh - vmin_mh)) for m in mh_arr]

for i in range(len(results)):
    ax.errorbar(mh_arr[i], med_arr[i],
                yerr=[[med_arr[i]-lo_arr[i]], [hi_arr[i]-med_arr[i]]],
                fmt='o', color=cols[i], ms=6, lw=1.3,
                capsize=2.5, ecolor=cols[i], zorder=3)

# Stone-Martinez ages on the same plot
sm_mh, sm_age, sm_ep, sm_en = [], [], [], []
for r in results:
    a, ep, en = get_sm_age(r['star_id'])
    if np.isfinite(a):
        sm_mh.append(r['mh_obs'])
        sm_age.append(a)
        sm_ep.append(ep)
        sm_en.append(en)

if sm_mh:
    ax.errorbar(sm_mh, sm_age,
                yerr=[sm_en, sm_ep],
                fmt='s', color='tomato', ms=5, lw=1.0,
                capsize=2.0, ecolor='tomato', zorder=4,
                label='Stone-Martinez (2025)', mfc='white', mec='tomato')
    ax.legend(fontsize=9)

ax.set_xlabel('[M/H]', fontsize=12)
ax.set_ylabel('Age (Gyr)', fontsize=12)
ax.set_xlim(-1.5, 0.6)
ax.set_ylim(bottom=0)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

sm_cb = plt.cm.ScalarMappable(cmap=cmap_mh,
        norm=plt.Normalize(vmin=vmin_mh, vmax=vmax_mh))
sm_cb.set_array([])
cb = fig.colorbar(sm_cb, ax=ax)
cb.set_label('obs [M/H]', fontsize=10)

fig.savefig('results/posteriors/age_metallicity_relation.png')
fig.savefig('results/posteriors/age_metallicity_relation.pdf')
plt.close(fig)
print("  Saved: results/posteriors/age_metallicity_relation.png\n")
# ── 6. Table ──────────────────────────────────────────────────────────────────
table = pd.DataFrame([{
    'star_id':    r['star_id'],
    'teff_obs':   r['teff_obs'],
    'logg_obs':   r['logg_obs'],
    'lum_obs':    r['lum_obs'],
    'mh_obs':     r['mh_obs'],
    'age_median': round(r['age_median'], 2),
    'age_lo':     round(r['age_lo'],     2),
    'age_hi':     round(r['age_hi'],     2),
    'age_err_lo': round(r['age_median'] - r['age_lo'], 2),
    'age_err_hi': round(r['age_hi'] - r['age_median'], 2),
} for r in results]).sort_values('mh_obs')

table.to_csv('results/posteriors/age_posterior_table.csv', index=False)
print("Summary table:")
print(table[['star_id', 'mh_obs', 'age_median',
             'age_err_lo', 'age_err_hi']].to_string(index=False))
print(f"\nSaved: results/posteriors/age_posterior_table.csv")
print("Done.")

# ── Stone-Martinez vs our ages ────────────────────────────────────────────────
print("Plotting Stone-Martinez comparison...")
our_age, our_lo, our_hi = [], [], []
sm_age_c, sm_ep_c, sm_en_c = [], [], []
match_ids = []

for r in results:
    a, ep, en = get_sm_age(r['star_id'])
    if np.isfinite(a):
        our_age.append(r['age_median'])
        our_lo.append(r['age_median'] - r['age_lo'])
        our_hi.append(r['age_hi'] - r['age_median'])
        sm_age_c.append(a)
        sm_ep_c.append(ep)
        sm_en_c.append(en)
        match_ids.append(r['star_id'])

if len(our_age) > 0:
    our_age  = np.array(our_age)
    sm_age_c = np.array(sm_age_c)
    age_max  = max(our_age.max() + max(our_hi), sm_age_c.max()) * 1.1
    cols_m   = [cmap_mh((r['mh_obs'] - vmin_mh) / (vmax_mh - vmin_mh))
                for r in results if np.isfinite(get_sm_age(r['star_id'])[0])]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for i in range(len(our_age)):
        ax.errorbar(sm_age_c[i], our_age[i],
                    xerr=[[sm_en_c[i]], [sm_ep_c[i]]],
                    yerr=[[our_lo[i]], [our_hi[i]]],
                    fmt='o', color=cols_m[i], ms=6, lw=1.0,
                    capsize=2.5, ecolor=cols_m[i], zorder=3)

    # 1:1 line
    lim = (0, age_max)
    ax.plot(lim, lim, 'k--', lw=1.0, alpha=0.5, zorder=1)

    ax.set_xlabel('Stone-Martinez (2025) age (Gyr)', fontsize=12)
    ax.set_ylabel('This work — age (Gyr)', fontsize=12)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    sm_cb = plt.cm.ScalarMappable(cmap=cmap_mh,
            norm=plt.Normalize(vmin=vmin_mh, vmax=vmax_mh))
    sm_cb.set_array([])
    cb = fig.colorbar(sm_cb, ax=ax)
    cb.set_label('obs [M/H]', fontsize=10)

    fig.savefig('results/posteriors/sm_comparison.png')
    fig.savefig('results/posteriors/sm_comparison.pdf')
    plt.close(fig)
    print(f"  Saved: results/posteriors/sm_comparison.png  ({len(our_age)} matched stars)")
else:
    print("  No matched stars for comparison plot.")

# ── Combined age distribution ─────────────────────────────────────────────────
print("Plotting combined age distribution...")
all_ages = np.concatenate([r['age_posterior'] for r in results])

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.hist(all_ages, bins=50, density=True,
        color='steelblue', alpha=0.4, edgecolor='white', lw=0.3,
        label=f'N={len(results)} stars')

from scipy.stats import gaussian_kde
kde = gaussian_kde(all_ages, bw_method=0.15)
age_grid = np.linspace(0, all_ages.max() * 1.05, 500)
ax.plot(age_grid, kde(age_grid), '-', color='steelblue', lw=2.0)

ax.axvline(np.median(all_ages), color='k', lw=1.5, ls='--',
           label=f'Median: {np.median(all_ages):.1f} Gyr')

ax.set_xlabel('Age (Gyr)', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.legend(fontsize=9)

fig.savefig('results/posteriors/combined_age_distribution.png')
fig.savefig('results/posteriors/combined_age_distribution.pdf')
plt.close(fig)
print(f"  Saved: results/posteriors/combined_age_distribution.png")
print(f"  Combined median age: {np.median(all_ages):.1f} Gyr")
print(f"  16th–84th pct: {np.percentile(all_ages,16):.1f}–{np.percentile(all_ages,84):.1f} Gyr")
