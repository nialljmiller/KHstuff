'''
make_age_posteriors.py  —  Step 4

For each valid RGB banana star, weight the MCMC samples by the Zoccali et al.
(2017) bulge MDF and produce an age posterior histogram.

Method
------
The banana gives samples from p(age, [Fe/H] | Teff, lum).
The MDF gives p([Fe/H]) for the bulge sightline (l,b) = (0°, -1°).

The age posterior is:
    p(age) ∝ ∫ p(age | [Fe/H]) × p_MDF([Fe/H]) d[Fe/H]

With MCMC samples this reduces to weighting each sample by the MDF KDE
evaluated at that sample's initial_met, then histogramming the weighted ages.

Inputs
------
zocalli.dat          — Zoccali et al. (2017) multi-field catalogue.
                       Only rows with IDs starting 'LRp0m1' are used
                       (the (l,b)=(0°,-1°) field, N≈432).
                       Column index 7 (last) = [Fe/H].
results/bananas/bananas.pkl

Outputs
-------
results/posteriors/
    <star_id>_posterior.png    — Figure 8-style: MDF / banana / age posterior
    age_posterior_summary.png  — All age posteriors overlaid
    age_posterior_table.csv    — Median, 16th, 84th pct ages for paper
'''

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings('ignore')

os.makedirs('results/posteriors', exist_ok=True)

AGE_UNIVERSE = 13.8

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
            feh = float(parts[7])
            zoc_rows.append(feh)
        except ValueError:
            continue

zoc_feh = np.array(zoc_rows)
print(f"  Zoccali (l,b)=(0,-1) field: N={len(zoc_feh)} stars")
print(f"  [Fe/H] range: {zoc_feh.min():.2f} to {zoc_feh.max():.2f}")
print(f"  Median [Fe/H]: {np.median(zoc_feh):.2f}\n")

# Build KDE from the Zoccali [Fe/H] distribution
mdf_kde = gaussian_kde(zoc_feh, bw_method=0.15)

# ── 2. Load banana chains ─────────────────────────────────────────────────────
print("Loading bananas.pkl...")
with open('results/bananas/bananas.pkl', 'rb') as f:
    bananas = pickle.load(f)
print(f"  {len(bananas)} stars in bananas.pkl\n")

def get_age_col(df):
    for c in ['age', 'Age(Gyr)']:
        if c in df.columns:
            return c
    return None

def classify_star(logg):
    return 'RGB' if (float(logg) < 2.2 or float(logg) > 3.0) else 'clump'

# ── 3. Compute age posteriors ─────────────────────────────────────────────────
TEFF_MAX = 5500.0
AGE_BINS  = np.linspace(0, AGE_UNIVERSE, 30)
FEH_GRID  = np.linspace(-1.1, 0.5, 200)

results = []

for star_id, output in bananas.items():
    # Science selection
    teff_obs = float(output['teff_obs'].iloc[0])
    logg_obs = float(output['logg_obs'].iloc[0])
    mh_obs   = float(output['mh_obs'].iloc[0])
    lum_obs  = float(output['lum_obs'].iloc[0])
    stellar_class = output['stellar_class'].iloc[0] if 'stellar_class' in output.columns else classify_star(logg_obs)

    if stellar_class != 'RGB' or teff_obs >= TEFF_MAX:
        continue

    age_col = get_age_col(output)
    if age_col is None:
        continue

    feh_samples = output['initial_met'].values
    age_samples = output[age_col].values

    # Physical filter
    ok = (np.isfinite(feh_samples) & np.isfinite(age_samples) &
          (age_samples > 0) )
    feh = feh_samples[ok]
    age = age_samples[ok]

    if len(feh) < 100:
        continue

    # ── Weight by MDF ─────────────────────────────────────────────────────────
    weights = mdf_kde(feh)
    weights = np.maximum(weights, 0)
    w_sum   = weights.sum()
    if w_sum == 0:
        print(f"  SKIP {star_id}: zero total MDF weight")
        continue
    weights /= w_sum

    # Weighted age posterior
    age_posterior = np.random.choice(age, size=10000, p=weights, replace=True)

    med_age = np.median(age_posterior)
    lo_age  = np.percentile(age_posterior, 16)
    hi_age  = np.percentile(age_posterior, 84)

    print(f"  {star_id}  [M/H]={mh_obs:.2f}  "
          f"age = {med_age:.1f} +{hi_age-med_age:.1f} -{med_age-lo_age:.1f} Gyr")

    results.append({
        'star_id':    star_id,
        'teff_obs':   teff_obs,
        'logg_obs':   logg_obs,
        'lum_obs':    lum_obs,
        'mh_obs':     mh_obs,
        'age_median': med_age,
        'age_lo':     lo_age,
        'age_hi':     hi_age,
        'age_posterior': age_posterior,
        'feh':        feh,
        'age':        age,
        'weights':    weights,
    })

print(f"\n{len(results)} stars with valid age posteriors.\n")

# ── 4. Per-star Figure 8-style plot ──────────────────────────────────────────
print("Plotting per-star Figure 8-style posteriors...")

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

cmap_mh = plt.cm.RdYlBu_r
vmin_mh, vmax_mh = -1.1, 0.45

for r in results:
    star_id = r['star_id']
    safe_id = star_id.replace('/', '_')
    c = cmap_mh((r['mh_obs'] - vmin_mh) / (vmax_mh - vmin_mh))

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1, 3],
                             width_ratios=[1, 2, 1],
                             hspace=0.05, wspace=0.35)

    # ── Top centre: MDF ───────────────────────────────────────────────────────
    ax_mdf = fig.add_subplot(gs[0, 1])
    ax_mdf.fill_between(FEH_GRID, mdf_kde(FEH_GRID),
                         color='steelblue', alpha=0.4)
    ax_mdf.plot(FEH_GRID, mdf_kde(FEH_GRID), color='steelblue', lw=1.5)
    ax_mdf.axvline(r['mh_obs'], color='k', lw=1.2, ls='--')
    ax_mdf.set_xlim(-1.05, 0.45)
    ax_mdf.set_ylabel('p([Fe/H])', fontsize=8)
    ax_mdf.set_xticklabels([])
    ax_mdf.set_title('Bulge MDF (Zoccali+2017)', fontsize=8)
    ax_mdf.tick_params(labelsize=7)

    # ── Centre: banana ────────────────────────────────────────────────────────
    ax_ban = fig.add_subplot(gs[1, 1])
    feh_mids, age_meds, age_lo, age_hi = median_banana(r['feh'], r['age'])

    if len(feh_mids) >= 3:
        ax_ban.fill_between(feh_mids, age_lo, age_hi,
                             color=c, alpha=0.25, linewidth=0)
        ax_ban.plot(feh_mids, age_meds, '-', color=c, lw=2.2)

    ax_ban.axvline(r['mh_obs'], color='k', lw=1.2, ls='--',
                    label=f"obs [M/H]={r['mh_obs']:.2f}")
    ax_ban.axhline(AGE_UNIVERSE, color='dimgrey', lw=0.8, ls=':', alpha=0.6)
    ax_ban.set_xlabel('[Fe/H] Assumed', fontsize=10)
    ax_ban.set_ylabel('Age Inferred (Gyr)', fontsize=10)
    ax_ban.set_xlim(-1.05, 0.45)
    ax_ban.set_ylim(0, AGE_UNIVERSE + 0.5)
    ax_ban.tick_params(labelsize=8)
    ax_ban.legend(fontsize=7, loc='upper right')

    # ── Right: age posterior ──────────────────────────────────────────────────
    ax_post = fig.add_subplot(gs[1, 2])
    ax_post.hist(r['age_posterior'], bins=AGE_BINS, orientation='horizontal',
                  color=c, alpha=0.85, edgecolor='white')
    ax_post.axhline(r['age_median'], color='k', lw=2,
                     label=f"{r['age_median']:.1f} Gyr")
    ax_post.axhline(r['age_lo'],     color='k', lw=1, ls='--')
    ax_post.axhline(r['age_hi'],     color='k', lw=1, ls='--')
    ax_post.axhline(AGE_UNIVERSE, color='dimgrey', lw=0.8, ls=':', alpha=0.6)
    ax_post.set_ylim(0, AGE_UNIVERSE + 0.5)
    ax_post.set_xlabel('N samples', fontsize=9)
    ax_post.set_ylabel('Age Posterior (Gyr)', fontsize=9)
    ax_post.yaxis.set_label_position('right')
    ax_post.yaxis.tick_right()
    ax_post.tick_params(labelsize=7)
    ax_post.legend(fontsize=8, loc='upper right')
    ax_post.set_title('Age Posterior', fontsize=8)

    # Hide unused top panels
    fig.add_subplot(gs[0, 0]).set_visible(False)
    fig.add_subplot(gs[0, 2]).set_visible(False)

    fig.suptitle(
        f"{star_id}\n"
        f"Teff={r['teff_obs']:.0f} K   logg={r['logg_obs']:.2f}   "
        f"obs[M/H]={r['mh_obs']:.2f}   "
        f"Age = {r['age_median']:.1f}"
        f"+{r['age_hi']-r['age_median']:.1f}"
        f"-{r['age_median']-r['age_lo']:.1f} Gyr",
        fontsize=9, fontweight='bold'
    )

    fig.savefig(f"results/posteriors/{safe_id}_posterior.png",
                dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)

print(f"  Saved {len(results)} posterior plots.\n")

# ── 5. Summary plot: all age posteriors overlaid ──────────────────────────────
print("Plotting age posterior summary...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: KDE of each star's age posterior, coloured by obs [M/H]
ax = axes[0]
age_grid = np.linspace(0, AGE_UNIVERSE, 200)
for r in results:
    c = cmap_mh((r['mh_obs'] - vmin_mh) / (vmax_mh - vmin_mh))
    try:
        kde = gaussian_kde(r['age_posterior'], bw_method=0.3)
        ax.plot(age_grid, kde(age_grid), '-', color=c, lw=1.5, alpha=0.7)
    except Exception:
        pass

ax.axvline(AGE_UNIVERSE, color='dimgrey', lw=1, ls=':')
ax.set_xlabel('Age (Gyr)', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)
ax.set_title('Age Posteriors — all RGB stars', fontsize=11)
ax.set_xlim(0, AGE_UNIVERSE + 0.5)

sm = plt.cm.ScalarMappable(cmap=cmap_mh,
     norm=plt.Normalize(vmin=vmin_mh, vmax=vmax_mh))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='obs [M/H]')

# Right: median age vs obs [M/H] — the age-metallicity relation
ax2 = axes[1]
mh_arr  = np.array([r['mh_obs']     for r in results])
med_arr = np.array([r['age_median'] for r in results])
lo_arr  = np.array([r['age_lo']     for r in results])
hi_arr  = np.array([r['age_hi']     for r in results])
colors  = [cmap_mh((m - vmin_mh) / (vmax_mh - vmin_mh)) for m in mh_arr]

for i in range(len(results)):
    ax2.errorbar(mh_arr[i], med_arr[i],
                 yerr=[[med_arr[i] - lo_arr[i]], [hi_arr[i] - med_arr[i]]],
                 fmt='o', color=colors[i], ms=7, lw=1.5, capsize=3,
                 ecolor=colors[i], zorder=3)

ax2.axhline(AGE_UNIVERSE, color='dimgrey', lw=1, ls=':')
ax2.set_xlabel('obs [M/H]', fontsize=12)
ax2.set_ylabel('Age (Gyr)', fontsize=12)
ax2.set_title('Age–Metallicity Relation (MDF-weighted posteriors)', fontsize=11)
ax2.set_ylim(0, AGE_UNIVERSE + 0.5)
ax2.set_xlim(-1.5, 0.6)

fig.suptitle(
    f"RGB Platinum Sample — MDF-Weighted Age Posteriors  (N={len(results)})\n"
    f"Zoccali et al. (2017) MDF  |  (l,b) = (0°, -1°)  |  N_MDF={len(zoc_feh)}",
    fontsize=11, fontweight='bold'
)
fig.tight_layout()
fig.savefig('results/posteriors/age_posterior_summary.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Saved: results/posteriors/age_posterior_summary.png\n")

# ── 6. Summary table ──────────────────────────────────────────────────────────
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
print(table[['star_id', 'mh_obs', 'age_median', 'age_err_lo', 'age_err_hi']].to_string(index=False))
print(f"\nSaved to results/posteriors/age_posterior_table.csv")
print("Done.")
