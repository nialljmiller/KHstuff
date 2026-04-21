"""
make_apokasc_bananas.py

APOKASC analogue of make_platinum_bananas.py.

This script intentionally mirrors the platinum-banana workflow as closely as
possible, with the main differences being:

    * input catalogue source
    * APOKASC-specific metadata fields
    * output directory: results/apokasc/

It computes age-metallicity banana curves for each APOKASC star using MCMC
(emcee) with a constrained 3D log-probability function.

The free sampling dimensions are:
    initial_mass  — stellar mass (M☉)
    initial_met   — [Fe/H] (the metallicity axis of the banana)
    eep           — Equivalent Evolutionary Phase (position on the track)

At every walker step, the constrained parameters are derived from initial_met:
    alpha_fe      = 0.0                        (fixed bulge/APOKASC assumption)
    initial_he    = compute_y(initial_met)     (helium enrichment law)
    mixing_length = compute_ML(initial_met)    (APOKASC3 calibration)

This produces samples from p(age, [Fe/H] | teff_obs, lum_obs, αMLT relation,
Y relation) — the full degenerate banana surface — rather than discrete
optimizer results.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import emcee
import kiauhoku as kh

warnings.filterwarnings('ignore')

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs('results/apokasc/chains', exist_ok=True)
os.makedirs('results/apokasc/plots', exist_ok=True)

# ── MCMC settings ─────────────────────────────────────────────────────────────
N_WALKERS = 32
N_BURNIN = 300
N_ITER = 1000
NDIM = 3

# Observational uncertainties for the Gaussian likelihood
# teff: use per-star e_teff from catalogue
# lum: APOKASC luminosity proxy has no formal per-star errors here; use 0.1 dex
LUM_SIGMA_DEFAULT = 0.10   # dex log(L/L☉)

# ── Physical relations ────────────────────────────────────────────────────────
Y_PRIMORDIAL = 0.2485
DYDZ = 1.4
Z_SOLAR = 0.0134
LOGG_SUN = 4.4374      # cgs
TEFF_SUN = 5777.0      # K

AGE_UNIVERSE = 13.8   # Gyr — retained for downstream filtering / metadata


def compute_y(feh):
    """Helium enrichment: Y = Y_p + (dY/dZ)*Z."""
    z = Z_SOLAR * 10**feh
    return float(np.clip(Y_PRIMORDIAL + DYDZ * z, 0.24, 0.32))


def compute_ML(feh, lo, hi):
    """Mixing-length calibration: α_MLT = 0.02*[Fe/H] + 1.94 (APOKASC3)."""
    return float(np.clip(0.02 * feh + 1.94, lo + 1e-6, hi - 1e-6))


def compute_lum(teff, logg, mass):
    """log(L/Lsun) from Stefan-Boltzmann + fundamental relation."""
    return np.log10(mass) + 4.0 * np.log10(teff / TEFF_SUN) - (logg - LOGG_SUN)


# ── Stellar classification ────────────────────────────────────────────────────
def classify_star(logg):
    if np.isnan(logg):
        return 'unknown'
    return 'RGB' if (logg < 2.2 or logg > 3.0) else 'clump'


def banana_log_prob(pos, interp, teff_obs, lum_obs, teff_sigma, lum_sigma,
                    bounds, ml_lo, ml_hi):
    initial_mass, initial_met, eep = pos

    mass_lo, mass_hi = bounds['initial_mass']
    met_lo, met_hi = bounds['initial_met']
    eep_lo, eep_hi = bounds['eep']

    # Clip 0.1 dex inside each metallicity boundary to avoid the grid edge
    # where interpolation is unreliable and walkers find spurious solutions.
    MET_EDGE_BUFFER = 0.1
    met_lo_eff = met_lo + MET_EDGE_BUFFER
    met_hi_eff = met_hi - MET_EDGE_BUFFER

    if not (mass_lo < initial_mass < mass_hi and
            met_lo_eff < initial_met < met_hi_eff and
            eep_lo < eep < eep_hi):
        return -np.inf, None

    alpha_fe = 0.0
    initial_he = compute_y(initial_met)
    mixing_length = compute_ML(initial_met, ml_lo, ml_hi)

    he_lo, he_hi = bounds['initial_he']
    if not (he_lo <= initial_he <= he_hi):
        return -np.inf, None

    full_index = (initial_mass, initial_met, alpha_fe,
                  initial_he, mixing_length, eep)

    try:
        star = interp.get_star_eep(full_index)
    except Exception:
        return -np.inf, None

    if star is None or star.isna().any():
        return -np.inf, None

    teff_model = float(star['teff'])
    lum_model = float(star['lum'])

    log_prob = (
        -0.5 * ((teff_obs - teff_model) / teff_sigma) ** 2
        -0.5 * ((lum_obs - lum_model) / lum_sigma) ** 2
    )

    return log_prob, star


def filter_off_grid_samples(output, teff_obs, lum_obs, teff_sigma, n_sigma=3.0):
    """
    Remove samples where the interpolated model Teff or lum is more than
    n_sigma away from the observed values. These are walkers that have
    drifted to regions where the likelihood is spuriously non-zero.
    """
    lum_sigma = LUM_SIGMA_DEFAULT

    teff_model = output['teff'].values
    lum_model  = output['lum'].values

    ok = (
        np.isfinite(teff_model) & np.isfinite(lum_model) &
        (np.abs(teff_model - teff_obs) / teff_sigma <= n_sigma) &
        (np.abs(lum_model  - lum_obs)  / lum_sigma  <= n_sigma)
    )
    n_removed = (~ok).sum()
    if n_removed > 0:
        print(f"  Removed {n_removed:,} off-grid samples "
              f"({100*n_removed/len(output):.1f}%) outside {n_sigma}σ in Teff or lum")
    return output[ok].reset_index(drop=True)


# ── Grid loading ──────────────────────────────────────────────────────────────
def load_grid():
    print("Loading JT2017t12 grid...")
    qstring = '201 <= eep'
    jtgrid = kh.load_eep_grid("JT2017t12").query(qstring)
    jtgrid['mass'] = jtgrid['Mass(Msun)']
    jtgrid['teff'] = 10**jtgrid['Log Teff(K)']
    jtgrid['lum'] = jtgrid['L/Lsun']
    jtgrid['met'] = jtgrid.index.get_level_values('initial_met')
    jtgrid['initial_he'] = jtgrid.index.get_level_values('initial_he')
    jtgrid['mixing_length'] = jtgrid.index.get_level_values('mixing_length')
    jtgrid['alpha_fe'] = jtgrid.index.get_level_values('alpha_fe')
    jtgrid['age'] = jtgrid['Age(Gyr)']

    # ── Save observable bounds read directly from the raw grid ────────────────
    import json
    grid_obs_bounds = {
        'lum_min':  float(jtgrid['lum'].min()),
        'lum_max':  float(jtgrid['lum'].max()),
        'teff_min': float(jtgrid['teff'].min()),
        'teff_max': float(jtgrid['teff'].max()),
    }
    os.makedirs('results', exist_ok=True)
    with open('results/grid_obs_bounds.json', 'w') as _fj:
        json.dump(grid_obs_bounds, _fj, indent=2)
    print(f"  Grid obs bounds: lum=[{grid_obs_bounds['lum_min']:.2f}, "
          f"{grid_obs_bounds['lum_max']:.2f}]  "
          f"teff=[{grid_obs_bounds['teff_min']:.0f}, "
          f"{grid_obs_bounds['teff_max']:.0f}] K")

    jtgrid.set_name('jtgrid')
    jtgrid = jtgrid.to_interpolator()
    print("Grid loaded.\n")

    bounds = {
        name: (float(vals.min()), float(vals.max()))
        for name, vals in zip(jtgrid.index_names, jtgrid.index_columns)
    }
    return jtgrid, bounds, grid_obs_bounds


# ── APOKASC catalogue loading ─────────────────────────────────────────────────
def load_apokasc(path='MeridithRomanApokascCalibLtest5ns3L.out'):
    raw = pd.read_csv(path, sep=r'\s+')

    raw = raw.rename(columns={
        '2MASSID': 'star_id',
        'Teff': 'teff_obs',
        'Logg': 'logg_obs',
        'Fe/H': 'mh_obs',
        'Teff_err': 'e_teff_obs',
        'IntAge': 'int_age',
        'IntMass': 'int_mass',
        'C/N': 'cn_class',
    })

    if 'e_teff_obs' not in raw.columns:
        raw['e_teff_obs'] = 100.0

    bad = (
        (raw['int_age'] <= 0)
        | (raw['int_age'] > AGE_UNIVERSE)
        | (raw['int_mass'] <= 0)
        | (raw['teff_obs'] <= 4000)
        | (raw['cn_class'] != 'RGB')
    )

    stars = raw[~bad].copy()
    stars['lum_obs'] = compute_lum(
        stars['teff_obs'].values,
        stars['logg_obs'].values,
        stars['int_mass'].values,
    )

    # ── Remove stars above the grid luminosity ceiling ─────────────────────────
    import json
    try:
        with open('results/grid_obs_bounds.json') as _fj:
            _gob = json.load(_fj)
        lum_max = _gob['lum_max']
        n_before = len(stars)
        stars = stars[stars['lum_obs'] <= lum_max].copy()
        print(f"  Removed {n_before - len(stars)} stars with lum_obs > grid ceiling "
              f"({lum_max:.2f} log L/Lsun)")
    except FileNotFoundError:
        print("  WARNING: results/grid_obs_bounds.json not found — "
              "run load_grid() first to generate it. No lum ceiling applied.")
    stars['skip_reason'] = 'none'
    stars['fit_loss'] = np.nan
    stars = stars.dropna(
        subset=['star_id', 'teff_obs', 'lum_obs', 'logg_obs', 'mh_obs', 'e_teff_obs']
    ).reset_index(drop=True)

    print(f"Catalogue: {len(stars)} APOKASC stars eligible for banana MCMC\n")
    return stars


# ── Initial walker positions ──────────────────────────────────────────────────
def make_initial_positions(star_row, bounds, n_walkers, rng):
    """
    Scatter walkers around fit-like columns if they exist.
    Falls back to grid centre if no such columns are available, exactly in the
    same spirit as make_platinum_bananas.py.
    """
    mass_lo, mass_hi = bounds['initial_mass']
    met_lo, met_hi = bounds['initial_met']
    eep_lo, eep_hi = bounds['eep']

    c_mass = float(star_row.get('fit_initial_mass', np.nan))
    c_met = float(star_row.get('fit_initial_met', np.nan))
    c_eep = float(star_row.get('fit_eep', np.nan))

    if np.isnan(c_mass):
        c_mass = (mass_lo + mass_hi) / 2
    if np.isnan(c_met):
        c_met = (met_lo + met_hi) / 2
    if np.isnan(c_eep):
        c_eep = (eep_lo + eep_hi) / 2

    w_mass = 0.05 * (mass_hi - mass_lo)
    w_met = 0.05 * (met_hi - met_lo)
    w_eep = 0.05 * (eep_hi - eep_lo)

    pos = np.column_stack([
        np.clip(rng.normal(c_mass, w_mass, n_walkers), mass_lo, mass_hi),
        np.clip(rng.normal(c_met, w_met, n_walkers), met_lo, met_hi),
        np.clip(rng.normal(c_eep, w_eep, n_walkers), eep_lo, eep_hi),
    ])
    return pos


# ── Per-star banana plot ──────────────────────────────────────────────────────
def save_banana_plot(star_id, flat_samples, blobs_df,
                     teff_obs, lum_obs, logg_obs, mh_obs,
                     aux_value, stellar_class):
    safe_id = star_id.replace('/', '_')

    class_color = {'RGB': 'steelblue', 'clump': 'seagreen', 'unknown': 'grey'}
    c = class_color.get(stellar_class, 'grey')

    age_col = 'age' if 'age' in blobs_df.columns else 'Age(Gyr)'
    if age_col not in blobs_df.columns:
        return

    feh = flat_samples['initial_met'].values
    age = blobs_df[age_col].values
    mask = np.isfinite(feh) & np.isfinite(age)
    feh, age = feh[mask], age[mask]
    if len(feh) < 5:
        return

    age_med = np.median(age)
    age_lo = np.percentile(age, 16)
    age_hi = np.percentile(age, 84)
    age_plus = age_hi - age_med
    age_minus = age_med - age_lo

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

    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_hist = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis('off')

    # ── Main banana: median relation with 16th–84th percentile band ──────────
    n_feh_bins = int(np.clip(np.sqrt(len(feh)), 18, 36))
    feh_bins = np.linspace(feh_min, feh_max, n_feh_bins)
    med_ages, lo_ages, hi_ages, feh_mids = [], [], [], []
    for j in range(len(feh_bins) - 1):
        mask_bin = (feh >= feh_bins[j]) & (feh < feh_bins[j + 1])
        if mask_bin.sum() > 5:
            ages_bin = age[mask_bin]
            med_ages.append(np.median(ages_bin))
            lo_ages.append(np.percentile(ages_bin, 16))
            hi_ages.append(np.percentile(ages_bin, 84))
            feh_mids.append((feh_bins[j] + feh_bins[j + 1]) / 2.0)

    if med_ages:
        feh_mids = np.asarray(feh_mids)
        med_ages = np.asarray(med_ages)
        lo_ages = np.asarray(lo_ages)
        hi_ages = np.asarray(hi_ages)
        ax_main.fill_between(feh_mids, lo_ages, hi_ages, color=c, alpha=0.30, lw=0)
        ax_main.plot(feh_mids, med_ages, color=c, lw=2.2)

    obs_label = f'obs [M/H] = {mh_obs:.2f}'
    ax_main.axvline(mh_obs, color='0.25', lw=1.4, ls='--', label=obs_label)
    ax_top.axvline(mh_obs, color='0.25', lw=1.4, ls='--')

    # The posterior summary lines drawn on the histogram also continue into the
    # main banana panel so the answer line is visually shared.
    answer_label = rf"${age_med:.1f}^{{+{age_plus:.1f}}}_{{-{age_minus:.1f}}}$ Gyr"
    ax_main.axhline(age_med, color='k', lw=2.0, zorder=1)
    ax_main.axhline(age_lo, color='b', lw=1.2, ls='--', zorder=1)
    ax_main.axhline(age_hi, color='b', lw=1.2, ls='--', zorder=1)

    comp_label = None
    if np.isfinite(aux_value):
        comp_label = f'IntAge: {aux_value:.1f} Gyr'
        ax_main.axhline(aux_value, color='tomato', lw=1.2, ls=':', zorder=1)

    ax_main.set_xlabel('[Fe/H] assumed', fontsize=11)
    ax_main.set_ylabel('Age inferred (Gyr)', fontsize=11)
    ax_main.set_xlim(x_lo, x_hi)
    ax_main.set_ylim(y_lo, y_hi)
    ax_main.legend(loc='upper right', fontsize=8, frameon=True)

    # ── Top marginal: p([Fe/H]) sharing x with the main banana ────────────────
    x_grid = np.linspace(x_lo, x_hi, 400)
    pdf = None
    if len(np.unique(feh)) > 1:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(feh)
            pdf = kde(x_grid)
        except Exception:
            counts, edges = np.histogram(feh, bins=max(20, n_feh_bins), density=True)
            centres = 0.5 * (edges[:-1] + edges[1:])
            pdf = np.interp(x_grid, centres, counts, left=0.0, right=0.0)

    if pdf is not None and np.all(np.isfinite(pdf)):
        ax_top.fill_between(x_grid, 0.0, pdf, color='lightsteelblue', alpha=0.85)
        ax_top.plot(x_grid, pdf, color=c, lw=2.0)
    else:
        ax_top.hist(feh, bins=max(20, n_feh_bins), density=True,
                    color='lightsteelblue', alpha=0.85)

    ax_top.set_ylabel(r'$p([\mathrm{Fe/H}])$', fontsize=11)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # ── Right marginal: age histogram sharing y with the main banana ──────────
    # ── Right marginal: age histogram sharing y with the main banana ──────────
    # The raw sqrt(N) bin rule becomes absurdly fine for long chains (e.g. 192k
    # samples gives ~438 bins), which makes the histogram unreadable. Keep the
    # binning visually stable and switch to log-counts when the dynamic range is
    # large so the long age tail remains visible.
    n_age_bins = int(np.clip(np.sqrt(len(age)) / 3.0, 18, 32))
    age_bins = np.linspace(y_lo, y_hi, n_age_bins + 1)
    age_counts, _, _ = ax_hist.hist(
        age,
        bins=age_bins,
        orientation='horizontal',
        color='salmon',
        alpha=0.90,
        edgecolor='white',
        linewidth=0.6,
    )

    positive_counts = age_counts[age_counts > 0]
    if positive_counts.size:
        x_max = float(positive_counts.max())
        x_min = float(max(1.0, positive_counts.min()))
        ax_hist.set_xlim(0.0, x_max * 1.08)
        #if x_max / x_min > 30.0:
        #    ax_hist.set_xscale('log')
        #    ax_hist.set_xlim(1.0, x_max * 1.15)
                        
    else:
       ax_hist.set_xlim(0.0, 1.0)

    ax_hist.axhline(age_med, color='k', lw=2.0, label=answer_label)
    ax_hist.axhline(age_lo, color='b', lw=1.2, ls='--', label='Age IQR')
    ax_hist.axhline(age_hi, color='b', lw=1.2, ls='--')
    if comp_label is not None:
        ax_hist.axhline(aux_value, color='purple', lw=1.2, ls=':', label=comp_label)

    ax_hist.set_xlabel(r'$N$ samples', fontsize=11)
    ax_hist.set_ylabel('Age (Gyr)', fontsize=11)
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position('right')
    ax_hist.legend(loc='upper right', fontsize=8, frameon=True)


                        
    n_eff = mask.sum()
    fig.subplots_adjust(top=0.95)
    fig.savefig(f'results/apokasc/plots/{safe_id}.pdf', dpi=130, bbox_inches='tight')                        

    fig.suptitle(
        f"{star_id}  [{stellar_class}]\n"
        f"Teff={teff_obs:.0f} K   logg={logg_obs:.2f}   lum={lum_obs:.2f}\n   "
        f"obs[M/H]={mh_obs:.2f}   N_samples={n_eff:,}",
        fontsize=10, fontweight='bold', y=0.97
    )

    fig.savefig(f'results/apokasc/plots/{safe_id}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


# ── Single star MCMC ──────────────────────────────────────────────────────────
def run_star(star_row, jtgrid, bounds):
    star_id = star_row['star_id']
    teff_obs = float(star_row['teff_obs'])
    lum_obs = float(star_row['lum_obs'])
    logg_obs = float(star_row['logg_obs'])
    mh_obs = float(star_row['mh_obs'])
    e_teff = float(star_row['e_teff_obs'])
    int_age = float(star_row['int_age']) if 'int_age' in star_row else np.nan
    int_mass = float(star_row['int_mass']) if 'int_mass' in star_row else np.nan
    aux_value = int_age if np.isfinite(int_age) else 999.0
    stellar_class = classify_star(logg_obs)
    safe_id = star_id.replace('/', '_')

    # ── Skip if star is outside grid observable bounds ────────────────────────
    import json
    try:
        with open('results/grid_obs_bounds.json') as _fj:
            _gob = json.load(_fj)
        if lum_obs > _gob['lum_max']:
            print(f"  SKIP {star_id}: lum_obs={lum_obs:.2f} > grid lum_max={_gob['lum_max']:.2f} "
                  f"— star above grid ceiling, luminosity constraint inactive")
            return None
    except FileNotFoundError:
        pass  # bounds file not yet written; proceed anyway

    print(f"\n{'─' * 60}")
    print(f"Star: {star_id}  [{stellar_class}]")
    print(
        f"  Teff={teff_obs:.0f}±{e_teff:.0f} K   logg={logg_obs:.2f}   "
        f"lum={lum_obs:.2f}   obs[M/H]={mh_obs:.2f}"
    )
    if np.isfinite(int_age) or np.isfinite(int_mass):
        print(f"  APOKASC metadata: IntAge={int_age:.2f} Gyr   IntMass={int_mass:.2f} Msun")
    print(f"  MCMC: {N_WALKERS} walkers × {N_BURNIN} burn-in + {N_ITER} iter")

    teff_sigma = max(e_teff, 50.0)
    lum_sigma = LUM_SIGMA_DEFAULT

    ml_lo, ml_hi = bounds['mixing_length']

    rng = np.random.default_rng(seed=abs(hash(star_id)) % (2**32))
    pos0 = make_initial_positions(star_row, bounds, N_WALKERS, rng)

    sampler = emcee.EnsembleSampler(
        N_WALKERS, NDIM, banana_log_prob,
        args=(jtgrid, teff_obs, lum_obs, teff_sigma, lum_sigma,
              bounds, ml_lo, ml_hi),
        blobs_dtype=[('star', object)],
    )

    print(f"  Running burn-in ({N_BURNIN} steps)...")
    pos, _, _, _ = sampler.run_mcmc(pos0, N_BURNIN, progress=False)
    sampler.reset()

    print(f"  Running production ({N_ITER} steps)...")
    sampler.run_mcmc(pos, N_ITER, progress=False)

    acc = np.mean(sampler.acceptance_fraction)
    print(f"  Mean acceptance fraction: {acc:.3f}  (ideal: 0.2–0.5)")

    flat_chain = sampler.flatchain
    flat_samples = pd.DataFrame(flat_chain, columns=['initial_mass', 'initial_met', 'eep'])

    flat_samples['alpha_fe'] = 0.0
    flat_samples['initial_he'] = flat_samples['initial_met'].apply(compute_y)
    flat_samples['mixing_length'] = flat_samples['initial_met'].apply(
        lambda f: compute_ML(f, ml_lo, ml_hi)
    )

    blob_list = []
    for b in sampler.flatblobs:
        star_b = b['star'] if b is not None else None
        if star_b is not None and isinstance(star_b, pd.Series):
            blob_list.append(star_b)
        else:
            blob_list.append(pd.Series(dtype='float64'))
    blobs_df = pd.DataFrame.from_records(blob_list)

    output = pd.concat(
        [flat_samples.reset_index(drop=True), blobs_df.reset_index(drop=True)],
        axis=1,
    )

    output = filter_off_grid_samples(output, teff_obs, lum_obs, teff_sigma)

    age_col = 'age' if 'age' in output.columns else 'Age(Gyr)'
    if age_col in output.columns:
        valid_ages = output[age_col].dropna()
        print(
            f"  Age posterior: median={valid_ages.median():.2f} Gyr  "
            f"16th={valid_ages.quantile(0.16):.2f}  "
            f"84th={valid_ages.quantile(0.84):.2f}"
        )
        print(
            f"  [Fe/H] explored: "
            f"{flat_samples['initial_met'].min():.2f} to "
            f"{flat_samples['initial_met'].max():.2f}"
        )
        print(f"  Valid samples: {len(valid_ages):,} / {len(output):,}")

    save_banana_plot(
        star_id, flat_samples, blobs_df,
        teff_obs, lum_obs, logg_obs, mh_obs,
        aux_value, stellar_class
    )
    print(f"  Plot saved: results/apokasc/plots/{safe_id}.png")

    result = {
        'star_id': star_id,
        'stellar_class': stellar_class,
        'teff_obs': teff_obs,
        'lum_obs': lum_obs,
        'logg_obs': logg_obs,
        'mh_obs': mh_obs,
        'e_teff': e_teff,
        'int_age': int_age,
        'int_mass': int_mass,
        'flat_samples': flat_samples,
        'blobs_df': blobs_df,
        'output': output,
        'acceptance_fraction': acc,
        'n_walkers': N_WALKERS,
        'n_burnin': N_BURNIN,
        'n_iter': N_ITER,
    }
    chain_path = f'results/apokasc/chains/{safe_id}.pkl'
    with open(chain_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"  Chain saved: {chain_path}")

    return result


# ── Combine all chains into summary outputs ───────────────────────────────────
def combine_chains():
    chain_dir = 'results/apokasc/chains'
    chain_files = [f for f in os.listdir(chain_dir) if f.endswith('.pkl')]
    print(f"Combining {len(chain_files)} star chains...")

    all_rows = []
    banana_dict = {}

    for fname in sorted(chain_files):
        with open(os.path.join(chain_dir, fname), 'rb') as f:
            res = pickle.load(f)

        star_id = res['star_id']
        output = res['output'].copy()
        output['star_id'] = star_id
        output['stellar_class'] = res['stellar_class']
        output['teff_obs'] = res['teff_obs']
        output['lum_obs'] = res['lum_obs']
        output['logg_obs'] = res['logg_obs']
        output['mh_obs'] = res['mh_obs']
         
        teff_obs   = res['teff_obs']
        lum_obs    = res['lum_obs']
        teff_sigma = max(res['e_teff'], 50.0)
        output = filter_off_grid_samples(output, teff_obs, lum_obs, teff_sigma)
       
        all_rows.append(output)
        banana_dict[star_id] = output

    if not all_rows:
        print("No chains found.")
        return

    banana_df = pd.concat(all_rows, ignore_index=True)
    banana_df.to_csv('results/apokasc/banana_data.csv', index=False)
    print(f"Banana table: {len(banana_df):,} rows  ({len(banana_dict)} stars)")
    print("Saved to results/apokasc/banana_data.csv")

    with open('results/apokasc/bananas.pkl', 'wb') as f:
        pickle.dump(banana_dict, f)
    print("Banana dict saved to results/apokasc/bananas.pkl")

    print("Making summary banana plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "APOKASC Sample — Age–[Fe/H] Bananas (MCMC)\n"
        "Left: RGB (logg < 2.2 or > 3.0)   Right: Clump (2.2 ≤ logg ≤ 3.0)",
        fontsize=12, fontweight='bold'
    )

    mh_all = banana_df['mh_obs'].dropna().values
    vmin, vmax = np.nanpercentile(mh_all, 5), np.nanpercentile(mh_all, 95)
    cmap = plt.cm.plasma
    age_col = 'age' if 'age' in banana_df.columns else 'Age(Gyr)'

    for ax, target_class in zip(axes, ['RGB', 'clump']):
        ax.set_title(target_class, fontsize=12)
        n_plotted = 0
        for sid, output in banana_dict.items():
            if output['stellar_class'].iloc[0] != target_class:
                continue
            mh_obs = output['mh_obs'].iloc[0]
            color = cmap((mh_obs - vmin) / max(vmax - vmin, 1e-6))

            if age_col not in output.columns:
                continue
            feh = output['initial_met'].values
            age = output[age_col].values
            mask = np.isfinite(feh) & np.isfinite(age)
            if mask.sum() < 10:
                continue

            feh_bins = np.linspace(-1.0, 0.4, 25)
            meds, feh_mids = [], []
            for j in range(len(feh_bins) - 1):
                b_mask = (feh[mask] >= feh_bins[j]) & (feh[mask] < feh_bins[j + 1])
                if b_mask.sum() > 5:
                    meds.append(np.median(age[mask][b_mask]))
                    feh_mids.append((feh_bins[j] + feh_bins[j + 1]) / 2)
            if len(meds) >= 2:
                ax.plot(feh_mids, meds, '-', color=color, lw=1.3, alpha=0.8)
                n_plotted += 1

        ax.set_xlabel('[Fe/H]', fontsize=11)
        ax.set_ylabel('Age (Gyr)', fontsize=11)
        ax.set_xlim(-1.05, 0.45)
        ax.set_ylim(bottom=0)
        ax.text(
            0.97, 0.97, f'N={n_plotted} stars',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='obs [M/H]', shrink=0.8)
    fig.savefig('results/apokasc/banana_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Summary plot saved to results/apokasc/banana_summary.png")

    print(f"""
─────────────────────────────────────────────────────────────
STAGED FOR STEP 4 (MDF sampling):

    import pickle, numpy as np
    with open('results/apokasc/bananas.pkl', 'rb') as f:
        bananas = pickle.load(f)

    output = bananas[star_id]   # DataFrame of MCMC samples
    age_col = 'age'             # or 'Age(Gyr)'

    # Method: weight MCMC samples by the bulge MDF
    from scipy.stats import gaussian_kde
    mdf_kde = gaussian_kde(zoccali_feh_samples)

    feh_samples = output['initial_met'].values
    age_samples = output[age_col].values
    mask = np.isfinite(feh_samples) & np.isfinite(age_samples)

    weights = mdf_kde(feh_samples[mask])
    weights /= weights.sum()

    # Weighted age posterior
    age_posterior_samples = np.random.choice(
        age_samples[mask], size=10000, p=weights, replace=True)
    # → histogram of age_posterior_samples = age posterior for this star
─────────────────────────────────────────────────────────────
""")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APOKASC banana MCMC')
    parser.add_argument('--star_id', type=str, default=None,
                        help='Run a single star by ID')
    parser.add_argument('--star_index', type=int, default=None,
                        help='Run a single star by 0-based index in stars_to_run')
    parser.add_argument('--combine', action='store_true',
                        help='Combine all completed chains into summary outputs')
    parser.add_argument('--n_walkers', type=int, default=32)
    parser.add_argument('--n_burnin',  type=int, default=300)
    parser.add_argument('--n_iter',    type=int, default=1000)
    args = parser.parse_args()
    N_WALKERS = args.n_walkers
    N_BURNIN  = args.n_burnin
    N_ITER    = args.n_iter

    if args.combine:
        combine_chains()
        sys.exit(0)

    jtgrid, bounds, grid_obs_bounds = load_grid()

    stars_to_run = load_apokasc()

    if args.star_id is not None:
        matches = stars_to_run[stars_to_run['star_id'] == args.star_id]
        if len(matches) == 0:
            print(f"ERROR: star_id '{args.star_id}' not found in eligible stars.")
            sys.exit(1)
        run_star(matches.iloc[0], jtgrid, bounds)

    elif args.star_index is not None:
        if args.star_index >= len(stars_to_run):
            print(f"ERROR: star_index {args.star_index} out of range "
                  f"(max {len(stars_to_run)-1}).")
            sys.exit(1)
        run_star(stars_to_run.iloc[args.star_index], jtgrid, bounds)

    else:
        already_done = set(
            f.replace('.pkl', '')
            for f in os.listdir('results/apokasc/chains')
            if f.endswith('.pkl')
        )
        n_total = len(stars_to_run)
        for i, (_, star_row) in enumerate(stars_to_run.iterrows()):
            safe_id = star_row['star_id'].replace('/', '_')
            #if safe_id in already_done:
            #    print(f"[{i+1:3d}/{n_total}] {star_row['star_id']}  — already done, skipping")
            #    continue
            print(f"[{i+1:3d}/{n_total}]", end='')
            run_star(star_row, jtgrid, bounds)

        print("\nAll stars done. Run with --combine to generate summary outputs.")
