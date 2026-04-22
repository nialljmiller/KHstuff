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
        'IntAge_err': 'e_int_age',
        'IntMass': 'int_mass',
        'C/N': 'cn_class',
        'Fe/H_err': 'e_mh_obs',
    })

    if 'e_teff_obs' not in raw.columns:
        raw['e_teff_obs'] = 100.0
    if 'e_mh_obs' not in raw.columns:
        raw['e_mh_obs'] = 0.1
    if 'e_int_age' not in raw.columns:
        raw['e_int_age'] = np.nan

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
                     aux_value, stellar_class,
                     alpha_fe=0.0, int_mass=float('nan'), e_teff=float('nan'),
                     acc=float('nan'),
                     e_mh_obs=0.1, e_int_age=float('nan'),
                     out_dir='results/apokasc/plots'):
    safe_id = star_id.replace('/', '_')
    os.makedirs(out_dir, exist_ok=True)

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

    # ── Top-right info box: all fundamental parameters ────────────────────────
    info_lines = [
        f"star_id:      {star_id}",
        f"class:        {stellar_class}",
        f"Teff:         {teff_obs:.0f} K",
        f"e_Teff:       {e_teff:.0f} K" if np.isfinite(e_teff) else "e_Teff:       —",
        f"Logg:         {logg_obs:.3f} cgs",
        f"Lum:          {lum_obs:.3f} log L/Lsun",
        f"[Fe/H] obs:   {mh_obs:.3f}",
        f"[alpha/Fe]:   {alpha_fe:.2f}",
        f"Mass:         {int_mass:.3f} Msun" if np.isfinite(int_mass) else "Mass:         —",
        f"IntAge:       {aux_value:.2f} Gyr" if np.isfinite(aux_value) else "IntAge:       —",
        f"accept. frac: {acc:.3f}" if np.isfinite(acc) else "accept. frac: —",
    ]
    ax_empty.text(
        0.98, 0.95, '\n'.join(info_lines),
        transform=ax_empty.transAxes,
        ha='right', va='top',
        fontsize=8.5, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                  edgecolor='#aaaaaa', alpha=0.9),
    )

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

    # ── Errorbar points at obs [M/H] ──────────────────────────────────────────
    # Inferred point: median + asymmetric 1-sigma from samples within ±FEH_WINDOW
    _feh_mask = (feh >= mh_obs - BESTFIT_FEH_WINDOW) & (feh <= mh_obs + BESTFIT_FEH_WINDOW)
    _inf_med = age_med  # fallback to global median if window is empty
    if _feh_mask.sum() >= 5:
        _age_at_obs = age[_feh_mask]
        _inf_med = float(np.median(_age_at_obs))
        _inf_lo  = float(np.percentile(_age_at_obs, 16))
        _inf_hi  = float(np.percentile(_age_at_obs, 84))
        ax_main.errorbar(
            mh_obs, _inf_med,
            yerr=[[_inf_med - _inf_lo], [_inf_hi - _inf_med]],
            fmt='o', color='steelblue', ms=7, lw=1.8,
            capsize=4, zorder=5, label='Inferred',
        )

    # APOKASC point: large star marker, no error bars
    comp_label = None
    if np.isfinite(aux_value):
        comp_label = f'APOKASC: {aux_value:.1f} Gyr'
        ax_main.plot(
            mh_obs, aux_value,
            '*', color='k', ms=14, zorder=5, label='APOKASC',
        )

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

    ax_hist.axhline(_inf_med, color='steelblue', lw=1.5, ls='--', zorder=2)
    if np.isfinite(aux_value):
        ax_hist.axhline(aux_value, color='k', lw=1.5, ls='-', zorder=2)

    ax_hist.set_xlabel(r'$N$ samples', fontsize=11)
    ax_hist.set_ylabel('Age (Gyr)', fontsize=11)
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position('right')
    ax_hist.legend(loc='upper right', fontsize=8, frameon=True)


                   
    n_eff = mask.sum()

    fig.subplots_adjust(top=0.93)
    fig.suptitle(
        f"{star_id}  [{stellar_class}]   N_samples={n_eff:,}",
        fontsize=10, fontweight='bold', y=0.99
    )

    fig.savefig(os.path.join(out_dir, f'{safe_id}.pdf'), dpi=130, bbox_inches='tight')
    png_path = os.path.join(out_dir, f'{safe_id}.png')
    fig.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return png_path


# ── Single star MCMC ──────────────────────────────────────────────────────────
def run_star(star_row, jtgrid, bounds):
    star_id = star_row['star_id']
    teff_obs = float(star_row['teff_obs'])
    lum_obs = float(star_row['lum_obs'])
    logg_obs = float(star_row['logg_obs'])
    mh_obs = float(star_row['mh_obs'])
    e_teff = float(star_row['e_teff_obs'])
    int_age = float(star_row['int_age']) if 'int_age' in star_row else np.nan
    e_int_age = float(star_row['e_int_age']) if 'e_int_age' in star_row else np.nan
    int_mass = float(star_row['int_mass']) if 'int_mass' in star_row else np.nan
    e_mh_obs = float(star_row['e_mh_obs']) if 'e_mh_obs' in star_row else 0.1
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
        aux_value, stellar_class,
        alpha_fe=0.0, int_mass=int_mass, e_teff=e_teff, acc=acc,
        e_mh_obs=e_mh_obs, e_int_age=e_int_age,
    )
    print(f"  Plot saved: results/apokasc/plots/{safe_id}.png")

    result = {
        'star_id': star_id,
        'stellar_class': stellar_class,
        'teff_obs': teff_obs,
        'lum_obs': lum_obs,
        'logg_obs': logg_obs,
        'mh_obs': mh_obs,
        'alpha_fe': 0.0,
        'e_teff': e_teff,
        'e_mh_obs': e_mh_obs,
        'int_age': int_age,
        'e_int_age': e_int_age,
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

    # ── Fundamental parameters table (one row per star) ───────────────────────
    # Columns: Teff, Logg, Luminosity, Metallicity, AlphaFe, Mass, IntAge
    fund_rows = []
    for fname in sorted(chain_files):
        with open(os.path.join(chain_dir, fname), 'rb') as f:
            res = pickle.load(f)
        fund_rows.append({
            'star_id':      res['star_id'],
            'stellar_class': res['stellar_class'],
            'Teff_K':       res['teff_obs'],
            'Logg_cgs':     res['logg_obs'],
            'Lum_logLsun':  res['lum_obs'],
            'FeH_obs':      res['mh_obs'],
            'AlphaFe':      res.get('alpha_fe', 0.0),
            'Mass_Msun':    res.get('int_mass', np.nan),
            'IntAge_Gyr':   res.get('int_age', np.nan),
            'e_Teff_K':     res.get('e_teff', np.nan),
            'acceptance_fraction': res.get('acceptance_fraction', np.nan),
        })
    fund_df = pd.DataFrame(fund_rows).sort_values('FeH_obs').reset_index(drop=True)
    fund_df.to_csv('results/apokasc/apokasc_fundamental_parameters.csv', index=False)
    print(f"Fundamental parameters saved to results/apokasc/apokasc_fundamental_parameters.csv"
          f"  ({len(fund_df)} stars)")

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


# ── Best-fit scoring ──────────────────────────────────────────────────────────
# Criterion: does the point (mh_obs, int_age) fall within the IQR
# (25th–75th percentile) of the banana at the observed metallicity?

BESTFIT_MIN_SAMPLES = 500
BESTFIT_FEH_WINDOW  = 0.15   # dex half-width around obs [M/H] for IQR read


def score_chain_bestfit(res):
    """
    Pass criterion: does the point (mh_obs, int_age) fall within the IQR
    (25th–75th percentile) of the banana at the observed metallicity?
    Fast — array ops only, no KDE.
    Returns (metrics_dict, passes_bool), or (None, False) if unscorable.
    """
    output  = res.get('output')
    if output is None:
        return None, False

    age_col = 'age' if 'age' in output.columns else 'Age(Gyr)'
    if age_col not in output.columns:
        return None, False

    mh_obs  = float(res['mh_obs'])
    int_age = float(res.get('int_age', np.nan))
    if not np.isfinite(int_age):
        return None, False

    feh = output['initial_met'].values
    age = output[age_col].values
    ok  = np.isfinite(feh) & np.isfinite(age) & (age > 0)

    if ok.sum() < BESTFIT_MIN_SAMPLES:
        return None, False

    mask = ok & (feh >= mh_obs - BESTFIT_FEH_WINDOW) & (feh <= mh_obs + BESTFIT_FEH_WINDOW)
    if mask.sum() < 20:
        return None, False

    age_at_obs = age[mask]
    q25 = float(np.percentile(age_at_obs, 25))
    q75 = float(np.percentile(age_at_obs, 75))
    passes = bool(q25 <= int_age <= q75)

    metrics = {
        'star_id':           res['star_id'],
        'stellar_class':     res['stellar_class'],
        'Teff_K':            float(res['teff_obs']),
        'e_Teff_K':          float(res.get('e_teff', np.nan)),
        'Logg_cgs':          float(res['logg_obs']),
        'Lum_logLsun':       float(res['lum_obs']),
        'FeH_obs':           round(mh_obs, 3),
        'AlphaFe':           float(res.get('alpha_fe', 0.0)),
        'Mass_Msun':         float(res.get('int_mass', np.nan)),
        'IntAge_Gyr':        round(int_age, 2),
        'banana_q25':        round(q25, 2),
        'banana_q75':        round(q75, 2),
        'acceptance_frac':   float(res.get('acceptance_fraction', np.nan)),
        'n_samples':         int(ok.sum()),
    }

    return metrics, passes


# ── Best-fit grid plot ─────────────────────────────────────────────────────────
def make_bestfit_grid(png_paths, out_path, n_cols=3):
    """
    Tile a list of PNG file paths into a single grid figure.
    Each cell shows the full individual banana plot image unchanged.
    """
    from matplotlib.image import imread as mpl_imread

    n = len(png_paths)
    if n == 0:
        print("  No best-fit plots to grid.")
        return

    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    # Base each cell on the aspect ratio of the first image
    img0 = mpl_imread(png_paths[0])
    h0, w0 = img0.shape[:2]
    cell_w = 6.0
    cell_h = cell_w * h0 / w0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        squeeze=False,
    )

    for idx, ax_row in enumerate(axes):
        for jdx, ax in enumerate(ax_row):
            k = idx * n_cols + jdx
            if k < len(png_paths):
                img = mpl_imread(png_paths[k])
                ax.imshow(img)
            ax.axis('off')

    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Grid saved: {out_path}")


# ── Best-fit TXT report ────────────────────────────────────────────────────────
def write_bestfit_txt(best_metrics, out_path):
    lines = []
    lines.append("=" * 100)
    lines.append("APOKASC BEST-FIT VALIDATION STARS")
    lines.append(
        f"Criterion: APOKASC (mh_obs, int_age) falls within the IQR (25th–75th pct) "
        f"of the banana at obs [M/H] ± {BESTFIT_FEH_WINDOW} dex"
    )
    lines.append(
        "Note: stars suitable for inclusion in the paper should have APOKASC metallicity\n"
        "      and APOKASC age intersecting within the banana (ideally within the IQR)."
    )
    lines.append("=" * 100)
    lines.append("")

    col_header = (
        f"{'Star ID':<32}  {'Class':>5}  "
        f"{'Teff':>6}  {'e_Teff':>6}  {'Logg':>6}  {'Lum':>6}  "
        f"{'[Fe/H]':>6}  {'[a/Fe]':>6}  {'Mass':>5}  "
        f"{'IntAge':>7}  {'Q25':>6}  {'Q75':>6}  "
        f"{'AccFrac':>7}  {'N_samp':>8}"
    )
    lines.append(col_header)
    lines.append("-" * 100)

    for m in best_metrics:
        e_teff_str = f"{m['e_Teff_K']:.0f}" if np.isfinite(m['e_Teff_K']) else "  —"
        mass_str   = f"{m['Mass_Msun']:.3f}" if np.isfinite(m['Mass_Msun']) else "  —  "
        acc_str    = f"{m['acceptance_frac']:.3f}" if np.isfinite(m['acceptance_frac']) else "  —  "
        lines.append(
            f"{m['star_id']:<32}  {m['stellar_class']:>5}  "
            f"{m['Teff_K']:>6.0f}  {e_teff_str:>6}  {m['Logg_cgs']:>6.3f}  "
            f"{m['Lum_logLsun']:>6.3f}  "
            f"{m['FeH_obs']:>6.3f}  {m['AlphaFe']:>6.2f}  {mass_str:>5}  "
            f"{m['IntAge_Gyr']:>7.2f}  {m['banana_q25']:>6.2f}  {m['banana_q75']:>6.2f}  "
            f"{acc_str:>7}  {m['n_samples']:>8,}"
        )

    lines.append("")
    lines.append(f"Total best-fit stars: {len(best_metrics)}")
    lines.append("=" * 100)

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Best-fit report: {out_path}")


# ── Replot all existing chains ────────────────────────────────────────────────
def replot_all_chains():
    chain_dir = 'results/apokasc/chains'
    plots_dir = 'results/apokasc/plots'
    chain_files = sorted(f for f in os.listdir(chain_dir) if f.endswith('.pkl'))
    print(f"Replotting {len(chain_files)} chains...")
    for i, fname in enumerate(chain_files):
        path = os.path.join(chain_dir, fname)
        try:
            with open(path, 'rb') as f:
                res = pickle.load(f)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue
        flat_samples = res.get('flat_samples', res.get('output'))
        blobs_df     = res.get('blobs_df',     res.get('output'))
        if flat_samples is None or blobs_df is None:
            print(f"  SKIP {fname}: missing flat_samples or blobs_df")
            continue
        int_age   = float(res.get('int_age', np.nan))
        aux_value = int_age if np.isfinite(int_age) else np.nan
        save_banana_plot(
            res['star_id'], flat_samples, blobs_df,
            res['teff_obs'], res['lum_obs'], res['logg_obs'], res['mh_obs'],
            aux_value, res['stellar_class'],
            alpha_fe=res.get('alpha_fe', 0.0),
            int_mass=res.get('int_mass', float('nan')),
            e_teff=res.get('e_teff', float('nan')),
            acc=res.get('acceptance_fraction', float('nan')),
            e_mh_obs=res.get('e_mh_obs', 0.1),
            e_int_age=res.get('e_int_age', float('nan')),
            out_dir=plots_dir,
        )
        print(f"  [{i+1}/{len(chain_files)}] {res['star_id']}")
    print("Replot done.")


# ── Best-fits only: pre-filter, plot, grid, report ────────────────────────────
def replot_best_fits():
    chain_dir   = 'results/apokasc/chains'
    bestfit_dir = 'results/apokasc/plots/best_fits'
    os.makedirs(bestfit_dir, exist_ok=True)

    chain_files = sorted(f for f in os.listdir(chain_dir) if f.endswith('.pkl'))
    print(f"Scoring {len(chain_files)} chains against best-fit criteria...")

    # ── Step 1: pre-filter to best-fit chains ─────────────────────────────────
    best_chains  = []   # (res, metrics) for passing stars
    for fname in chain_files:
        path = os.path.join(chain_dir, fname)
        try:
            with open(path, 'rb') as f:
                res = pickle.load(f)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue
        metrics, passes = score_chain_bestfit(res)
        if passes:
            best_chains.append((res, metrics))

    print(f"  {len(best_chains)} / {len(chain_files)} chains pass best-fit criteria.")

    if not best_chains:
        print("  No best-fit stars found — nothing to plot.")
        return

    # Sort by APOKASC age error ascending
    best_chains.sort(key=lambda x: x[1]['IntAge_Gyr'])

    # ── Step 2: make individual plots for best-fit stars only ─────────────────
    png_paths = []
    for i, (res, metrics) in enumerate(best_chains):
        int_age   = float(res.get('int_age', np.nan))
        aux_value = int_age if np.isfinite(int_age) else np.nan
        flat_samples = res.get('flat_samples', res.get('output'))
        blobs_df     = res.get('blobs_df',     res.get('output'))
        png_path = save_banana_plot(
            res['star_id'], flat_samples, blobs_df,
            res['teff_obs'], res['lum_obs'], res['logg_obs'], res['mh_obs'],
            aux_value, res['stellar_class'],
            alpha_fe=res.get('alpha_fe', 0.0),
            int_mass=res.get('int_mass', float('nan')),
            e_teff=res.get('e_teff', float('nan')),
            acc=res.get('acceptance_fraction', float('nan')),
            e_mh_obs=res.get('e_mh_obs', 0.1),
            e_int_age=res.get('e_int_age', float('nan')),
            out_dir=bestfit_dir,
        )
        if png_path:
            png_paths.append(png_path)
        print(f"  [{i+1}/{len(best_chains)}] {res['star_id']}  "
              f"(IntAge={metrics['IntAge_Gyr']:.2f}  Q25={metrics['banana_q25']:.2f}  "
              f"Q75={metrics['banana_q75']:.2f})")

    # ── Step 3: grid of all best-fit plots ────────────────────────────────────
    grid_path = 'results/apokasc/plots/best_fits_grid.png'
    make_bestfit_grid(png_paths, grid_path, n_cols=3)

    # ── Step 4: fundamental parameters report ─────────────────────────────────
    txt_path = 'results/apokasc/best_fits_report.txt'
    write_bestfit_txt([m for _, m in best_chains], txt_path)

    print(f"\nBest-fit outputs:")
    print(f"  Individual plots : {bestfit_dir}/")
    print(f"  Grid             : {grid_path}")
    print(f"  Report           : {txt_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APOKASC banana MCMC')
    parser.add_argument('--star_id',    type=str,  default=None)
    parser.add_argument('--star_index', type=int,  default=None)
    parser.add_argument('--combine',    action='store_true',
                        help='Combine all completed chains into summary outputs')
    parser.add_argument('--replot',     action='store_true',
                        help='Regenerate plots for all existing chains without rerunning MCMC')
    parser.add_argument('--best',       action='store_true',
                        help='Plot only best-fit chains, produce grid and report')
    parser.add_argument('--n_walkers',  type=int,  default=32)
    parser.add_argument('--n_burnin',   type=int,  default=300)
    parser.add_argument('--n_iter',     type=int,  default=1000)
    args = parser.parse_args()
    N_WALKERS = args.n_walkers
    N_BURNIN  = args.n_burnin
    N_ITER    = args.n_iter

    if args.combine:
        combine_chains()
        sys.exit(0)

    if args.replot:
        replot_all_chains()
        sys.exit(0)

    if args.best:
        replot_best_fits()
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
            if safe_id in already_done:
                print(f"[{i+1:3d}/{n_total}] {star_row['star_id']}  — already done, skipping")
                continue
            print(f"[{i+1:3d}/{n_total}]", end='')
            run_star(star_row, jtgrid, bounds)

        print("\nAll stars done. Run with --combine to generate summary outputs.")
