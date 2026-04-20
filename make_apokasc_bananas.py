import os
import sys
import gc
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
import emcee
import kiauhoku as kh

warnings.filterwarnings('ignore')

os.makedirs('results/apokasc/chains', exist_ok=True)
os.makedirs('results/apokasc/plots', exist_ok=True)

# ── MCMC settings ─────────────────────────────────────────────────────────────
N_WALKERS = 32
N_BURNIN = 300
N_ITER = 1000
NDIM = 3

# ── Observational uncertainties / priors ─────────────────────────────────────
LUM_SIGMA_DEFAULT = 0.10   # dex
LOGG_SIGMA_DEFAULT = 0.05  # dex
MH_SIGMA_DEFAULT = 0.05    # dex
AGE_UNIVERSE = 13.8        # Gyr

# Store only tiny numeric blobs instead of full interpolated star objects.
BLOBS_DTYPE = np.dtype([
    ('teff', 'f8'),
    ('lum', 'f8'),
    ('logg', 'f8'),
    ('met', 'f8'),
    ('mass', 'f8'),
    ('age', 'f8'),
])

# ── Physical relations ────────────────────────────────────────────────────────
Y_PRIMORDIAL = 0.2485
DYDZ = 1.4
Z_SOLAR = 0.0134
LOGG_SUN = 4.4374      # cgs
TEFF_SUN = 5777.0      # K


def compute_y(feh):
    z = Z_SOLAR * 10**feh
    return float(np.clip(Y_PRIMORDIAL + DYDZ * z, 0.24, 0.32))


def compute_ML(feh, lo, hi):
    return float(np.clip(0.02 * feh + 1.94, lo + 1e-6, hi - 1e-6))


def compute_lum(teff, logg, mass):
    """log(L/Lsun) from Stefan-Boltzmann + fundamental relation."""
    return np.log10(mass) + 4.0 * np.log10(teff / TEFF_SUN) - (logg - LOGG_SUN)


def classify_star(logg):
    if np.isnan(logg):
        return 'unknown'
    return 'RGB' if (logg < 2.2 or logg > 3.0) else 'clump'


def first_present(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def get_row_sigma(star_row, candidates, default_value):
    for name in candidates:
        if name in star_row.index:
            value = pd.to_numeric(star_row[name], errors='coerce')
            if np.isfinite(value) and value > 0:
                return float(value)
    return float(default_value)


def get_star_value(star, preferred_name, fallback=np.nan):
    if preferred_name in star.index:
        return float(star[preferred_name])
    return float(fallback)


def run_kiauhoku_coarse_fit(star_row, interp, bounds):
    """
    Use Kiauhoku's gridsearch_fit as a coarse initializer for the walkers.
    This is wrapped in a conservative try/except because the exact installed
    Kiauhoku signature can vary by environment.
    """
    teff_obs = float(star_row['teff_obs'])
    lum_obs = float(star_row['lum_obs'])
    logg_obs = float(star_row['logg_obs'])
    mh_obs = float(star_row['mh_obs'])
    e_teff = float(star_row['e_teff'])
    e_logg = float(star_row['e_logg'])
    e_mh = float(star_row['e_mh'])

    star = {
        'teff': teff_obs,
        'lum': lum_obs,
        'met': mh_obs,
        'logg': logg_obs,
        'alpha_fe': 0.0,
        'initial_he': compute_y(mh_obs),
    }
    scale = {
        'teff': max(e_teff, 50.0),
        'lum': LUM_SIGMA_DEFAULT,
        'met': max(e_mh, 0.02),
        'logg': max(e_logg, 0.02),
        'alpha_fe': 0.05,
        'initial_he': 0.01,
    }

    bounds_list = [
        bounds['initial_mass'],
        bounds['initial_met'],
        bounds['alpha_fe'],
        bounds['initial_he'],
        bounds['mixing_length'],
    ]

    attempts = [
        dict(scale=scale, tol=1e-2, maxiter=100, bounds=bounds_list),
        dict(scale=scale, tol=1e-2, maxiter=100),
        dict(scale=scale),
        {},
    ]

    last_error = None
    for kwargs in attempts:
        model, fit = interp.gridsearch_fit(star, **kwargs)
        fit_success = bool(getattr(fit, 'success', True))
        if not fit_success:
            last_error = RuntimeError(f"gridsearch_fit returned success={fit_success}")
            continue
        if model is None:
            last_error = RuntimeError('gridsearch_fit returned model=None')
            continue
        return {
            'initial_mass': get_star_value(model, 'initial_mass'),
            'initial_met': get_star_value(model, 'initial_met'),
            'eep': get_star_value(model, 'eep'),
            'fit': fit,
            'model': model,
            'kwargs': kwargs,
        }

    print(f"    Kiauhoku coarse fit unavailable; falling back to broad init ({last_error})")
    return None


def banana_log_prob(
    pos,
    interp,
    teff_obs,
    lum_obs,
    logg_obs,
    mh_obs,
    teff_sigma,
    lum_sigma,
    logg_sigma,
    mh_sigma,
    bounds,
    ml_lo,
    ml_hi,
):
    initial_mass, initial_met, eep = pos

    mass_lo, mass_hi = bounds['initial_mass']
    met_lo, met_hi = bounds['initial_met']
    eep_lo, eep_hi = bounds['eep']

    if not (mass_lo < initial_mass < mass_hi and
            met_lo < initial_met < met_hi and
            eep_lo < eep < eep_hi):
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    alpha_fe = 0.0
    initial_he = compute_y(initial_met)
    mixing_length = compute_ML(initial_met, ml_lo, ml_hi)

    he_lo, he_hi = bounds['initial_he']
    if not (he_lo <= initial_he <= he_hi):
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    full_index = (initial_mass, initial_met, alpha_fe,
                  initial_he, mixing_length, eep)
    try:
        star = interp.get_star_eep(full_index)
    except Exception:
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if star is None or star.isna().any():
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    teff_model = get_star_value(star, 'teff')
    lum_model = get_star_value(star, 'lum')
    logg_model = get_star_value(star, 'logg')
    met_model = get_star_value(star, 'met', fallback=initial_met)
    mass_model = get_star_value(star, 'mass', fallback=initial_mass)
    if 'age' in star.index:
        age_model = float(star['age'])
    elif 'Age(Gyr)' in star.index:
        age_model = float(star['Age(Gyr)'])
    else:
        age_model = np.nan

    if not np.isfinite(age_model) or age_model <= 0 or age_model > AGE_UNIVERSE:
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if not np.isfinite(logg_model) or not np.isfinite(met_model):
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    log_prob = (
        -0.5 * ((teff_obs - teff_model) / teff_sigma) ** 2
        -0.5 * ((lum_obs - lum_model) / lum_sigma) ** 2
        -0.5 * ((logg_obs - logg_model) / logg_sigma) ** 2
        -0.5 * ((mh_obs - met_model) / mh_sigma) ** 2
    )
    return log_prob, teff_model, lum_model, logg_model, met_model, mass_model, age_model


# ── Grid loading ──────────────────────────────────────────────────────────────
def load_grid():
    print("Loading JT2017t12 grid...")
    qstring = '201 <= eep'
    jtgrid = kh.load_eep_grid("JT2017t12").query(qstring)
    jtgrid['mass'] = jtgrid['Mass(Msun)']
    jtgrid['teff'] = 10**jtgrid['Log Teff(K)']
    jtgrid['lum'] = jtgrid['L/Lsun']
    jtgrid['logg'] = jtgrid['Log G(cgs)'] if 'Log G(cgs)' in jtgrid.columns else jtgrid['logg']
    jtgrid['met'] = jtgrid.index.get_level_values('initial_met')
    jtgrid['initial_he'] = jtgrid.index.get_level_values('initial_he')
    jtgrid['mixing_length'] = jtgrid.index.get_level_values('mixing_length')
    jtgrid['alpha_fe'] = jtgrid.index.get_level_values('alpha_fe')
    jtgrid['age'] = jtgrid['Age(Gyr)']
    jtgrid.set_name('jtgrid')
    jtgrid = jtgrid.to_interpolator()
    print("Grid loaded.\n")

    bounds = {
        name: (float(vals.min()), float(vals.max()))
        for name, vals in zip(jtgrid.index_names, jtgrid.index_columns)
    }
    return jtgrid, bounds


# ── Initial walker positions ──────────────────────────────────────────────────
def make_initial_positions(star_row, bounds, n_walkers, rng, coarse_fit=None):
    mass_lo, mass_hi = bounds['initial_mass']
    met_lo, met_hi = bounds['initial_met']
    eep_lo, eep_hi = bounds['eep']

    if coarse_fit is not None:
        c_mass = float(np.clip(coarse_fit['initial_mass'], mass_lo, mass_hi))
        c_met = float(np.clip(coarse_fit['initial_met'], met_lo, met_hi))
        if np.isfinite(coarse_fit['eep']):
            c_eep = float(np.clip(coarse_fit['eep'], eep_lo, eep_hi))
        else:
            c_eep = 0.5 * (eep_lo + eep_hi)

        w_mass = max(0.02 * (mass_hi - mass_lo), 0.03)
        w_met = max(0.02 * (met_hi - met_lo), 0.02)
        w_eep = max(0.02 * (eep_hi - eep_lo), 3.0)
    else:
        c_mass = 0.5 * (mass_lo + mass_hi)
        c_met = float(star_row['mh_obs'])
        c_eep = 0.5 * (eep_lo + eep_hi)

        w_mass = 0.05 * (mass_hi - mass_lo)
        w_met = 0.05 * (met_hi - met_lo)
        w_eep = 0.05 * (eep_hi - eep_lo)

    pos = np.column_stack([
        np.clip(rng.normal(c_mass, w_mass, n_walkers), mass_lo, mass_hi),
        np.clip(rng.normal(c_met, w_met, n_walkers), met_lo, met_hi),
        np.clip(rng.normal(c_eep, w_eep, n_walkers), eep_lo, eep_hi),
    ])

    # jitter any pathological all-identical walkers
    pos += rng.normal(0.0, [1e-4, 1e-4, 1e-3], size=pos.shape)
    pos[:, 0] = np.clip(pos[:, 0], mass_lo + 1e-6, mass_hi - 1e-6)
    pos[:, 1] = np.clip(pos[:, 1], met_lo + 1e-6, met_hi - 1e-6)
    pos[:, 2] = np.clip(pos[:, 2], eep_lo + 1e-6, eep_hi - 1e-6)
    return pos


# ── Per-star run ──────────────────────────────────────────────────────────────
def run_star(star_row, jtgrid, bounds, n_walkers=N_WALKERS,
             n_burnin=N_BURNIN, n_iter=N_ITER):
    star_id = str(star_row['star_id'])
    safe_id = star_id.replace('/', '_')

    chain_path = f'results/apokasc/chains/{safe_id}.pkl'
    if os.path.exists(chain_path):
        print(f"  {star_id}  — already done, skipping")
        return None

    teff_obs = float(star_row['teff_obs'])
    lum_obs = float(star_row['lum_obs'])
    logg_obs = float(star_row['logg_obs'])
    mh_obs = float(star_row['mh_obs'])
    int_age = float(star_row['int_age'])
    int_mass = float(star_row['int_mass'])
    e_teff = float(star_row['e_teff'])
    e_logg = float(star_row['e_logg'])
    e_mh = float(star_row['e_mh'])

    stellar_class = classify_star(logg_obs)
    ml_lo = bounds['mixing_length'][0]
    ml_hi = bounds['mixing_length'][1]

    print(
        f"  {star_id}  T={teff_obs:.0f}K  logg={logg_obs:.2f}  [Fe/H]={mh_obs:.2f}"
        f"  IntAge={int_age:.1f}Gyr  IntMass={int_mass:.2f}Msun  class={stellar_class}"
    )

    seed = int(np.frombuffer(star_id.encode('utf-8'), dtype=np.uint8).sum() + 1009 * len(star_id))
    rng = np.random.default_rng(seed)

    coarse_fit = run_kiauhoku_coarse_fit(star_row, jtgrid, bounds)
    if coarse_fit is not None:
        print(
            f"    coarse fit → M={coarse_fit['initial_mass']:.3f}, "
            f"[Fe/H]={coarse_fit['initial_met']:.3f}, eep={coarse_fit['eep']:.2f}"
        )

    p0 = make_initial_positions(star_row, bounds, n_walkers, rng, coarse_fit=coarse_fit)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        NDIM,
        banana_log_prob,
        blobs_dtype=BLOBS_DTYPE,
        kwargs=dict(
            interp=jtgrid,
            teff_obs=teff_obs,
            lum_obs=lum_obs,
            logg_obs=logg_obs,
            mh_obs=mh_obs,
            teff_sigma=e_teff,
            lum_sigma=LUM_SIGMA_DEFAULT,
            logg_sigma=e_logg,
            mh_sigma=e_mh,
            bounds=bounds,
            ml_lo=ml_lo,
            ml_hi=ml_hi,
        )
    )

    # Burn-in separately, then reset so burn-in is not retained in memory.
    state = sampler.run_mcmc(p0, n_burnin, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, n_iter, progress=False)

    flat_samples = sampler.get_chain(flat=True)
    flat_blobs = sampler.get_blobs(flat=True)
    log_prob = sampler.get_log_prob(flat=True)
    acc = float(np.mean(sampler.acceptance_fraction))

    output = pd.DataFrame(flat_samples, columns=['initial_mass', 'initial_met', 'eep'])
    output['alpha_fe'] = 0.0
    output['initial_he'] = output['initial_met'].apply(compute_y)
    output['mixing_length'] = output['initial_met'].apply(lambda feh: compute_ML(feh, ml_lo, ml_hi))
    output['log_prob'] = log_prob

    if flat_blobs is not None and len(flat_blobs) == len(output):
        output['teff'] = flat_blobs['teff']
        output['lum'] = flat_blobs['lum']
        output['logg'] = flat_blobs['logg']
        output['met'] = flat_blobs['met']
        output['mass'] = flat_blobs['mass']
        output['age'] = flat_blobs['age']
    else:
        output['teff'] = np.nan
        output['lum'] = np.nan
        output['logg'] = np.nan
        output['met'] = np.nan
        output['mass'] = np.nan
        output['age'] = np.nan

    output['teff_obs'] = teff_obs
    output['lum_obs'] = lum_obs
    output['logg_obs'] = logg_obs
    output['mh_obs'] = mh_obs
    output['stellar_class'] = stellar_class

    valid = (
        np.isfinite(output['age'].values)
        & np.isfinite(output['log_prob'].values)
        & (output['age'].values > 0)
        & (output['age'].values <= AGE_UNIVERSE)
    )
    if valid.any():
        posterior_age = output.loc[valid, 'age'].values
        posterior_age_med = float(np.median(posterior_age))
        posterior_age_lo = float(np.percentile(posterior_age, 16))
        posterior_age_hi = float(np.percentile(posterior_age, 84))
    else:
        posterior_age_med = np.nan
        posterior_age_lo = np.nan
        posterior_age_hi = np.nan

    result = {
        'star_id': star_id,
        'stellar_class': stellar_class,
        'teff_obs': teff_obs,
        'lum_obs': lum_obs,
        'logg_obs': logg_obs,
        'mh_obs': mh_obs,
        'int_age': int_age,
        'int_mass': int_mass,
        'e_teff': e_teff,
        'e_logg': e_logg,
        'e_mh': e_mh,
        'coarse_fit': coarse_fit,
        'posterior_age_median': posterior_age_med,
        'posterior_age_lo': posterior_age_lo,
        'posterior_age_hi': posterior_age_hi,
        'output': output,
        'acceptance_fraction': acc,
        'n_walkers': n_walkers,
        'n_burnin': n_burnin,
        'n_iter': n_iter,
    }
    with open(chain_path, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    del sampler, flat_samples, flat_blobs, log_prob, output, result
    gc.collect()

    print(
        f"    → saved  (acc={acc:.2f})"
        f"  posterior age={posterior_age_med:.2f}"
        f" +{posterior_age_hi - posterior_age_med:.2f}"
        f" -{posterior_age_med - posterior_age_lo:.2f} Gyr"
        if np.isfinite(posterior_age_med)
        else f"    → saved  (acc={acc:.2f})  posterior age unavailable"
    )
    return True


# ── Load APOKASC catalogue ────────────────────────────────────────────────────
def load_apokasc(path='MeridithRomanApokascCalibLtest5ns3L.out'):
    raw = pd.read_csv(path, sep=r'\s+')

    raw = raw.rename(columns={
        '2MASSID': 'star_id',
        'Teff': 'teff_obs',
        'Logg': 'logg_obs',
        'Fe/H': 'mh_obs',
        'Teff_err': 'e_teff',
        'IntAge': 'int_age',
        'IntMass': 'int_mass',
        'C/N': 'cn_class',
    })

    logg_err_col = first_present(raw.columns, ['e_logg', 'logg_err', 'Logg_err', 'Logg_Err', 'e_Logg'])
    mh_err_col = first_present(raw.columns, ['e_mh', 'mh_err', 'Fe/H_err', 'FeH_err', '[Fe/H]_err', 'e_Fe/H'])
    if logg_err_col is not None and logg_err_col != 'e_logg':
        raw = raw.rename(columns={logg_err_col: 'e_logg'})
    if mh_err_col is not None and mh_err_col != 'e_mh':
        raw = raw.rename(columns={mh_err_col: 'e_mh'})

    if 'e_logg' not in raw.columns:
        raw['e_logg'] = LOGG_SIGMA_DEFAULT
    if 'e_mh' not in raw.columns:
        raw['e_mh'] = MH_SIGMA_DEFAULT
    if 'e_teff' not in raw.columns:
        raw['e_teff'] = 100.0

    bad = (
        (raw['int_age'] <= 0)
        | (raw['int_age'] > AGE_UNIVERSE)
        | (raw['int_mass'] <= 0)
        | (raw['teff_obs'] <= 0)
        | (raw['cn_class'] != 'RGB')
    )

    raw = raw[~bad].copy()

    # Keep using the APOKASC/seismic proxy luminosity already present in this
    # workflow, but the likelihood now also conditions directly on logg and [M/H].
    raw['lum_obs'] = compute_lum(raw['teff_obs'], raw['logg_obs'], raw['int_mass'])

    raw = raw.dropna(subset=['star_id', 'teff_obs', 'lum_obs', 'logg_obs', 'mh_obs', 'int_age'])
    raw['e_teff'] = raw['e_teff'].apply(lambda x: x if np.isfinite(x) and x > 0 else 100.0)
    raw['e_logg'] = raw['e_logg'].apply(lambda x: x if np.isfinite(x) and x > 0 else LOGG_SIGMA_DEFAULT)
    raw['e_mh'] = raw['e_mh'].apply(lambda x: x if np.isfinite(x) and x > 0 else MH_SIGMA_DEFAULT)
    raw = raw.reset_index(drop=True)
    print(f"APOKASC catalogue: {len(raw)} stars after filtering\n")
    return raw


# ── Combine + plot ────────────────────────────────────────────────────────────
def combine_and_plot():
    chain_dir = 'results/apokasc/chains'
    chain_files = sorted(f for f in os.listdir(chain_dir) if f.endswith('.pkl'))
    print(f"Combining {len(chain_files)} chains...")

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    our_ages, our_lo, our_hi, seismic_ages, mh_vals = [], [], [], [], []

    for fname in chain_files:
        with open(os.path.join(chain_dir, fname), 'rb') as f:
            res = pickle.load(f)

        output = res['output']
        int_age = res['int_age']
        mh_obs = res['mh_obs']

        age_col = 'age' if 'age' in output.columns else 'Age(Gyr)'
        if age_col not in output.columns:
            continue

        age = output[age_col].values
        logp = output['log_prob'].values if 'log_prob' in output.columns else np.full(len(output), np.nan)
        ok = np.isfinite(age) & np.isfinite(logp) & (age > 0) & (age <= AGE_UNIVERSE)
        if ok.sum() < 50:
            continue

        posterior_age = age[ok]
        med = np.median(posterior_age)
        lo = np.percentile(posterior_age, 16)
        hi = np.percentile(posterior_age, 84)

        our_ages.append(med)
        our_lo.append(med - lo)
        our_hi.append(hi - med)
        seismic_ages.append(int_age)
        mh_vals.append(mh_obs)

    if not our_ages:
        print("No valid stars to plot.")
        return

    our_ages = np.array(our_ages)
    seismic_ages = np.array(seismic_ages)
    mh_vals = np.array(mh_vals)
    vmin, vmax = -1.1, 0.45
    cmap = plt.cm.RdYlBu_r
    cols = [cmap((m - vmin) / (vmax - vmin)) for m in mh_vals]
    age_max = max(our_ages.max() + max(our_hi), seismic_ages.max()) * 1.1

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for i in range(len(our_ages)):
        ax.errorbar(
            seismic_ages[i], our_ages[i],
            yerr=[[our_lo[i]], [our_hi[i]]],
            fmt='o', color=cols[i], ms=5, lw=1.0,
            capsize=2.0, ecolor=cols[i], zorder=3
        )
    lim = (0, age_max)
    ax.plot(lim, lim, 'k--', lw=1.0, alpha=0.5)
    ax.set_xlabel('Asteroseismic age (Gyr)', fontsize=11)
    ax.set_ylabel('This work — posterior age (Gyr)', fontsize=11)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='obs [M/H]')
    fig.savefig('results/apokasc/age_comparison.png')
    fig.savefig('results/apokasc/age_comparison.pdf')
    plt.close(fig)

    residuals = our_ages - seismic_ages
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i in range(len(our_ages)):
        ax.errorbar(
            seismic_ages[i], residuals[i],
            yerr=[[our_lo[i]], [our_hi[i]]],
            fmt='o', color=cols[i], ms=5, lw=1.0,
            capsize=2.0, ecolor=cols[i], zorder=3
        )
    ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.5)
    ax.axhline(np.mean(residuals), color='tomato', lw=1.2,
               label=f'Mean offset: {np.mean(residuals):.1f} Gyr')
    ax.set_xlabel('Asteroseismic age (Gyr)', fontsize=11)
    ax.set_ylabel('This work − asteroseismic (Gyr)', fontsize=11)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(fontsize=9)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='obs [M/H]')
    fig.savefig('results/apokasc/residual_vs_seismic_age.png')
    fig.savefig('results/apokasc/residual_vs_seismic_age.pdf')
    plt.close(fig)

    print(f"\n{len(our_ages)} stars plotted.")
    print(f"Mean offset (this work − seismic): {np.mean(residuals):.2f} Gyr")
    print(f"Std of residuals: {np.std(residuals):.2f} Gyr")
    print("Saved: results/apokasc/age_comparison.png/.pdf")
    print("Saved: results/apokasc/residual_vs_seismic_age.png/.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='APOKASC banana MCMC')
    parser.add_argument('--star_index', type=int, default=None)
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--n_walkers', type=int, default=N_WALKERS)
    parser.add_argument('--n_burnin', type=int, default=N_BURNIN)
    parser.add_argument('--n_iter', type=int, default=N_ITER)
    args = parser.parse_args()

    stars = load_apokasc()

    if args.combine:
        combine_and_plot()
        sys.exit(0)

    jtgrid, bounds = load_grid()

    if args.star_index is not None:
        if args.star_index < 0 or args.star_index >= len(stars):
            print(f"ERROR: star_index {args.star_index} out of range (valid 0..{len(stars)-1}).")
            sys.exit(1)
        run_star(
            stars.iloc[args.star_index],
            jtgrid,
            bounds,
            n_walkers=args.n_walkers,
            n_burnin=args.n_burnin,
            n_iter=args.n_iter,
        )
    else:
        n = len(stars)
        for i, (_, row) in enumerate(stars.iterrows()):
            print(f"[{i+1:4d}/{n}]", end=' ')
            run_star(
                row,
                jtgrid,
                bounds,
                n_walkers=args.n_walkers,
                n_burnin=args.n_burnin,
                n_iter=args.n_iter,
            )
        print("\nAll done. Run with --combine to generate plots.")
