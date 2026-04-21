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

    if not (mass_lo < initial_mass < mass_hi and
            met_lo < initial_met < met_hi and
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
    jtgrid.set_name('jtgrid')
    jtgrid = jtgrid.to_interpolator()
    print("Grid loaded.\n")

    bounds = {
        name: (float(vals.min()), float(vals.max()))
        for name, vals in zip(jtgrid.index_names, jtgrid.index_columns)
    }
    return jtgrid, bounds


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
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    safe_id = star_id.replace('/', '_')

    class_color = {'RGB': 'steelblue', 'clump': 'seagreen', 'unknown': 'grey'}
    c = class_color.get(stellar_class, 'grey')

    age_col = 'age' if 'age' in blobs_df.columns else 'Age(Gyr)'
    if age_col not in blobs_df.columns:
        plt.close(fig)
        return

    feh = flat_samples['initial_met'].values
    age = blobs_df[age_col].values
    mask = np.isfinite(feh) & np.isfinite(age)
    feh, age = feh[mask], age[mask]

    # ── Left: 2D density banana ───────────────────────────────────────────────
    ax = axes[0]
    if len(feh) > 10:
        h = ax.hexbin(feh, age, gridsize=30, cmap='YlOrRd',
                      mincnt=1, linewidths=0.2)
        plt.colorbar(h, ax=ax, label='Sample count')
    ax.axvline(mh_obs, color='k', lw=1.2, ls='--',
               label=f'obs [M/H]={mh_obs:.2f}')
    ax.set_xlabel('[Fe/H]', fontsize=11)
    ax.set_ylabel('Age (Gyr)', fontsize=11)
    ax.set_title(f'Banana: p(age, [Fe/H])  [{stellar_class}]', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    # ── Middle: median banana curve with 1σ band ──────────────────────────────
    ax = axes[1]
    feh_bins = np.linspace(feh.min(), feh.max(), 30)
    med_ages, lo_ages, hi_ages, feh_mids = [], [], [], []
    for j in range(len(feh_bins) - 1):
        mask_bin = (feh >= feh_bins[j]) & (feh < feh_bins[j + 1])
        if mask_bin.sum() > 5:
            ages_bin = age[mask_bin]
            med_ages.append(np.median(ages_bin))
            lo_ages.append(np.percentile(ages_bin, 16))
            hi_ages.append(np.percentile(ages_bin, 84))
            feh_mids.append((feh_bins[j] + feh_bins[j + 1]) / 2)
    if med_ages:
        feh_mids = np.array(feh_mids)
        ax.fill_between(feh_mids, lo_ages, hi_ages,
                        color=c, alpha=0.3, label='16–84th pct')
        ax.plot(feh_mids, med_ages, '-', color=c, lw=2, label='Median')
    ax.axvline(mh_obs, color='k', lw=1.2, ls='--')
    ax.set_xlabel('[Fe/H]', fontsize=11)
    ax.set_ylabel('Age (Gyr)', fontsize=11)
    ax.set_title('Median ± 1σ banana', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    n_eff = mask.sum()
    fig.suptitle(
        f"{star_id}  [{stellar_class}]\n"
        f"Teff={teff_obs:.0f} K   logg={logg_obs:.2f}   lum={lum_obs:.2f}   "
        f"obs[M/H]={mh_obs:.2f}   IntAge={aux_value:.3f}   "
        f"N_samples={n_eff:,}",
        fontsize=10, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(f'results/apokasc/plots/{safe_id}.png', dpi=110, bbox_inches='tight')
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

    stars_to_run = load_apokasc()

    if args.combine:
        combine_chains()
        sys.exit(0)

    jtgrid, bounds = load_grid()

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
