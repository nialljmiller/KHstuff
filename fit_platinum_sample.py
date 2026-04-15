'''
fit_platinum_sample.py

Batch fit all stars in platinum_sample_flame.fits using the JT2017t12 kiauhoku grid.

Outputs
-------
results/fit_results.csv          — table of best-fit parameters for all stars
results/diagnostics/star_N.png  — per-star diagnostic plot (includes grid overlay
                                   and optimizer path)
results/summary_plots.png        — summary plots across the full sample
                                   (boundary hits shown separately)
'''

import os
import types
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from astropy.io import fits
import warnings
import kiauhoku as kh

warnings.filterwarnings('ignore')

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs('results/diagnostics', exist_ok=True)

# ── Load the grid ─────────────────────────────────────────────────────────────
print("Loading JT2017t12 grid...")
qstring = '201 <= eep'
evolve_met = False

jtgrid = kh.load_eep_grid("JT2017t12").query(qstring)
jtgrid['mass']          = jtgrid['Mass(Msun)']
jtgrid['teff']          = 10**jtgrid['Log Teff(K)']
jtgrid['lum']           = jtgrid['L/Lsun']
jtgrid['met']           = jtgrid.index.get_level_values('initial_met')
jtgrid['initial_he']    = jtgrid.index.get_level_values('initial_he')
jtgrid['mixing_length'] = jtgrid.index.get_level_values('mixing_length')
jtgrid['alpha_fe']      = jtgrid.index.get_level_values('alpha_fe')
jtgrid['age']           = jtgrid['Age(Gyr)']
jtgrid.set_name('jtgrid')

# ── Cache raw grid data for overlay plots (before converting to interpolator) ─
raw_grid = jtgrid.copy()
raw_grid_teff = raw_grid['teff'].values
raw_grid_lum  = raw_grid['lum'].values
raw_grid_logg = raw_grid['logg'].values
raw_grid_met  = raw_grid['met'].values
# Compute convex-hull-like bounds for shaded overlay on HR/Kiel
# (just take the actual extent of the grid in observable space)
GRID_OBS_TEFF_MIN = float(np.nanpercentile(raw_grid_teff, 0.5))
GRID_OBS_TEFF_MAX = float(np.nanpercentile(raw_grid_teff, 99.5))
GRID_OBS_LUM_MIN  = float(np.nanpercentile(raw_grid_lum, 0.5))
GRID_OBS_LUM_MAX  = float(np.nanpercentile(raw_grid_lum, 99.5))
GRID_OBS_LOGG_MIN = float(np.nanpercentile(raw_grid_logg, 0.5))
GRID_OBS_LOGG_MAX = float(np.nanpercentile(raw_grid_logg, 99.5))

jtgrid = jtgrid.to_interpolator()
print("Grid loaded.\n")

# ── Grid index bounds ─────────────────────────────────────────────────────────
# Used for boundary-hit detection and soft penalty
GRID_BOUNDS = {}
for name, vals in zip(jtgrid.index_names, jtgrid.index_columns):
    GRID_BOUNDS[name] = (float(vals.min()), float(vals.max()))

# Boundary detection: flag a fit if any index param is within this fraction
# of the grid edge
BOUNDARY_FLAG_MARGIN_FRAC  = 0.01   # 1% of range = "at the boundary"

# Soft penalty: starts at this fraction from the edge, grows quadratically
BOUNDARY_PENALTY_MARGIN_FRAC = 0.05  # 5% of range
BOUNDARY_PENALTY_WEIGHT      = 2.0   # multiplied into the loss

# ── Boundary detection helper ─────────────────────────────────────────────────
def is_boundary_hit(fit_index, interp, margin_frac=BOUNDARY_FLAG_MARGIN_FRAC):
    '''Return True if any fitted index parameter is within margin_frac of a
    grid edge. Also return a list of which parameters hit the boundary.'''
    hit_params = []
    for val, ic, name in zip(fit_index, interp.index_columns, interp.index_names):
        lo, hi = float(ic.min()), float(ic.max())
        span = hi - lo
        if span == 0:
            continue
        margin = margin_frac * span
        if val <= lo + margin or val >= hi - margin:
            hit_params.append(name)
    return len(hit_params) > 0, hit_params

# ── Optimizer path log (mutable so the monkey-patched method can append) ──────
_path_log = []

# ── Monkey-patch the loss function to add boundary penalty + path logging ─────
def _mse_with_boundary_and_log(self, index, star_dict, scale=False):
    index_c = self._clamp_index(index)
    star = self.get_star_eep(index_c)

    # Log this optimizer step
    _path_log.append({
        'index': tuple(index_c),
        'teff':  float(star['teff'])  if 'teff'  in star.index and not np.isnan(star['teff'])  else np.nan,
        'lum':   float(star['lum'])   if 'lum'   in star.index and not np.isnan(star['lum'])   else np.nan,
        'logg':  float(star['logg'])  if 'logg'  in star.index and not np.isnan(star['logg'])  else np.nan,
    })

    if star.isna().any():
        return 1e30

    # Base MSE
    if scale is None or scale is False:
        sq_err = np.array([(star[l] - star_dict[l])**2 for l in star_dict])
    else:
        sq_err = np.array([((star[l] - star_dict[l]) / scale[l])**2 for l in star_dict])
    base_loss = float(np.average(sq_err))

    # Soft boundary penalty — quadratic ramp within BOUNDARY_PENALTY_MARGIN_FRAC
    penalty = 0.0
    n_dims  = len(self.index_columns)
    for val, ic in zip(index_c, self.index_columns):
        lo, hi = float(ic.min()), float(ic.max())
        span   = hi - lo
        if span == 0:
            continue
        margin = BOUNDARY_PENALTY_MARGIN_FRAC * span
        if val < lo + margin:
            penalty += ((lo + margin - val) / margin) ** 2
        if val > hi - margin:
            penalty += ((val - (hi - margin)) / margin) ** 2

    return base_loss + BOUNDARY_PENALTY_WEIGHT * penalty / max(n_dims, 1)

jtgrid._meansquarederror = types.MethodType(_mse_with_boundary_and_log, jtgrid)

# ── Grid observable ranges (for pre-filtering) ────────────────────────────────
GRID_TEFF_MIN = 3500.0
GRID_TEFF_MAX = 12000.0
GRID_LOGG_MIN = -0.5
GRID_LOGG_MAX = 4.5
GRID_MH_MIN   = -2.6
GRID_MH_MAX   = 0.6

# ── Helium-metallicity relation ───────────────────────────────────────────────
Y_PRIMORDIAL = 0.2485
DYDZ         = 1.4
Z_SOLAR      = 0.0134

def compute_initial_he(mh):
    Z = Z_SOLAR * 10**mh
    Y = Y_PRIMORDIAL + DYDZ * Z
    return np.clip(Y, 0.24, 0.32)

# ── Fitting scale ─────────────────────────────────────────────────────────────
SCALE = {
    'teff':     100.0,
    'logg':     0.10,
    'met':      0.10,
    'alpha_fe': 0.10,
    'lum':      0.10,
}

FIT_TOL = 1e-3

# ── Load the FITS catalogue ───────────────────────────────────────────────────
print("Reading platinum_sample_flame.fits...")
with fits.open('platinum_sample_flame.fits') as hdul:
    cat = hdul[1].data

n_stars = len(cat)
print(f"  {n_stars} stars to fit.\n")

# ── Containers for results ────────────────────────────────────────────────────
result_rows = []

# ── Per-star fitting loop ─────────────────────────────────────────────────────
for i, row in enumerate(cat):
    star_id   = row['sdss4_apogee_id'].strip()
    teff_obs  = float(row['teff_astra'])
    e_teff    = float(row['e_teff_astra'])
    logg_obs  = float(row['logg_astra'])
    e_logg    = float(row['e_logg_astra'])
    mh_obs    = float(row['mh_astra'])
    e_mh      = float(row['e_mh_astra'])
    alpha_obs = float(row['alpha_m_astra'])
    e_alpha   = float(row['e_alpha_m_astra'])
    lum_obs   = float(row['log_lum_lsun'])

    he_est = compute_initial_he(mh_obs)

    # Pre-check: skip stars clearly outside the grid
    out_of_grid = (
        teff_obs < GRID_TEFF_MIN or teff_obs > GRID_TEFF_MAX or
        logg_obs < GRID_LOGG_MIN or logg_obs > GRID_LOGG_MAX or
        mh_obs   < GRID_MH_MIN   or mh_obs   > GRID_MH_MAX
    )
    if out_of_grid:
        print(f"[{i+1:3d}/{n_stars}] {star_id}  SKIPPED — observables outside grid range "
              f"(Teff={teff_obs:.0f}, logg={logg_obs:.2f}, [M/H]={mh_obs:.2f})")
        res = {
            'star_id': star_id, 'ra': float(row['ra']), 'dec': float(row['dec']),
            'teff_obs': teff_obs, 'e_teff_obs': e_teff,
            'logg_obs': logg_obs, 'e_logg_obs': e_logg,
            'mh_obs': mh_obs, 'e_mh_obs': e_mh,
            'alpha_obs': alpha_obs, 'e_alpha_obs': e_alpha,
            'lum_obs': lum_obs, 'he_est': he_est,
            'fit_success': False, 'fit_loss': np.nan,
            'fit_converged': False, 'boundary_hit': False,
            'boundary_params': '',
            'skip_reason': 'out_of_grid',
        }
        for param in ['initial_mass','initial_met','initial_he','alpha_fe',
                      'mixing_length','eep','mass','teff','lum','met','logg','age']:
            res[f'fit_{param}'] = np.nan
        for param in ['age','mass','mixing_length','initial_met']:
            res[f'nn_std_{param}'] = np.nan
        res['nn_dist_best'] = np.nan
        result_rows.append(res)
        continue

    print(f"[{i+1:3d}/{n_stars}] {star_id}  Teff={teff_obs:.0f}K  logg={logg_obs:.2f}  "
          f"[M/H]={mh_obs:.2f}  lum={lum_obs:.2f}")

    star_dict = {
        'teff':       teff_obs,
        'logg':       logg_obs,
        'met':        mh_obs,
        'alpha_fe':   alpha_obs,
        'lum':        lum_obs,
        'initial_he': he_est,
    }
    scale = dict(SCALE)
    scale['initial_he'] = 0.005

    # ── Clear path log, then fit ──────────────────────────────────────────────
    _path_log.clear()
    try:
        model, fit = jtgrid.gridsearch_fit(
            star_dict, scale=scale, tol=FIT_TOL, verbose=False
        )
    except Exception as exc:
        print(f"    ERROR: {exc}")
        model, fit = None, None

    star_path = list(_path_log)   # snapshot path for this star

    # ── Nearest neighbours ────────────────────────────────────────────────────
    try:
        nn = jtgrid.nearest_match(star_dict, n=20, scale=scale)
    except Exception:
        nn = None

    # ── Boundary hit detection ────────────────────────────────────────────────
    boundary_hit   = False
    boundary_params = []
    if model is not None and fit is not None:
        fit_index = tuple(fit.x)
        boundary_hit, boundary_params = is_boundary_hit(fit_index, jtgrid)

    # ── Pack results ─────────────────────────────────────────────────────────
    res = {
        'skip_reason':     'none',
        'star_id':         star_id,
        'ra':              float(row['ra']),
        'dec':             float(row['dec']),
        'teff_obs':        teff_obs,   'e_teff_obs':  e_teff,
        'logg_obs':        logg_obs,   'e_logg_obs':  e_logg,
        'mh_obs':          mh_obs,     'e_mh_obs':    e_mh,
        'alpha_obs':       alpha_obs,  'e_alpha_obs': e_alpha,
        'lum_obs':         lum_obs,
        'he_est':          he_est,
        'fit_success':     fit.success if fit is not None else False,
        'fit_loss':        float(fit.fun) if fit is not None else np.nan,
        'fit_converged':   (getattr(fit, 'fun', np.nan) <= FIT_TOL
                            if fit is not None else False),
        'boundary_hit':    boundary_hit,
        'boundary_params': ','.join(boundary_params),
    }

    if model is not None:
        for param in ['initial_mass', 'initial_met', 'initial_he', 'alpha_fe',
                      'mixing_length', 'eep', 'mass', 'teff', 'lum', 'met',
                      'logg', 'age']:
            res[f'fit_{param}'] = float(model[param]) if param in model.index else np.nan
    else:
        for param in ['initial_mass', 'initial_met', 'initial_he', 'alpha_fe',
                      'mixing_length', 'eep', 'mass', 'teff', 'lum', 'met',
                      'logg', 'age']:
            res[f'fit_{param}'] = np.nan

    if nn is not None:
        top5 = nn.head(5)
        for param in ['age', 'mass', 'mixing_length', 'initial_met']:
            res[f'nn_std_{param}'] = float(top5[param].std()) if param in top5.columns else np.nan
        res['nn_dist_best'] = float(nn.iloc[0]['distance'])
    else:
        for param in ['age', 'mass', 'mixing_length', 'initial_met']:
            res[f'nn_std_{param}'] = np.nan
        res['nn_dist_best'] = np.nan

    result_rows.append(res)

    # ── Per-star diagnostic plot ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    title_color   = 'firebrick' if boundary_hit else 'black'
    boundary_note = (f"  ⚠ BOUNDARY HIT: {', '.join(boundary_params)}"
                     if boundary_hit else "")

    result_str = (
        f"\nLoss: {res['fit_loss']:.4f}   Converged: {res['fit_converged']}"
        f"Age: {res.get('fit_age', np.nan):.2f} Gyr   "
        f"Mass: {res.get('fit_mass', np.nan):.3f} M☉   "
        f"α_MLT: {res.get('fit_mixing_length', np.nan):.3f}   "
        f"Optimizer steps: {len(star_path)}"
    )


    fig.suptitle(f"{star_id}   [{i+1}/{n_stars}]{boundary_note}{result_str}",
                 fontsize=12, fontweight='bold', color=title_color)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.1, wspace=0.1)

    ax_hrd           = fig.add_subplot(gs[0, 0])
    ax_kiel          = fig.add_subplot(gs[0, 1])
    ax_path_hrd      = fig.add_subplot(gs[0, 2])
    ax_path_kiel     = fig.add_subplot(gs[0, 3])
    ax_hrd_zoom      = fig.add_subplot(gs[1, 0])
    ax_kiel_zoom     = fig.add_subplot(gs[1, 1])
    ax_path_hrd_zoom = fig.add_subplot(gs[1, 2])
    ax_path_kiel_zoom= fig.add_subplot(gs[1, 3])

    _obs_kw = dict(fmt='ko', ms=7, zorder=6, label='Observed')
    _fit_kw = dict(marker='*', s=200, zorder=7, label='Best fit')
    fit_color = 'firebrick' if boundary_hit else 'dodgerblue'

    # ── Pre-compute path arrays ───────────────────────────────────────────────
    if star_path:
        path_teff = np.array([p['teff'] for p in star_path])
        path_lum  = np.array([p['lum']  for p in star_path])
        path_logg = np.array([p['logg'] for p in star_path])
    else:
        path_teff = path_lum = path_logg = np.array([])

    valid_hrd  = np.isfinite(path_teff) & np.isfinite(path_lum)
    valid_kiel = np.isfinite(path_teff) & np.isfinite(path_logg)

    # ── Compute zoom window ───────────────────────────────────────────────────
    # Gather all relevant teff/lum/logg points to set zoom bounds
    zoom_teff_pts = [teff_obs]
    zoom_lum_pts  = [lum_obs]
    zoom_logg_pts = [logg_obs]
    if model is not None and not np.isnan(res.get('fit_teff', np.nan)):
        zoom_teff_pts.append(res['fit_teff'])
        zoom_lum_pts.append(res['fit_lum'])
        zoom_logg_pts.append(res['fit_logg'])
    if nn is not None:
        zoom_teff_pts.extend(nn['teff'].dropna().tolist())
        zoom_lum_pts.extend(nn['lum'].dropna().tolist())
        zoom_logg_pts.extend(nn['logg'].dropna().tolist())

    def _zoom_limits(pts, pad_frac=0.3, min_span=None):
        lo, hi = np.nanmin(pts), np.nanmax(pts)
        span = hi - lo
        if min_span is not None and span < min_span:
            mid = (lo + hi) / 2
            lo, hi = mid - min_span / 2, mid + min_span / 2
            span = min_span
        pad = pad_frac * span
        return lo - pad, hi + pad

    zt_lo, zt_hi   = _zoom_limits(zoom_teff_pts, min_span=500)
    zlum_lo, zlum_hi = _zoom_limits(zoom_lum_pts, min_span=0.5)
    zlogg_lo, zlogg_hi = _zoom_limits(zoom_logg_pts, min_span=0.3)

    # ── Helper: draw one panel's content ─────────────────────────────────────
    def _draw_hrd_content(ax):
        ax.scatter(raw_grid_teff, raw_grid_lum,
                   c='lightgrey', s=0.5, alpha=0.3, zorder=1, rasterized=True, label='Grid')
        if nn is not None:
            ax.scatter(nn['teff'], nn['lum'], c=nn['distance'],
                       cmap='YlOrRd_r', s=15, zorder=3, label='Top 20 NN')
        ax.errorbar(teff_obs, lum_obs, xerr=e_teff, **_obs_kw)
        if model is not None and not np.isnan(res['fit_teff']):
            ax.scatter(res['fit_teff'], res['fit_lum'], c=fit_color, **_fit_kw)
        ax.set_xlabel('Teff (K)', fontsize=8)
        ax.set_ylabel('log(L/L☉)', fontsize=8)
        ax.invert_xaxis()

    def _draw_kiel_content(ax):
        ax.scatter(raw_grid_teff, raw_grid_logg,
                   c='lightgrey', s=0.5, alpha=0.3, zorder=1, rasterized=True, label='Grid')
        if nn is not None:
            ax.scatter(nn['teff'], nn['logg'], c=nn['distance'],
                       cmap='YlOrRd_r', s=15, zorder=3)
        ax.errorbar(teff_obs, logg_obs, xerr=e_teff, yerr=e_logg, **_obs_kw)
        if model is not None and not np.isnan(res['fit_teff']):
            ax.scatter(res['fit_teff'], res['fit_logg'], c=fit_color, **_fit_kw)
        ax.set_xlabel('Teff (K)', fontsize=8)
        ax.set_ylabel('log g', fontsize=8)
        ax.invert_xaxis()
        ax.invert_yaxis()

    def _draw_path_hrd_content(ax):
        ax.scatter(raw_grid_teff, raw_grid_lum,
                   c='lightgrey', s=0.5, alpha=0.2, zorder=1, rasterized=True)
        if valid_hrd.any():
            steps = np.arange(valid_hrd.sum())
            sc = ax.scatter(path_teff[valid_hrd], path_lum[valid_hrd],
                            c=steps, cmap='cool', s=8, alpha=0.6, zorder=4, label='Path')
            plt.colorbar(sc, ax=ax, label='Step', pad=0.0)
        ax.errorbar(teff_obs, lum_obs, xerr=e_teff, **_obs_kw)
        if model is not None and not np.isnan(res['fit_teff']):
            ax.scatter(res['fit_teff'], res['fit_lum'], c=fit_color, **_fit_kw)
        ax.set_xlabel('Teff (K)', fontsize=8)
        ax.set_ylabel('log(L/L☉)', fontsize=8)
        ax.invert_xaxis()

    def _draw_path_kiel_content(ax):
        ax.scatter(raw_grid_teff, raw_grid_logg,
                   c='lightgrey', s=0.5, alpha=0.2, zorder=1, rasterized=True)
        if valid_kiel.any():
            steps2 = np.arange(valid_kiel.sum())
            sc2 = ax.scatter(path_teff[valid_kiel], path_logg[valid_kiel],
                             c=steps2, cmap='cool', s=8, alpha=0.6, zorder=4)
            plt.colorbar(sc2, ax=ax, label='Step', pad=0.0)
        ax.errorbar(teff_obs, logg_obs, xerr=e_teff, yerr=e_logg, **_obs_kw)
        if model is not None and not np.isnan(res['fit_teff']):
            ax.scatter(res['fit_teff'], res['fit_logg'], c=fit_color, **_fit_kw)
        ax.set_xlabel('Teff (K)', fontsize=8)
        ax.set_ylabel('log g', fontsize=8)
        ax.invert_xaxis()
        ax.invert_yaxis()

    # ── Row 1: full-view panels ───────────────────────────────────────────────
    ax_hrd.set_title("H-R Diagram", fontsize=9)
    _draw_hrd_content(ax_hrd)
    ax_hrd.legend(fontsize=6, loc='upper right')

    ax_kiel.set_title("Kiel Diagram", fontsize=9)
    _draw_kiel_content(ax_kiel)

    ax_path_hrd.set_title("Optimizer path (H-R)", fontsize=9)
    _draw_path_hrd_content(ax_path_hrd)
    ax_path_hrd.legend(fontsize=6)

    ax_path_kiel.set_title("Optimizer path (Kiel)", fontsize=9)
    _draw_path_kiel_content(ax_path_kiel)

    # ── Row 2: zoomed-in versions of the same 4 panels ───────────────────────
    #ax_hrd_zoom.set_title("H-R Diagram (zoom)", fontsize=9)
    _draw_hrd_content(ax_hrd_zoom)
    ax_hrd_zoom.set_xlim(zt_hi, zt_lo)   # inverted: hi on left
    ax_hrd_zoom.set_ylim(zlum_lo, zlum_hi)

    #ax_kiel_zoom.set_title("Kiel Diagram (zoom)", fontsize=9)
    _draw_kiel_content(ax_kiel_zoom)
    ax_kiel_zoom.set_xlim(zt_hi, zt_lo)  # inverted
    ax_kiel_zoom.set_ylim(zlogg_hi, zlogg_lo)  # inverted

    #ax_path_hrd_zoom.set_title("Optimizer path (H-R zoom)", fontsize=9)
    _draw_path_hrd_content(ax_path_hrd_zoom)
    ax_path_hrd_zoom.set_xlim(zt_hi, zt_lo)
    ax_path_hrd_zoom.set_ylim(zlum_lo, zlum_hi)

    ax_path_kiel_zoom.set_title("Optimizer path (Kiel zoom)", fontsize=9)
    _draw_path_kiel_content(ax_path_kiel_zoom)
    ax_path_kiel_zoom.set_xlim(zt_hi, zt_lo)
    ax_path_kiel_zoom.set_ylim(zlogg_hi, zlogg_lo)


    outpath = f'results/diagnostics/star_{i+1:03d}_{star_id.replace("/","_")}.png'
    fig.savefig(outpath, dpi=110, bbox_inches='tight')
    plt.close(fig)

    bnd_flag = " ⚠ BOUNDARY" if boundary_hit else ""
    print(f"    → Age={res.get('fit_age', np.nan):.2f} Gyr  "
          f"Mass={res.get('fit_mass', np.nan):.3f} M☉  "
          f"α_MLT={res.get('fit_mixing_length', np.nan):.3f}  "
          f"loss={res['fit_loss']:.4f}{bnd_flag}")

# ── Save results table ────────────────────────────────────────────────────────
results_df = pd.DataFrame(result_rows)
results_df.to_csv('results/fit_results.csv', index=False)
print(f"\nResults saved to results/fit_results.csv")

# ── Summary plots ─────────────────────────────────────────────────────────────
print("Making summary plots...")
os.makedirs('results/individual_plots', exist_ok=True)

df_all  = results_df[results_df['fit_success']].copy()
df_good = df_all[~df_all['boundary_hit']].copy()
df_bnd  = df_all[ df_all['boundary_hit']].copy()
n_good  = len(df_good)
n_bnd   = len(df_bnd)
n_skip  = (results_df['skip_reason'] == 'out_of_grid').sum()

def scatter_with_boundary(ax, x_col, y_col, color_col, cmap, df_good, df_bnd,
                           cbar_label, invert_x=False, invert_y=False):
    '''Scatter plot with boundary hits shown as red X markers.'''
    if len(df_good) > 0:
        sc = ax.scatter(df_good[x_col], df_good[y_col],
                        c=df_good[color_col], cmap=cmap,
                        s=50, zorder=3, label='Good fit')
        plt.colorbar(sc, ax=ax, label=cbar_label)
    if len(df_bnd) > 0:
        ax.scatter(df_bnd[x_col], df_bnd[y_col],
                   c='firebrick', marker='x', s=60, zorder=4,
                   linewidths=1.5, label='Boundary hit')
    if invert_x: ax.invert_xaxis()
    if invert_y: ax.invert_yaxis()

# ── Panel drawing functions ────────────────────────────────────────────────────
def draw_age_hist(ax):
    if len(df_good):
        ax.hist(df_good['fit_age'].dropna(), bins=20,
                color='steelblue', edgecolor='white', alpha=0.85, label='Good fit')
    if len(df_bnd):
        ax.hist(df_bnd['fit_age'].dropna(), bins=20,
                color='firebrick', edgecolor='white', alpha=0.6, label='Boundary hit')
    ax.set_xlabel('Age (Gyr)', fontsize=11)
    ax.set_ylabel('N', fontsize=11)
    ax.set_title('Age distribution', fontsize=12)
    ax.legend(fontsize=9)

def draw_mass_hist(ax):
    if len(df_good):
        ax.hist(df_good['fit_mass'].dropna(), bins=20,
                color='seagreen', edgecolor='white', alpha=0.85, label='Good fit')
    if len(df_bnd):
        ax.hist(df_bnd['fit_mass'].dropna(), bins=20,
                color='firebrick', edgecolor='white', alpha=0.6, label='Boundary hit')
    lo_m, hi_m = GRID_BOUNDS.get('initial_mass', (None, None))
    if lo_m is not None:
        ax.axvline(lo_m, color='purple', lw=1.5, ls='--', label='Grid edge')
        ax.axvline(hi_m, color='purple', lw=1.5, ls='--')
    ax.set_xlabel('Mass (M☉)', fontsize=11)
    ax.set_ylabel('N', fontsize=11)
    ax.set_title('Mass distribution', fontsize=12)
    ax.legend(fontsize=9)

def draw_mixing_length_hist(ax):
    if len(df_good):
        ax.hist(df_good['fit_mixing_length'].dropna(), bins=20,
                color='coral', edgecolor='white', alpha=0.85, label='Good fit')
    if len(df_bnd):
        ax.hist(df_bnd['fit_mixing_length'].dropna(), bins=20,
                color='firebrick', edgecolor='white', alpha=0.6, label='Boundary hit')
    lo_ml, hi_ml = GRID_BOUNDS.get('mixing_length', (None, None))
    if lo_ml is not None:
        ax.axvline(lo_ml, color='purple', lw=1.5, ls='--', label='Grid edge')
        ax.axvline(hi_ml, color='purple', lw=1.5, ls='--')
    ax.set_xlabel('Mixing length α', fontsize=11)
    ax.set_ylabel('N', fontsize=11)
    ax.set_title('Mixing length distribution', fontsize=12)
    ax.legend(fontsize=9)

def draw_hrd(ax):
    ax.scatter(raw_grid_teff, raw_grid_lum,
               c='lightgrey', s=1, alpha=0.2, zorder=1, rasterized=True, label='Grid')
    scatter_with_boundary(ax, 'teff_obs', 'lum_obs', 'fit_age', 'plasma',
                          df_good, df_bnd, 'Age (Gyr)', invert_x=True)
    ax.set_xlabel('Teff (K)', fontsize=11)
    ax.set_ylabel('log(L/L☉)', fontsize=11)
    ax.set_title('H-R diagram (colour = age)', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')

def draw_kiel(ax):
    ax.scatter(raw_grid_teff, raw_grid_logg,
               c='lightgrey', s=1, alpha=0.2, zorder=1, rasterized=True, label='Grid')
    scatter_with_boundary(ax, 'teff_obs', 'logg_obs', 'fit_mass', 'viridis',
                          df_good, df_bnd, 'Mass (M☉)', invert_x=True, invert_y=True)
    ax.set_xlabel('Teff (K)', fontsize=11)
    ax.set_ylabel('log g', fontsize=11)
    ax.set_title('Kiel diagram (colour = mass)', fontsize=12)
    ax.legend(fontsize=8)

def draw_alpha_vs_mh(ax):
    if len(df_good):
        sc = ax.scatter(df_good['mh_obs'], df_good['fit_mixing_length'],
                        c=df_good['fit_age'], cmap='plasma',
                        s=40, alpha=0.8, label='Good fit')
        plt.colorbar(sc, ax=ax, label='Age (Gyr)')
    if len(df_bnd):
        ax.scatter(df_bnd['mh_obs'], df_bnd['fit_mixing_length'],
                   c='firebrick', marker='x', s=60, linewidths=1.5, label='Boundary hit')
    lo_ml, hi_ml = GRID_BOUNDS.get('mixing_length', (None, None))
    if lo_ml is not None:
        ax.axhline(lo_ml, color='purple', lw=1, ls='--', alpha=0.6, label='Grid edge')
        ax.axhline(hi_ml, color='purple', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('[M/H]', fontsize=11)
    ax.set_ylabel('α_MLT', fontsize=11)
    ax.set_title('α_MLT vs [M/H]', fontsize=12)
    ax.legend(fontsize=9)

def draw_teff_resid(ax):
    if len(df_good):
        ax.hist((df_good['teff_obs'] - df_good['fit_teff']).dropna(),
                bins=20, color='slateblue', edgecolor='white', alpha=0.85, label='Good fit')
    if len(df_bnd):
        ax.hist((df_bnd['teff_obs'] - df_bnd['fit_teff']).dropna(),
                bins=20, color='firebrick', edgecolor='white', alpha=0.6, label='Boundary hit')
    ax.axvline(0, color='k', lw=1.5)
    ax.set_xlabel('Teff_obs − Teff_fit (K)', fontsize=11)
    ax.set_ylabel('N', fontsize=11)
    ax.set_title('Teff residuals', fontsize=12)
    ax.legend(fontsize=9)

def draw_logg_resid(ax):
    if len(df_good):
        ax.hist((df_good['logg_obs'] - df_good['fit_logg']).dropna(),
                bins=20, color='darkorange', edgecolor='white', alpha=0.85, label='Good fit')
    if len(df_bnd):
        ax.hist((df_bnd['logg_obs'] - df_bnd['fit_logg']).dropna(),
                bins=20, color='firebrick', edgecolor='white', alpha=0.6, label='Boundary hit')
    ax.axvline(0, color='k', lw=1.5)
    ax.set_xlabel('logg_obs − logg_fit', fontsize=11)
    ax.set_ylabel('N', fontsize=11)
    ax.set_title('log g residuals', fontsize=12)
    ax.legend(fontsize=9)

def draw_loss_dist(ax):
    if len(df_good):
        ax.hist(df_good['fit_loss'].dropna(), bins=25,
                color='grey', edgecolor='white', alpha=0.85, label='Good fit')
    if len(df_bnd):
        ax.hist(df_bnd['fit_loss'].dropna(), bins=25,
                color='firebrick', edgecolor='white', alpha=0.6, label='Boundary hit')
    ax.axvline(FIT_TOL, color='r', lw=1.5, ls='--', label=f'tol={FIT_TOL}')
    ax.set_xlabel('Fit loss', fontsize=11)
    ax.set_ylabel('N', fontsize=11)
    ax.set_title('Loss distribution (red = tolerance)', fontsize=12)
    ax.legend(fontsize=9)

# ── Summary figure (3×3) ──────────────────────────────────────────────────────
panel_funcs = [
    draw_age_hist, draw_mass_hist, draw_mixing_length_hist,
    draw_hrd, draw_kiel, draw_alpha_vs_mh,
    draw_teff_resid, draw_logg_resid, draw_loss_dist,
]
panel_names = [
    '01_age_dist', '02_mass_dist', '03_mixing_length_dist',
    '04_hrd', '05_kiel', '06_alpha_vs_mh',
    '07_teff_residuals', '08_logg_residuals', '09_loss_dist',
]

fig, axes = plt.subplots(3, 3, figsize=(17, 14))
fig.suptitle(
    f"Platinum Sample — JT2017t12 Fit Summary\n"
    f"N={len(df_all)}/{n_stars} fitted   "
    f"Good: {n_good}   Boundary: {n_bnd}   Skipped: {n_skip}",
    fontsize=13, fontweight='bold')
for ax, fn in zip(axes.flatten(), panel_funcs):
    fn(ax)
plt.tight_layout()
fig.savefig('results/summary_plots.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Summary plots saved to results/summary_plots.png")

# ── Individual plots ──────────────────────────────────────────────────────────
for name, fn in zip(panel_names, panel_funcs):
    fig, ax = plt.subplots(figsize=(7, 5))
    fn(ax)
    plt.tight_layout()
    fig.savefig(f'results/individual_plots/{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
print("Individual plots saved to results/individual_plots/")

print(f"\nDone.  Good fits: {n_good}  Boundary hits: {n_bnd}  Skipped: {n_skip}")
