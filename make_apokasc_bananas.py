"""
patch_chains_with_age_errors.py

Injects e_int_age_hi and e_int_age_lo into existing chain pickles
without rerunning MCMC. Matches on 2MASSID / star_id.

Usage:
    python patch_chains_with_age_errors.py
"""

import os
import pickle
import numpy as np
import pandas as pd

CHAIN_DIR = 'results/apokasc/chains'
ERR_FILE  = 'MeridithRomanApokascCalibLtest5ns3L_with_errs.out'

# ── Load error table ──────────────────────────────────────────────────────────
err_df = pd.read_csv(ERR_FILE, sep=r'\s+')
err_df['star_id'] = err_df['2MASSID'].astype(str).str.strip()
err_lookup = err_df.set_index('star_id')[['IntAge_err_hi', 'IntAge_err_lo']].to_dict('index')

print(f"Loaded {len(err_lookup)} stars from {ERR_FILE}")

# ── Patch each chain ──────────────────────────────────────────────────────────
chain_files = sorted(f for f in os.listdir(CHAIN_DIR) if f.endswith('.pkl'))
print(f"Patching {len(chain_files)} chains...")

n_patched = 0
n_missing = 0

for fname in chain_files:
    path = os.path.join(CHAIN_DIR, fname)
    with open(path, 'rb') as f:
        res = pickle.load(f)

    star_id = res['star_id']
    if star_id in err_lookup:
        row = err_lookup[star_id]
        res['e_int_age_hi'] = float(row['IntAge_err_hi']) if pd.notna(row['IntAge_err_hi']) else np.nan
        res['e_int_age_lo'] = float(row['IntAge_err_lo']) if pd.notna(row['IntAge_err_lo']) else np.nan
        with open(path, 'wb') as f:
            pickle.dump(res, f)
        n_patched += 1
    else:
        n_missing += 1
        print(f"  WARNING: {star_id} not found in error table")

print(f"\nDone. Patched {n_patched} chains, {n_missing} not found in error table.")
print("Now re-run: python make_apokasc_bananas.py --best")
