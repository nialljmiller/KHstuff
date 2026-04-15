# quick_combine.py — run this instead of --combine
import os, pickle
import numpy as np
import pandas as pd

chain_dir = 'results/bananas/chains'
chain_files = sorted(f for f in os.listdir(chain_dir) if f.endswith('.pkl'))
print(f"Combining {len(chain_files)} chains...")

all_rows    = []
banana_dict = {}

for fname in chain_files:
    with open(os.path.join(chain_dir, fname), 'rb') as f:
        res = pickle.load(f)

    star_id      = res['star_id']
    flat_samples = res['flat_samples']
    blobs_df     = res['blobs_df']

    # Drop any blob columns that duplicate flat_samples columns
    overlap = [c for c in blobs_df.columns if c in flat_samples.columns]
    blobs_clean = blobs_df.drop(columns=overlap)

    output = pd.concat([flat_samples.reset_index(drop=True),
                        blobs_clean.reset_index(drop=True)], axis=1)
    output['star_id']       = star_id
    output['stellar_class'] = res.get('stellar_class', 'unknown')
    output['teff_obs']      = res['teff_obs']
    output['lum_obs']       = res['lum_obs']
    output['logg_obs']      = res['logg_obs']
    output['mh_obs']        = res['mh_obs']

    banana_dict[star_id] = output
    all_rows.append(output)
    print(f"  {star_id}: {len(output):,} samples")

banana_df = pd.concat(all_rows, ignore_index=True)
banana_df.to_csv('results/bananas/banana_data.csv', index=False)
print(f"\nSaved {len(banana_df):,} rows to banana_data.csv")

with open('results/bananas/bananas.pkl', 'wb') as f:
    pickle.dump(banana_dict, f)
print("Saved bananas.pkl")
