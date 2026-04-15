'''
custom_install.py

This is a template to help users install custom grids to kiauhoku. The template
is based on the installation for YREC grids.

Necessary:
(1) a function called `setup` that returns a pandas MultiIndexed 
    DataFrame containing all your evolution tracks
(2) a variable `name` that is set to whatever 
    you want your installed grid to be named
(3) a variable `raw_grids_path` that sets the path to where
    your raw grid is downloaded.

Optional:
 - You can customize the functions you want to use to define your EEPs
   (see my_RGBump for an example). You'll need to group them in a dictionary
   called `eep_functions` that gets imported by kiauhoku. The allowed keys are
    'prems' for Pre-Main Sequence
    'zams'  for Zero-Age Main Sequence
    'eams'  for Early-Age Main Sequence
    'iams'  for Intermediate-Age Main Sequence
    'tams'  for Terminal-Age Main Sequence
    'rgbump' for the Red Giant Branch Bump (~ halfway up to the TRGB)
 - You can customize the metric function you use to space the EEPs
   (see my_HRD for an example). It either must be called `metric_function`
   or be aliased as `metric_function`.
'''

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

name = 'JT2017t11'
#raw_grids_path = '/data/home/jtayar/EVOLUTION/zuul/0/tayar/YRECgrid/nodiff_out4zKH' #/home/zach/Downloads/grids/yrec'
raw_grids_path = '/home/jtayar/Desktop/RunDir/KHbiggrid'

# Assign labels used in eep conversion
eep_params = dict(
    age = 'Age(Gyr)',
    log_central_temp = 'logT(cen)',
    core_hydrogen_frac = 'Xcen',
    hydrogen_lum = 'H lum (Lsun)',
    lum = 'L/Lsun',
    logg = 'logg',
    log_teff = 'Log Teff(K)',
    core_helium_frac = 'Ycen',
    teff_scale = 20, # used in metric function
    lum_scale = 1, # used in metric function
    # `intervals` is a list containing the number of secondary Equivalent
    # Evolutionary Phases (EEPs) between each pair of primary EEPs.
    intervals = [200, # Between PreMS and ZAMS
                  50, # Between ZAMS and EAMS 
                 100, # Between EAMS and IAMS
                 100, # IAMS-TAMS
                 150], # TAMS-RGBump
)

def my_RGBump(track, eep_params, i0=None):
#    '''
#    Modified from eep.get_RGBump to make luminosity logarithmic
#    '''
#
    lum = eep_params['lum']
    log_teff = eep_params['log_teff']
    N = len(track)

    lum_tr = track.loc[i0:, lum]
    logT_tr = track.loc[i0:, log_teff]

    lum_greater = (lum_tr > 1)
    if not lum_greater.any():
        return -1
    RGBump = lum_greater.idxmax() + 1

    while logT_tr[RGBump] < logT_tr[RGBump-1] and RGBump < N-1:
        RGBump += 1

    # Two cases: 1) We didn't reach an extremum, in which case RGBump gets
    # set as the final index of the track. In this case, return -1.
    # 2) We found the extremum, in which case RGBump gets set
    # as the index corresponding to the extremum.
    if RGBump >= N-1:
        return -1
    return RGBump-1
    
def _first_true_index(bools):
    '''
    Given a pandas Series of bools, returns the index of the first occurrence
    of `True`. **Index-based, NOT location-based**
    I.e., say x = pd.Series({0: False, 2: False, 4: True}), then
    _first_true_index(x) will return index 4, not the positional index 2.

    If no value in `bools` is True, returns -1.
    '''
    if not bools.any():
        return -1
    return bools.idxmax()
    
def my_RGBump2(track, eep_params, i0=None):
    '''
    Modified from eep.get_RGBump to make luminosity logarithmic
    '''

    lum = eep_params['lum']
    log_teff = eep_params['log_teff']
    N = len(track)

    lum_tr = track.loc[i0:, lum]
    logT_tr = track.loc[i0:, log_teff]

    RGBump = _first_true_index(lum_tr > 2.5) + 1
#    print(RGBump)
    if RGBump == 0:
        return -1
    return RGBump-1 #RGBump-1

#    while logT_tr[RGBump] < logT_tr[RGBump-1] and RGBump < N-1:
#        RGBump += 1

    # Two cases: 1) We didn't reach an extremum, in which case RGBump gets
    # set as the final index of the track. In this case, return -1.
    # 2) We found the extremum, in which case RGBump gets set
    # as the index corresponding to the extremum.
#    if RGBump >= N-1:
#        return -1



def my_HRD(track, eep_params):
    '''
    Adapted from eep._HRD_distance to fix lum logarithm
    '''

    # Allow for scaling to make changes in Teff and L comparable
    Tscale = eep_params['teff_scale']
    Lscale = eep_params['lum_scale']

    log_teff = eep_params['log_teff']
    lum = eep_params['lum']

    logTeff = track[log_teff]
    logLum = track[lum]

    N = len(track)
    dist = np.zeros(N)
    for i in range(1, N):
        temp_dist = (((logTeff.iloc[i] - logTeff.iloc[i-1])*Tscale)**2
                    + ((logLum.iloc[i] - logLum.iloc[i-1])*Lscale)**2)
        dist[i] = dist[i-1] + np.sqrt(temp_dist)

    return dist
## JT test
#nope this is bad then it does nothing for the giant branch
#eep_functions = {
#    'rgbump': my_RGBump
#}

eep_functions = {
    'rgbump': my_RGBump2
}


metric_function = my_HRD

def read_columns(path):
    with open(path, 'r') as f:
        columns = [l.strip() for l in f]
    
    return columns

def parse_filename(filename):
    file_str = filename.replace('.track4', '')

#    mass = float(file_str[:4].replace('_', '.'))
    mass = float(file_str[1:4])/100.

    met_i = file_str.find('fh') + 2
    met_str = file_str[met_i:met_i+4]
    met = float(met_str[1:])/100
    if  met != 0 and met_str[0] == 'm':
        met *= -1

    alpha_i = file_str.find('al') + 2
    alpha_str = file_str[alpha_i:alpha_i+2]
    alpha = float(alpha_str)/10

    he_i=file_str.find('y') + 1
    he_str = file_str[he_i:he_i+3]
    he = float(he_str)/1000
    if he==0.273:
    	he=0.272683
    
    ml = float(file_str[15:17])/10.
    if ml==1.7:
    	ml=1.724485
    if ml==1.2:
    	ml=1.224485
    if ml==2.2:
    	ml=2.224485


    return mass, met, alpha, he, ml


def from_yrec(path, columns=None):
    if columns is None:
        raw_grids_path = os.path.dirname(path)
        columns = read_columns(os.path.join(raw_grids_path, 'column_labels.txt'))

    fname = os.path.basename(path)
    initial_mass, initial_met, initial_alpha, initial_he, mixing_length = parse_filename(fname)

    data = np.loadtxt(path)
    s = np.arange(len(data))
    m = np.ones_like(s) * initial_mass
    z = np.ones_like(s) * initial_met
    al = np.ones_like(s) * initial_alpha
    he = np.ones_like(s) * initial_he
    ml = np.ones_like(s) * mixing_length

    # Build multi-indexed DataFrame, dropping unwanted columns
    multi_index = pd.MultiIndex.from_tuples(zip(m, z, al, he, ml, s),
        names=['initial_mass', 'initial_met', 'alpha_fe', 'initial_he', 'mixing_length', 'step'])
    df = pd.DataFrame(data, index=multi_index, columns=columns)
    df = df.drop(columns=[c for c in columns if '#' in c])

    return df

def setup(raw_grids_path=raw_grids_path, progress=True):
    df_list = []
    filelist = [f for f in os.listdir(raw_grids_path) if '.track4' in f]
    columns = read_columns(os.path.join(raw_grids_path, 'column_labels.txt'))

    if progress:
        file_iter = tqdm(filelist)
    else:
        file_iter = filelist

    for fname in file_iter:
        fpath = os.path.join(raw_grids_path, fname)
#        print(fname)
        try:
            df_list.append(from_yrec(fpath, columns))
        except:
            print(fname)


    dfs = pd.concat(df_list).sort_index()
    # If you want to compute a total hydrogen luminosity, uncomment the next line
    #dfs[eep_params['hydrogen lum']] = dfs[['ppI', 'ppII', 'ppIII']].sum(axis=1)

    return dfs 
