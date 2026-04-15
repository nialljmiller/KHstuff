import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import kiauhoku as kh
import pandas as pd
from astropy.table import Table

def fit_all_grids(star, *args, **kwargs):
    gridnames = []
    models = []
    for gname, interp in zip(
        ['jtgrid'],
        [jtgrid]):
        model, fit = interp.gridsearch_fit(star, *args, **kwargs)
        if fit.success:
            gridnames.append(gname)
            models.append(
                model[['initial_mass', 'initial_met', 'initial_he', 'alpha_fe', 'mixing_length', 'eep', 'mass', 'teff', 'lum', 'met', 'logg', 'age']]
            )
    models = pd.concat(models, axis=1)
    models.columns = gridnames

    return models

def compute_statistics(models, exclude=None):
    stats = models.copy()
    if exclude is not None:
        stats = stats.drop(columns=exclude)

    mean = stats.mean(axis=1)
    stdev = stats.std(axis=1, ddof=1)
    max_offset = stats.max(axis=1) - stats.min(axis=1)

    stats['mean'] = mean
    stats['stdev'] = stdev
    stats['max offset'] = max_offset

    return stats


#create the kiauhoku grid
# use grid points between ZAMS (201) and upper giant branch (605)
qstring = '201 <= eep'

# Whether to fit evolved metallicity (True) or use the initial metallicity.
# False is probably fine if you're not on the giant branch.
evolve_met = False

# load grid, remove unwanted rows
jtgrid = kh.load_eep_grid("JT2017t11").query(qstring)
# set column names to some standard
jtgrid['mass'] = jtgrid['Mass(Msun)']
jtgrid['teff'] = 10**jtgrid['Log Teff(K)']
jtgrid['lum'] = jtgrid['L/Lsun'] #10**
if evolve_met:
    jtgrid['met'] = np.log10(jtgrid['Zsurf']/jtgrid['Xsurf']/0.0253)
else:
    jtgrid['met'] = jtgrid.index.get_level_values('initial_met')
jtgrid['initial_he'] = jtgrid.index.get_level_values('initial_he')
jtgrid['mixing_length']= jtgrid.index.get_level_values('mixing_length')
jtgrid['alpha_fe']= jtgrid.index.get_level_values('alpha_fe')
jtgrid['age'] = jtgrid['Age(Gyr)']
# set name for readability of output
jtgrid.set_name('jtgrid')
#jtgrid
# cast to interpolator
jtgrid =jtgrid.to_interpolator()

#define a star to fit

#APOKASC Red giant : get Segmentation fault (core dumped). Not obviously a memory issue.        
star4= {'teff':4718.9, 'lum':1.82, 'met':-0.2617, 
        'logg':2.515, 'alpha_fe':0.0815 , 'initial_he': 0.2640}
#from the less fancy version this      
#mass=10^(logg)/10^(4.44)/(teff/5777)^4*10^logLLsun=1.76 Msun
#put into APOKASCMeridithRightAnswer.txt, get mixing length=1.79 age=0.71 Gyr  
        
#the sun: success!       
star5= {'teff':5777, 'lum':0, 'met':0, 
        'logg':4.4, 'alpha_fe':0 , 'initial_he': 0.272}
    
scale1 = {'teff':1000, 'lum':0.1, 'met':0.1, 'logg':0.1, 'alpha_fe':0.1, 'initial_he':0.01}
model4 = fit_all_grids(star5, scale=scale1, tol=1e-2, maxiter=100, bounds=[(0.7, 2.0), (-1.0,0.0), (0.0, 0.2), (0.24, 0.3), (1.2, 2.3)]
print(model4)
