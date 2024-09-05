#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:40:49 2023

@author: epark
"""

from getfloats_postprocessing import do_matchups_with_thresholds
import glob

# Can specify up to 3 threshold criteria
# param options ['TEMP','PSAL','DEPTH','PRES','SIGMA0']

param_thresh = {'SIGMA0': 0.01}

param_thresh = {'PSAL': 0.005,
              'TEMP': 0.005,
              'PRES': 100}

time_thresh = 60*24*5 # minutes; +/- time window from float profile

doxy_type = 'DOXY_drift_corrected'


mooring_flist = sorted(glob.glob('/Volumes/FATDATABABY/OSNAP grid interpolation/Ungridded oxygen/*.csv'))

#mooring_flist=mooring_flist[-1:]

do_matchups_with_thresholds(mooring_flist, param_thresh, time_thresh, doxy_type)
