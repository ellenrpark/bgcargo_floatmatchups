#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:40:49 2023

@author: epark
"""

from getfloats_postprocessing import do_matchups_with_thresholds

# Can specify up to 3 threshold criteria
# param options ['TEMP','PSAL','DEPTH','PRES','SIGMA0']

param_thresh = {'SIGMA0': 0.5,
              'TEMP': 0.1,
              'PSAL': 0.05}

time_thresh = 60 # minutes; +/- time window from float profile

doxy_type = 'DOXY_exp'

mooring_flist = ['test_data/204380_compensated_oxygen.csv'] # glob.glob(dir with post processed mooring data)

do_matchups_with_thresholds(mooring_flist, param_thresh, time_thresh, doxy_type)
