#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 12:49:07 2023

@author: epark
"""

from getfloats_gohsnap import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

matchup_info = pd.read_csv('matchup_info.csv')
# Must be formated to have LATITUDE, LONGITUDE, START_DATE, END_DATE, OUT_FNAME
# Must have DEPTH or PRES...you specify

use_local_dac = False

dist_thresh = 25 # km...how far away from each lat-lon spot to look

# Note: using nominal sensor depth from deployment table
d_type = 'DEPTH' # What vertical axis to look along DEPTH or PRES
d_thresh = 20 #  depth_bin threshold (+/-) around nominal deployment depth, dbar/m

params =['DOXY'] # look at floats with the following parameters
over_write = False # over_write pre-processed data


do_matchups(matchup_info, params, dist_thresh, d_type, d_thresh, over_write,use_local_dac)
