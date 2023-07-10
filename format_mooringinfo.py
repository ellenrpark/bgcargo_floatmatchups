#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:32:58 2023

@author: epark
"""

import pandas as pd
import numpy as np


mooring_info = pd.read_csv('summary_table_deployment_table.csv')

##### Format mooring info
# Subset data for key info, reformat and make output file names
out_flist = [str(mooring_info.loc[:,'serial_number'].values[i])+'_'+mooring_info.loc[:,'station'].values[i]+'_'+str(mooring_info.loc[:,'DEPTH'].values[i])+'dbar' for i in np.arange(mooring_info.shape[0])]
s_dates = [[np.NaN]]*mooring_info.shape[0]
e_dates = [[np.NaN]]*mooring_info.shape[0]

for i in np.arange(mooring_info.shape[0]):
    if type(mooring_info.loc[:,'deploy_date'][i]) == str:
        dep_date = mooring_info.loc[:,'deploy_date'][i].split('-')
    else:
        # If no deployment date, will not do match up
        dep_date = np.NaN
        
    if type(mooring_info.loc[:,'recovery_date'][i]) == str:
        rec_date = mooring_info.loc[:,'recovery_date'][i].split('-')
    else:
        # If no recovery date, will use end of 2022
        rec_date = ['2022','12','31']
    
    # If date range is valid, continue with analysis
    if type(dep_date)==list:
        start_date = pd.Timestamp(int(dep_date[0]),int(dep_date[1]),int(dep_date[2]))
        end_date = pd.Timestamp(int(rec_date[0]),int(rec_date[1]),int(rec_date[2]))
        
        s_dates[i]=start_date
        e_dates[i]=end_date
        
mooring_info = mooring_info.assign(START_DATE = s_dates)
mooring_info = mooring_info.assign(END_DATE = e_dates)
mooring_info = mooring_info.assign(OUT_FNAME = out_flist)

matchup_info = mooring_info.loc[:,['LATITUDE','LONGITUDE','DEPTH','START_DATE','END_DATE','OUT_FNAME']]

matchup_info.to_csv('matchup_info.csv')