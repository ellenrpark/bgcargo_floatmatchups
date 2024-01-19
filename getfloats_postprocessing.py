#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:21:53 2023

@author: epark
"""

import pandas as pd
import numpy as np
import gsw
import glob
import os
import matplotlib.pyplot as plt

def GetThreshInds(float_data, sensor_mean, param_thresh):
    
    thresh_keys = list(param_thresh.keys())
    
    if len(thresh_keys)==1:
        inds = np.where((float_data.loc[:,thresh_keys[0]]<=sensor_mean.loc[thresh_keys[0]]+param_thresh[thresh_keys[0]]) & \
                        (float_data.loc[:,thresh_keys[0]]>=sensor_mean.loc[thresh_keys[0]]-param_thresh[thresh_keys[0]]))[0]
    elif len(thresh_keys) == 2:
        inds = np.where((float_data.loc[:,thresh_keys[0]]<=sensor_mean.loc[thresh_keys[0]]+param_thresh[thresh_keys[0]]) & \
                        (float_data.loc[:,thresh_keys[0]]>=sensor_mean.loc[thresh_keys[0]]-param_thresh[thresh_keys[0]]) & \
                            (float_data.loc[:,thresh_keys[1]]<=sensor_mean.loc[thresh_keys[1]]+param_thresh[thresh_keys[1]]) & \
                            (float_data.loc[:,thresh_keys[1]]>=sensor_mean.loc[thresh_keys[1]]-param_thresh[thresh_keys[1]]))[0]
    elif len(thresh_keys)==3:
        inds = np.where((float_data.loc[:,thresh_keys[0]]<=sensor_mean.loc[thresh_keys[0]]+param_thresh[thresh_keys[0]]) & \
                        (float_data.loc[:,thresh_keys[0]]>=sensor_mean.loc[thresh_keys[0]]-param_thresh[thresh_keys[0]]) & \
                            (float_data.loc[:,thresh_keys[1]]<=sensor_mean.loc[thresh_keys[1]]+param_thresh[thresh_keys[1]]) & \
                            (float_data.loc[:,thresh_keys[1]]>=sensor_mean.loc[thresh_keys[1]]-param_thresh[thresh_keys[1]]) & \
                                (float_data.loc[:,thresh_keys[2]]<=sensor_mean.loc[thresh_keys[2]]+param_thresh[thresh_keys[2]]) & \
                                (float_data.loc[:,thresh_keys[2]]>=sensor_mean.loc[thresh_keys[2]]-param_thresh[thresh_keys[2]]))[0]
        
    return inds

def do_matchups_with_thresholds(mooring_flist, param_thresh, time_thresh, doxy_type):
    # Load mooring info table to get mooring lat-lon positions
    mooring_info = pd.read_csv('matchup_info.csv')
    
    # other info
    cvals = ['PRES','TEMP','PSAL','DOXY','DEPTH','SIGMA0','PRES_ERROR','TEMP_ERROR',
             'PSAL_ERROR','DOXY_ERROR']
    
    mooring_param_list = ['TEMP','PSAL','PRES','DOXY','SIGMA0']
    
    for mi in np.arange(len(mooring_flist)):
        fname = mooring_flist[mi]
        
        # Load sensor data
        sensor_data = pd.read_csv(fname)
        
        print('\nMatch-up for: '+fname.split('/')[-1].split('.')[0])
        # get sensor SN number from file name
        sensor = fname.split('/')[-1].split('_')[0]
        s_inds = np.where(np.array([fi.split('_')[0] for fi in mooring_info.loc[:,'OUT_FNAME'].values]) == sensor)[0][0]
        
        # calculate sigma0
        SA = gsw.SA_from_SP(sensor_data.loc[:,'PSAL'].values,
                            sensor_data.loc[:,'PRES'].values, 
                            mooring_info.loc[:,'LONGITUDE'].values[s_inds], 
                            mooring_info.loc[:,'LATITUDE'].values[s_inds])
        
        CT = gsw.CT_from_t(SA, sensor_data.loc[:,'TEMP'].values, sensor_data.loc[:,'PRES'].values,)
        sigma0 = gsw.sigma0(SA,CT)+1000
        sensor_data = sensor_data.assign(SIGMA0 = sigma0)
        
        # Rename time column
        sensor_data = sensor_data.rename(columns={"Unnamed: 0": "JULD"})
        
        # load matchup file
        matchup_fname = glob.glob('matchup_output/'+sensor+'*.csv')
        
        plt.figure(figsize = (10,8))
        plt.plot([pd.Timestamp(pi) for pi in sensor_data.loc[:,'JULD'].values], 
                 sensor_data.loc[:,doxy_type].values, 'k-', label = 'moored sensor')
        if len(matchup_fname)>0:
            # Matchup file exists
            
            matchup_info = pd.read_csv(matchup_fname[0])
            
            processed_info = matchup_info.copy()
            
            mooring_values = np.zeros((matchup_info.shape[0],len(mooring_param_list),2))*np.NaN
            # Get list of float matchup file names
            float_list = [fi.split('/')[-1].split('.')[0]+'.pkl' for fi in matchup_info.loc[:,'file'].values]
            
            for fi in np.arange(len(float_list)):
                float_fname = 'matchup_output/pickle/'+float_list[fi]
                
                print(float_list[fi])
                
                float_data = pd.read_pickle(float_fname)
                float_data = float_data.iloc[2:, :] # Drop first 2 rows because metadata
                
                float_date = pd.Timestamp(matchup_info.loc[:,'date_format'].values[fi])
                
                # Get mooring data that falls within the time window
                # number of seconds before/after float profile
                
                mooring_times = np.array([(pd.Timestamp(pi)-float_date).total_seconds() for pi in sensor_data.loc[:,'JULD'].values])
                ti = np.where((mooring_times/60 >= time_thresh*-1) & \
                              (mooring_times/60 <= time_thresh))[0]
                    
                if ti.shape[0]>0:
                    # There is data to take average over
                    # Take mean of all columns except date columns
                    sensor_mean = sensor_data.iloc[ti, 1:].mean()
                    sensor_std = sensor_data.iloc[ti, 1:].std()
                    
                    for si in np.arange(len(mooring_param_list)):
                        mooring_values[fi, si,0] = sensor_mean.loc[mooring_param_list[si]]
                        mooring_values[fi, si,1] = sensor_std.loc[mooring_param_list[si]]
                        
                    # Now find float data that falls within specified thresholds
                    matchup_inds = GetThreshInds(float_data, sensor_mean, param_thresh)
                    
                    if matchup_inds.shape[0]>0:
                        # Data overlap
                        float_mean = float_data.iloc[matchup_inds, :].mean()
                        float_std = float_data.iloc[matchup_inds, :].std()
        
                        for ci in cvals:
                            processed_info.loc[matchup_info.index.values[fi], ci]=float_mean.loc[ci]
                            processed_info.loc[matchup_info.index.values[fi], ci+'_STD']=float_std.loc[ci]
        
                    else:
                        # No matchup...set values to NaN
        
                        for ci in cvals:
                            processed_info.loc[matchup_info.index.values[fi], ci]=np.NaN
                            processed_info.loc[matchup_info.index.values[fi], ci+'_STD']=np.NaN
                            
                    if matchup_inds.shape[0]>0:
                        if np.isnan(float_mean.loc['DOXY']) == False:
                            plt.scatter(float_date, float_mean.loc['DOXY'],
                                         label = float_list[0].split('.')[0]+' | '+ matchup_info.loc[:,'DOXY_MODE'].values[fi])
                            plt.errorbar(float_date, float_mean.loc['DOXY'], yerr = float_mean.loc['DOXY_ERROR'])
                
            # Save mooring values
            for si in np.arange(len(mooring_param_list)):
                processed_info = processed_info.assign(**{mooring_param_list[si]+'_MOORING': mooring_values[:, si,0]})
                processed_info = processed_info.assign(**{mooring_param_list[si]+'_STD_MOORING': mooring_values[:, si,1]})
               
                
            plt.legend()
            plt.title(matchup_fname[0].split('/')[-1].split('.')[0])
            
            # Save processed values in same format as matchup
            if os.path.exists('processed_output/') == False:
                os.mkdir('processed_output/')
                
            if os.path.exists('processed_output/figs/') == False:
                os.mkdir('processed_output/figs/')
                
            plt.savefig('processed_output/figs/'+matchup_fname[0].split('/')[-1].split('.')[0]+'.jpg')
            plt.close()
                
            processed_info.to_csv('processed_output/'+matchup_fname[0].split('/')[-1])
            
        else:
            print('No matchup data found.')
            plt.close()
    