#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:51:38 2023

@author: epark
"""

import pandas as pd
import numpy as np
import urllib
from geopy import distance
import xarray as xr
import math
import gsw
import os
import requests
import pickle
import shutil
import matplotlib.pyplot as plt

def download_indexfiles():

    url_path = 'https://data-argo.ifremer.fr/'
    synthetic_indx = 'argo_synthetic-profile_index.txt'
    fname = url_path+synthetic_indx
    # Dowload info from website
    with urllib.request.urlopen(fname) as f:
        html = f.read().decode('utf-8')

    data = list(map(lambda x: x.split(','),html.split("\n")))

    # Drop the header and last line ('\n') split
    df = pd.DataFrame(data[9:-1],columns = data[8])

    return df

def getfloat_matchups(index_file, lat, lon, dist_thresh, start_date, end_date, params,use_local_dac=False, *local_dac_dir):
    # For a given lat-lon position & distance threshold (in km)
    # Find all the floats that match-up in the specified date time position
    # For floats with a given set of sensors

    print('Looking for match-ups...')

    s_time = start_date
    e_time = end_date
    deg_thresh = dist_thresh /111 *10 # 111 km/º

    # To imporve efficiencies, drop everything that is > 10 times farther than distance threshold
    index_file = index_file.drop(index_file.loc[index_file.loc[:,'latitude']=='','latitude'].index)
    index_file= index_file.loc[(index_file.loc[:,'latitude'].astype('float')<=lat+deg_thresh) & (index_file.loc[:,'latitude'].astype('float')>=lat-deg_thresh) &
                   (index_file.loc[:,'longitude'].astype('float')<=lon+deg_thresh) & (index_file.loc[:,'longitude'].astype('float')>=lon-deg_thresh),:]

    # Check1: Determine what floats/profiles fall in correct time, location, and parameter space
    target = (lat, lon)
    dists_flag = np.zeros(index_file.shape[0])*np.NaN
    params_flag = np.zeros(index_file.shape[0])*np.NaN
    dates_flag = np.zeros(index_file.shape[0])*np.NaN

    for i in np.arange(index_file.shape[0]):

        if index_file.loc[:,'latitude'].values[i] == '':
            plat = np.NaN
        else:
            plat = float(index_file.loc[:,'latitude'].values[i])

        if index_file.loc[:,'longitude'].values[i] == '':
            plon=np.NaN
        else:
            plon = float(index_file.loc[:,'longitude'].values[i])

        if np.isnan(plat)== False and np.isnan(plon) == False:
            float_pos = (plat, plon)
            dists = distance.distance(target, float_pos).km
        else:
            dists = np.inf

        # Check position
        if dists < dist_thresh:
            dists_flag[i] = True
        else:
            dists_flag[i] = False

        # Check parameters
        p_count = 0
        for p in params:
            if p in index_file.loc[:,'parameters'].values[i].split(' '):
                p_count = p_count + 1

        if p_count == len(params):
            params_flag[i] = True
        else:
            params_flag[i] = False

        # Check date
        if type(index_file.loc[:,'date'].values[i]) == str:
            if index_file.loc[:,'date'].values[i] == '':
                dval = np.NaN
            else:
                dval = int(index_file.loc[:,'date'].values[i])
        else:
            dval = index_file.loc[:,'date'].values[i]

        if np.isnan(dval) == False:
            pd_str = str(int(index_file.loc[:,'date'].values[i]))
            prof_date = pd.Timestamp(int(pd_str[:4]),int(pd_str[4:6]),int(pd_str[6:8]),
                                     int(pd_str[8:10]),int(pd_str[10:12]),int(pd_str[12:]))

            if prof_date>=s_time and prof_date<=e_time:
                dates_flag[i] = True
        else:
            dates_flag[i] = False

    # Drop floats outside of target distance
    check1 = dists_flag + params_flag + dates_flag
    index_file = index_file.assign(CHECK1 = check1)

    good_index = index_file.loc[index_file.loc[:,'CHECK1']==3, :]

    # If realtime and delayed mode profiles are present
    # include only delayed mode
    fnames_short = [fi.split('/')[-1].split('.')[0] for fi in good_index.loc[:,'file'].values]
    good_ggi = []
    
    ggi = 0
    while ggi < good_index.shape[0]:
        
        if ggi !=  good_index.shape[0]-1:
            
            wmo_1 = fnames_short[ggi].split('_')[0][2:]
            prof_1 = fnames_short[ggi].split('_')[-1].split('.')[0][:3]
            wmo_2 = fnames_short[ggi+1].split('_')[0][2:]
            prof_2 = fnames_short[ggi+1].split('_')[-1].split('.')[0][:3]
            
            if (wmo_1 == wmo_2) and (prof_1 == prof_2):
                
                if (fnames_short[ggi][:2] == 'SR') and (fnames_short[ggi+1][:2] == 'SR'):
                    # Save delayed mode
                    print(fnames_short[ggi], 'and', fnames_short[ggi+1], 'both present')
                    print('only including ',fnames_short[ggi+1])
                    
                    good_ggi = good_ggi+[ggi+1]
                    ggi = ggi + 2
                elif (fnames_short[ggi][:2] == 'SD') and (fnames_short[ggi+1][:2] == 'SD'):
                    # Proper format...save correct formatted one
                    print(fnames_short[ggi], 'and', fnames_short[ggi+1], 'both present')
                    print('only including ',fnames_short[ggi])
                    good_ggi = good_ggi+[ggi]
                    ggi = ggi + 2
                    
                else:
                    print('ERROR. Kill code ad debug.')
                    print(fnames_short[ggi], 'and', fnames_short[ggi+1], 'both present')
    
            else:
                good_ggi = good_ggi+[ggi]
                ggi = ggi + 1
        else:
            good_ggi = good_ggi+[ggi]
            ggi = ggi + 1

    good_index = good_index.iloc[good_ggi, :]

    print('Complete.')
    print('\nThere are ',good_index.shape[0],' profile match-ups.')
    return good_index

def QCDataByParameter(floatdata, param, datamode):
    # Given data & parameter --> return qc'd data for that
    # data: (N_PROF, N_LEVELS)
    # datamode: (N_PROF, N_PARAM)

    AllQCLevels=[b'0',b'1',b'2',b'3',b'4',b'5',b'6',b'7',b'8',b'9']
    AllQCLevels_i=[0,1,2,3,4,5,6,7,8,9]
    goodQC_flags = [1,2,5,8]
    for i in goodQC_flags:
        AllQCLevels_i.remove(i)

    raw = floatdata[param].values
    raw_qc = floatdata[param+'_QC'].values

    adj = floatdata[param+'_ADJUSTED'].values
    adj_qc = floatdata[param+'_ADJUSTED_QC'].values
    adj_error = floatdata[param+'_ADJUSTED_ERROR'].values

    # QC by each profile
    # if R: use PARAM, PARAM_QC
    # elif D/A: use PARAM_ADJUSTED, PARAM_ADJUSTED_QC
    all_data = np.zeros(raw.shape)*np.NaN
    all_data_qc = np.zeros(raw.shape)*np.NaN
    all_data_error = np.zeros(raw.shape)*np.NaN

    for k in np.arange(raw.shape[0]):
        if datamode == 'R':
                all_data[k,:] = raw[k,:]
                all_data_qc[k,:] = raw_qc[k,:]
        else:
            all_data[k,:] = adj[k,:]
            all_data_qc[k,:]  = adj_qc[k,:]
            all_data_error[k,:]  = adj_error[k,:]

    # QC the data
    all_data = pd.DataFrame(all_data)
    all_data_qc = pd.DataFrame(all_data_qc)
    all_data_error = pd.DataFrame(all_data_error)

    for badqc in AllQCLevels_i:
        #all_data.iloc[all_data_qc.iloc[:,:]==badqc,all_data_qc.iloc[:,:]==badqc]=np.NaN
        all_data=np.where(all_data_qc == badqc, np.NaN, all_data)
        all_data_error=np.where(all_data_qc == badqc, np.NaN, all_data_error)
        #all_data.replace(all_data_qc.iloc[:,:]==badqc, np.NaN)

    for i in np.arange(all_data.shape[0]):
        for j in np.arange(all_data.shape[1]):

            if math.isnan(all_data_qc.iloc[i,j]) == True and np.isnan(all_data[i,j]) == False:
                all_data[i,j]= np.NaN
                all_data_error[i,j]= np.NaN

    qc_data = all_data
    qc_flags = np.array(all_data_qc)
    qc_error = all_data_error

    return qc_data, qc_flags, qc_error

def getfloat_values_atdepth(good_index, params,d_bin, use_local_dac=False, over_write=True, *local_dac_dir):


    print('\nGetting float values at depth.')
    good_index = good_index.drop('CHECK1', axis = 1)

    p_cnames = []
    for p in params:
        p_cnames = p_cnames+[p,p+'_STD']

    all_cols = good_index.columns.to_list()+['PRES','PRES_STD','DEPTH','DEPTH_STD','SIGMA0','SIGMA0_STD','TEMP','TEMP_STD','PSAL','PSAL_STD']+p_cnames
    d_summary = pd.DataFrame(columns =all_cols)

    for i in np.arange(good_index.shape[0]):
        fname = good_index.loc[:,'file'].values[i]

        # Check if file already exits
        pname = 'matchup_output/pickle/'+fname.split('/')[-1].split('.')[0]+'.pkl'

        if os.path.exists(pname) and over_write == False:
            print(fname, 'already processed...loading pickle.')
            df = pd.read_pickle(pname)
            print(fname, 'Complete.')
        else:

            if use_local_dac == True:

                # Check is path exists/you have this copy
                if os.path.exists(local_dac_dir[0]+'dac/'+fname):
                    print('Using local copy of ', fname)
                    data = xr.open_dataset(local_dac_dir[0]+'dac/'+fname)
                else:
                    # download from the internet
                    print('File not found...downloading from internet.')
                    print('Downloading...', fname)
                    url_path = 'https://data-argo.ifremer.fr/'
                    url = url_path+'dac/'+fname
                    data = xr.open_dataset(requests.get(url).content)
                    print('Complete.')
            else:
                # Determine how to download data from the interwebs
                print('Downloading...', fname)
                url_path = 'https://data-argo.ifremer.fr/'
                url = url_path+'dac/'+fname
                data = xr.open_dataset(requests.get(url).content)
                print('Complete.')


                # Quality control P, T, S and target parameters
                all_parameters = ['PRES','TEMP','PSAL']+params
                all_values = np.zeros((data.PRES.values.shape[1],len(all_parameters)))*np.NaN
                all_values_error = np.zeros((data.PRES.values.shape[1],len(all_parameters)))*np.NaN

                all_float_p =  np.array(''.join([x.decode('utf-8') for x in list(data.PARAMETER.values[0][0])]).split(' '))
                all_float_p = all_float_p[all_float_p!='']

                all_param_mode = [[]]*(len(all_parameters)+2)
                all_param_comments = [[]]*(len(all_parameters)+2)
                for j in np.arange(len(all_parameters)):

                    param = all_parameters[j]
                    # Get parameter index
                    p_ind = np.where(all_float_p==param)[0]

                    datamode = data.PARAMETER_DATA_MODE.values[0,p_ind][0].decode('utf-8')
                    all_param_mode[j]=datamode

                    qc_data, qc_flags, qc_error = QCDataByParameter(data, param, datamode)

                    all_values[:,j] = qc_data
                    all_values_error[:,j]=qc_error
                    # Get calibration information
                    cal_info = ''
                    for jj in np.arange(data.SCIENTIFIC_CALIB_COMMENT.values.shape[1]):
                        cc = [x.decode('utf-8') for x in list(data.SCIENTIFIC_CALIB_COMMENT.values[0,jj,:])]
                        cal_info = cal_info + cc[p_ind[0]].split('  ')[0]

                    all_param_comments[j]=cal_info

                df=pd.DataFrame(all_values, columns = all_parameters)

                # Add errors
                dft=pd.DataFrame(all_values_error, columns = [x+'_ERROR' for x in all_parameters])
                df = pd.concat((df,dft),axis=1)

                z = gsw.z_from_p(df.loc[:,'PRES'].values, np.ones(df.shape[0])*data.LATITUDE.values[0])*-1
                df = df.assign(DEPTH = z)

                # Calculate potential density
                SA = gsw.SA_from_SP(df.loc[:,'PSAL'].values, df.loc[:,'PRES'].values,
                                    np.ones(df.shape[0])*data.LONGITUDE.values[0],np.ones(df.shape[0])*data.LATITUDE.values[0])
                CT = gsw.CT_from_t(SA, df.loc[:,'TEMP'].values, df.loc[:,'PRES'].values)
                sigma0  = gsw.sigma0(SA, CT)+1000
                df = df.assign(SIGMA0 = sigma0)

                dft=pd.DataFrame(np.array(all_param_mode,dtype='object').reshape(1,-1), columns = all_parameters+['DEPTH','SIGMA0'])
                df = pd.concat((dft,df), ignore_index=True)

                dft=pd.DataFrame(np.array(all_param_comments,dtype='object').reshape(1,-1), columns = all_parameters+['DEPTH','SIGMA0'])
                df = pd.concat((dft,df), ignore_index=True)



                # Save as pickle
                print('Saving processed data to pickle...')
                df.to_pickle(pname)
                print('Complete.')

                data.close()

        # For each specified depth bin
        for d_type in list(d_bin.keys()):
            if 'DEPTH' in d_type:
                d_var = 'DEPTH'
            elif 'PRES' in d_type:
                d_var = 'PRES'
            elif 'SIGMA0' in d_type:
                d_var = 'SIGMA0'

            comments = df.iloc[0,:]
            data_mode = df.iloc[1,:]
            df = df.iloc[2:,:]

            subset = df.loc[(df.loc[:,d_var]>=d_bin[d_type][1]) & (df.loc[:,d_var]<=d_bin[d_type][2]),:]

            s_mean = subset.mean()
            s_std = subset.std()

            # Save data
            core_df = pd.Series({'PRES': s_mean.loc['PRES'],
                                 'PRES_STD': s_std.loc['PRES'],
                                 'PRES_ERROR': s_mean.loc['PRES_ERROR'],
                                 'PRES_ERROR_STD': s_std.loc['PRES_ERROR'],
                                 'PRES_COMMENT': comments.loc['PRES'],
                                 'PRES_MODE': data_mode.loc['PRES'],
                                 'DEPTH': s_mean.loc['DEPTH'],
                                 'DEPTH_STD': s_std.loc['DEPTH'],
                                 'SIGMA0': s_mean.loc['SIGMA0'],
                                 'SIGMA0_STD': s_std.loc['SIGMA0'],
                                 'TEMP': s_mean.loc['TEMP'],
                                 'TEMP_STD': s_std.loc['TEMP'],
                                 'TEMP_ERROR': s_mean.loc['TEMP_ERROR'],
                                 'TEMP_ERROR_STD': s_std.loc['TEMP_ERROR'],
                                 'TEMP_COMMENT': comments.loc['TEMP'],
                                 'TEMP_MODE': data_mode.loc['TEMP'],
                                 'PSAL': s_mean.loc['PSAL'],
                                 'PSAL_STD': s_std.loc['PSAL'],
                                 'PSAL_ERROR': s_mean.loc['PSAL_ERROR'],
                                 'PSAL_ERROR_STD': s_std.loc['PSAL_ERROR'],
                                 'PSAL_COMMENT': comments.loc['PSAL'],
                                 'PSAL_MODE': data_mode.loc['PSAL']
                                 })

            for p in params:
                p_df = pd.Series({p: s_mean.loc[p],
                                  p+'_STD': s_std.loc[p],
                                  p+'_ERROR':s_mean.loc[p+'_ERROR'],
                                  p+'_ERROR_STD':s_std.loc[p+'_ERROR'],
                                  p+'_COMMENT': comments.loc[p],
                                  p+'_MODE': data_mode.loc[p]
                                  })

                core_df = pd.concat((core_df, p_df))


            dft_all=pd.concat((good_index.iloc[i,:], core_df))

            d_summary = pd.concat((d_summary, dft_all.to_frame().T))


    d_summary = d_summary.reset_index(drop=True)

    print('Complete.')
    return d_summary


def do_matchups(matchup_info, params, dist_thresh, d_type, d_thresh, over_write = False,use_local_dac=False, *local_dac_dir):

    if over_write == True:
        print('Clearing matchup_output directory.')
        shutil.rmtree('matchup_output/')
        print('Complete.')

    if os.path.exists('matchup_output/') == False:
        os.mkdir('matchup_output/')
    if os.path.exists('matchup_output/figs/') == False:
        os.mkdir('matchup_output/figs/')
    if os.path.exists('matchup_output/pickle/') == False:
        os.mkdir('matchup_output/pickle/')

    if use_local_dac == True:
        print('Using local copy of index file.')
        index_file = pd.read_csv(local_dac_dir[0]+'argo_synthetic-profile_index.txt', skiprows=8)
    else:
        print('Downloading index file.')
        index_file = download_indexfiles()
        print('Complete.')

    for i in np.arange(matchup_info.shape[0]):
    #for i in np.arange(matchup_info.shape[0])[65:66]:
        out_fname = matchup_info.loc[:,'OUT_FNAME'].values[i]

        print('\nMATCH UPS FOR: ', out_fname)

        #################################
        # Format lat, lon, and date bounds
        #################################

        lat = matchup_info.loc[:,'LATITUDE'].values[i]
        lon = matchup_info.loc[:,'LONGITUDE'].values[i]

        start_date = matchup_info.loc[:,'START_DATE'].values[i]
        end_date = matchup_info.loc[:,'END_DATE'].values[i]

        if type(start_date) == str and ('nan' in start_date) == False:
            start_date = pd.Timestamp(int(start_date.split(' ')[0].split('-')[0]),
                                      int(start_date.split(' ')[0].split('-')[1]),
                                      int(start_date.split(' ')[0].split('-')[2]),
                                      int(start_date.split(' ')[1].split(':')[0]),
                                      int(start_date.split(' ')[1].split(':')[1]),
                                      int(start_date.split(' ')[1].split(':')[2]))

        if type(end_date) == str and ('nan' in end_date) == False:
            end_date = pd.Timestamp(int(end_date.split(' ')[0].split('-')[0]),
                                    int(end_date.split(' ')[0].split('-')[1]),
                                    int(end_date.split(' ')[0].split('-')[2]),
                                    int(end_date.split(' ')[1].split(':')[0]),
                                    int(end_date.split(' ')[1].split(':')[1]),
                                    int(end_date.split(' ')[1].split(':')[2]))

        # If date range is valid, continue with analysis
        if type(start_date) == pd._libs.tslibs.timestamps.Timestamp and type(end_date) == pd._libs.tslibs.timestamps.Timestamp:

            # 'PRES' or 'DEPTH', bin val, min val, max val
            d_val = matchup_info.loc[:,d_type].values[i]
            d_bin = {d_type: [d_val, d_val-d_thresh, d_val+d_thresh]}

            if use_local_dac == False:
                # If not using local copy of gdac, will do all from online
                good_index = getfloat_matchups(index_file, lat, lon, dist_thresh, start_date, end_date, params)
                d_summary = getfloat_values_atdepth(good_index, params,d_bin, use_local_dac, over_write)
            else:
                # Use local copy, must specify path
                good_index = getfloat_matchups(index_file,lat, lon, dist_thresh, start_date, end_date, params, use_local_dac,local_dac_dir)
                d_summary = getfloat_values_atdepth(good_index, params,d_bin, use_local_dac, over_write,local_dac_dir)

            # Reformat and save reformated dates because argo format is dumb/not helpful
            pd_dates = [pd.Timestamp(int(pd_str[:4]),int(pd_str[4:6]),int(pd_str[6:8]),
                                      int(pd_str[8:10]),int(pd_str[10:12]),int(pd_str[12:])) for pd_str in d_summary.loc[:,'date'].values]
            d_summary= d_summary.assign(date_format = pd_dates)

            # Drop rows if there is no good argo data
            check_names = d_summary.columns.to_list()[10:(20+2*len(params))]
            d_summary = d_summary.drop(d_summary.index.values[np.where(d_summary.loc[:,check_names].sum(axis=1)==0)])

            ############################################
            # If there are match-ups, plot and save data
            ############################################
            if d_summary.shape[0]>0:
                print('\nThere are ', d_summary.shape[0], 'good match-ups.\n')
                print('Saving data...')
                fig = plt.figure(figsize = (8,10))

                ax1 = fig.add_subplot(3,1,1)
                ax11 = ax1.twinx()
                ax2 = fig.add_subplot(3,1,2, sharex=ax1)
                ax22 = ax2.twinx()
                ax3 = fig.add_subplot(3,1,3, sharex=ax1)

                ax1.errorbar(d_summary.loc[:,'date_format'], d_summary.loc[:,'PRES'], yerr= d_summary.loc[:,'PRES_STD'], capsize = 5, linestyle = 'none')
                ax11.errorbar(d_summary.loc[:,'date_format'], d_summary.loc[:,'SIGMA0'], yerr= d_summary.loc[:,'SIGMA0_STD'], capsize = 5, linestyle = 'none', color = 'tab:orange')
                ax1.set_ylabel('Pressure (dbar)', color = 'tab:blue')
                ax11.set_ylabel('Potential Density (kg/m3)', color = 'tab:orange')
                plt.setp(ax1.get_xticklabels(), visible=False)


                ax2.errorbar(d_summary.loc[:,'date_format'], d_summary.loc[:,'TEMP'], yerr= d_summary.loc[:,'TEMP_STD'], capsize = 5, linestyle = 'none')
                ax2.set_ylabel('TEMP (ºC)', color = 'tab:blue')
                ax22.errorbar(d_summary.loc[:,'date_format'], d_summary.loc[:,'PSAL'], yerr= d_summary.loc[:,'PSAL_STD'], capsize = 5,linestyle = 'none', color = 'tab:orange')
                ax22.set_ylabel('PSAL (PSU)',color = 'tab:orange')
                plt.setp(ax2.get_xticklabels(), visible=False)


                ax3.errorbar(d_summary.loc[:,'date_format'], d_summary.loc[:,'DOXY'], yerr= d_summary.loc[:,'DOXY_STD'], capsize = 5,linestyle = 'none')
                ax3.set_ylabel('DOXY (µmol/kg)')
                ax3.set_title('Oxygen')

                ax3.tick_params(axis='x', rotation=45)
                fig.suptitle(out_fname)
                fig.subplots_adjust(hspace = 0.3)


                plt.savefig('matchup_output/figs/'+out_fname+'.jpg')
                plt.close()
                d_summary.to_csv('matchup_output/'+out_fname+'.csv')
                print('Complete.')
