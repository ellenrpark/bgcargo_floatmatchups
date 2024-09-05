#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:28:37 2024

@author: epark
"""

from getfloats_postprocessing import do_matchups_with_thresholds
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Can specify up to 3 threshold criteria
# param options ['TEMP','PSAL','DEPTH','PRES','SIGMA0']

param_list = [{'PSAL': 0.005,'THETA0': 0.005,'PRES': 100},
               {'PSAL': 0.01,'THETA0': 0.01,'PRES': 10},
               {'SIGMA0': 0.01, 'PRES': 100}]

days = 5
time_thresh = 60*24*days # minutes; +/- time window from float profile

doxy_type = 'DOXY_drift_corrected'

mooring_flist = sorted(glob.glob('/Volumes/FATDATABABY/OSNAP grid interpolation/Ungridded oxygen/*.csv'))


vals = ['ARGO_ERR_MIN','ARGO_ERR_MAX',
                                     'OVERALL_ER','OVERALL_RE', 'OVERALL_N','OVERALL_NP',
                                     'AIR_ER','AIR_RE', 'AIR_N','AIR_NP',
                                     'CTD_NCEP_ER','CTD_NCEP_RE', 'CTD_NCEP_N','CTD_NCEP_NP',
                                     'CTD_WOA_ER','CTD_WOA_RE', 'CTD_WOA_N','CTD_WOA_NP',
                                     'WOA_ER','WOA_RE', 'WOA_N','WOA_NP',]

qc_results = pd.DataFrame(columns = vals)

for param_thresh in param_list:

    # Do match ups
    do_matchups_with_thresholds(mooring_flist, param_thresh, time_thresh, doxy_type)

    # Get statistics
    # Load match up results
    flist = sorted(glob.glob('/Users/epark/Documents/GitHub/bgcargo_floatmatchups/processed_output/*.csv'))

    colnames = ['file','latitude','longitude','date_format',
                'DEPTH', 'TEMP','PSAL',
     'DOXY','DOXY_STD','DOXY_ERROR','DOXY_ERROR_STD',
     'DOXY_COMMENT','DOXY_MODE',
     'DOXY_MOORING','DOXY_STD_MOORING',
     'TEMP_MOORING','PSAL_MOORING',
     'MOORING','date']


    # Concatenate all results
    df_all = pd.DataFrame(columns = colnames)


    for fname in flist:
        data  = pd.read_csv(fname)

        data = data.assign(MOORING = [fname.split('_')[-2]]*data.shape[0])

        df_all = pd.concat((df_all, data.loc[:,colnames]))

    df_all =df_all.loc[(df_all.loc[:,'DOXY'].isna() == False) & \
                       (df_all.loc[:,'DOXY_MOORING'].isna() == False), :]
    a=np.array([pd.Timestamp(ti)<pd.Timestamp(2022,7,1) for ti in\
                df_all.loc[:,'date_format'].values])

    df_all = df_all.iloc[np.where(a==True)[0], :]

    df_all = df_all.reset_index(drop=True)
    df_all = df_all.assign(ARGO_ERROR=df_all.loc[:,'DOXY_ERROR']/df_all.loc[:,'DOXY']*100)
    df_all = df_all.assign(MOORING_ARGO_ERROR= \
                           abs((df_all.loc[:,'DOXY_MOORING'] - df_all.loc[:,'DOXY']))/df_all.loc[:,'DOXY_MOORING']*100)
    df_all = df_all.assign(MOORING_ARGO_DIF= df_all.loc[:,'DOXY_MOORING'] - df_all.loc[:,'DOXY'])

    # Sort by QC type
    QC_type = [[]]*df_all.shape[0]
    QC_type_val = np.zeros(df_all.shape[0])*np.NaN

    m_list = sorted(df_all.loc[:,'MOORING'].unique())
    m_type = np.zeros(df_all.shape[0])*np.NaN

    for di in np.arange(df_all.shape[0]):

        m_type[di] =m_list.index(df_all.loc[:,'MOORING'].values[di])
        a = df_all.loc[:,'DOXY_COMMENT'].values[di]

        if a == 'Adjustment done on PPOX_DOXY;Temporal drift estimated from NCEP data '+ \
            '- Gain based on comparison between Argo cycle 1 and reference profile ov18_d_102/':
            cal_type= 'CTD_NCEP'
            QC_type_val[di]=3

        elif a== 'Adjustment done on PPOX_DOXY;Temporal drift estimated from NCEP data - '+\
            'Gain based on comparison between Argo cycle 3 and reference profile ov18_d_105/':
                cal_type ='CTD_NCEP'
                QC_type_val[di]=3
        elif a == 'Bittig et al. (2021); float in-air data vs. NCEP reanalysis':
            cal_type = 'AIR'
            QC_type_val[di]=2

        elif 'DOXY_ADJUSTED' in a:

            if 'DM adjustment' in a:
                cal_type = 'ADJ_DM'
                QC_type_val[di]=0
            else:
                cal_type='ADJ_WOA'
                QC_type_val[di]=1

        elif a == 'Polynomial calibration coeficients were used. G determined by surface '+\
            'measurement comparison to World Ocean Atlas 2009.See Takeshita et al.2013,doi:10.1002/jgrc.20399':
                cal_type = 'WOA'
                QC_type_val[di]=5
        elif a == 'Pressure effect correction on DOXYAdjustment done on PPOX_DOXY;'+\
            'Temporal drift estimated from NCEP data - Gain based on comparison between Argo '+\
                'cycle 1 and reference profile ov18_d_81/':
            cal_type = 'CTD_NCEP'
            QC_type_val[di]=3
        elif a == 'Pressure effect correction on DOXYAdjustment done on PPOX_DOXY;Temporal'+\
            ' drift estimated from WOA climatology at surface - Gain based on comparison between Argo '+\
                'cycle 1 and reference profile ov18_d_77/':
            cal_type = 'CTD_WOA'
            QC_type_val[di]=4
        else:
            cal_type = 'NONE'
            QC_type_val[di]=6

        QC_type[di]=cal_type

    df_all = df_all.assign(QC_TYPE=QC_type)
    df_all = df_all.assign(QC_TYPE_VAL=QC_type_val)
    df_all = df_all.assign(WMO = [int(fi.split('/')[1]) for fi in df_all.loc[:,'file'].values])


    adjusted = df_all.loc[df_all.loc[:,'DOXY_MODE']=='A',:]
    delayed = df_all.loc[df_all.loc[:,'DOXY_MODE']=='D',:]

    ##

    results = [delayed.loc[:,'ARGO_ERROR'].min(), delayed.loc[:,'ARGO_ERROR'].max(),
               delayed.loc[:,'MOORING_ARGO_DIF'].median(),
               delayed.loc[:,'MOORING_ARGO_ERROR'].median(),
               delayed.shape[0],
               delayed.loc[:,'file'].unique().shape[0]]

    m_list = ['AIR','CTD_NCEP','CTD_WOA','WOA']

    for qctype in m_list:
        subset = df_all.loc[df_all.loc[:,'QC_TYPE']==qctype]

        if subset.shape[0]>0:
            subset_results  = [subset.loc[:,'MOORING_ARGO_DIF'].median(),
                               subset.loc[:,'MOORING_ARGO_ERROR'].median(),
                               subset.shape[0],
                               subset.loc[:,'file'].unique().shape[0]]
        else:
            subset_results = [np.NaN, np.NaN, np.NaN, np.NaN]

        results = results + subset_results

    qc_results = pd.concat((qc_results, pd.DataFrame(np.array(results).reshape(1,-1), columns = vals)), axis = 0)

qc_results.to_csv('other/qcmatchupresults.csv')
