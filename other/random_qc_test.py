#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:01:05 2024

@author: epark
"""

import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 12})

flist = sorted(glob.glob('/Users/epark/Documents/GitHub/bgcargo_floatmatchups/processed_output/*.csv'))

colnames = ['file','latitude','longitude','date_format',
            'DEPTH', 'TEMP','PSAL',
 'DOXY','DOXY_STD','DOXY_ERROR','DOXY_ERROR_STD',
 'DOXY_COMMENT','DOXY_MODE',
 'DOXY_MOORING','DOXY_STD_MOORING', 
 'TEMP_MOORING','PSAL_MOORING',
 'MOORING','date']

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

fig = plt.figure(figsize = (10,6))
r=1; c =2

ax1 = fig.add_subplot(r,c,1)

x = adjusted.loc[:,'ARGO_ERROR']
y = adjusted.loc[:,'MOORING_ARGO_ERROR']

ax1.scatter(x,y)
ax1.set_xlabel('ARGO DOXY ERROR (%)')
ax1.set_ylabel('RELATIVE ERROR')
ts = 'ADJUSTED\nMEDIAN REL ERROR: '+str(np.round(np.nanmedian(y),2))
ax1.set_title(ts)

ax2 = fig.add_subplot(r,c,2, sharex=ax1, sharey=ax1)
x = delayed.loc[:,'ARGO_ERROR']
y = delayed.loc[:,'MOORING_ARGO_ERROR']

ax2.scatter(x, y)
ax2.set_xlabel('ARGO DOXY ERROR (%)')
ts = 'DELAYD\nMEDIAN REL ERROR: '+str(np.round(np.nanmedian(y),2))
ax2.set_title(ts)

plt.figure(figsize = (6.5, 4))
x = df_all.groupby('QC_TYPE').median().index.values
y = df_all.groupby('QC_TYPE').median().loc[:,'MOORING_ARGO_ERROR']
y_minus = y- df_all.groupby('QC_TYPE').quantile(0.25).loc[:,'MOORING_ARGO_ERROR']
y_plus =  df_all.groupby('QC_TYPE').quantile(0.75).loc[:,'MOORING_ARGO_ERROR']-y

plt.bar(x, y, label = 'MEDIAN')
plt.errorbar(x,y,yerr=[y_minus, y_plus],
             linestyle = 'None', color = 'k',
             capsize = 5, label = '25-75th Q')
plt.legend()
plt.ylabel('RELATIVE ERROR (%)')

plt.figure(figsize = (6.5, 4))
x = df_all.groupby('QC_TYPE').median().index.values
y = df_all.groupby('QC_TYPE').median().loc[:,'MOORING_ARGO_DIF']
y_minus =  y-df_all.groupby('QC_TYPE').quantile(0.25).loc[:,'MOORING_ARGO_DIF']
y_plus =  df_all.groupby('QC_TYPE').quantile(0.75).loc[:,'MOORING_ARGO_DIF']-y

plt.bar(x, y, label = 'MEDIAN')
plt.errorbar(x,y,yerr=[y_minus, y_plus],
             linestyle = 'None', color = 'k',
             capsize = 5, label = '25-75th Q')
plt.legend()
plt.ylabel('MOORING - FLOAT (µmol/kg)')

cmm = plt.get_cmap('tab20', len(m_list))

#######
######
######
type_list = [m_list,
             ['ADJ_DM', 'ADJ_WOA','AIR','CTD_NCEP','CTD_WOA','WOA'],
             sorted(df_all.loc[:,'WMO'].unique())]
param_list = ['MOORING','QC_TYPE', 'WMO']

for pi in np.arange(len(type_list)):
    m_list = type_list[pi]
    param = param_list[pi]
    
    fig = plt.figure(figsize=(25,10))
    
    ax = fig.add_subplot(1,3,1)
    ax.grid()
    ax.plot([260,320],[260, 320],'k-')
    
    for mi in np.arange(len(m_list)):
        inds = np.where(df_all.loc[:,param].values == m_list[mi])[0]
        if inds.shape[0]>0:
            ax.scatter(df_all.loc[:,'DOXY'].values[inds],
                        df_all.loc[:,'DOXY_MOORING'].values[inds],
                        c = cmm(np.ones(inds.shape[0],dtype=int)*mi),
                    label = m_list[mi])
    ax.legend(ncol=2, fontsize = 16)
    
    ts = 'ALL\nMEDIAN DIF: '+\
        str(np.round(np.nanmedian(df_all.loc[:,'DOXY_MOORING'].values-df_all.loc[:,'DOXY'].values),2))
    ax.set_title(ts)
    ax.set_xlabel('ARGO DOXY (µmol/kg)')
    ax.set_ylabel('MOORING DOXY (µmol/kg)')
    
    
    ax = fig.add_subplot(1,3,2)
    ax.grid()
    ax.plot([260,320],[260, 320],'k-')
    
    for mi in np.arange(len(m_list)):
        inds = np.where(adjusted.loc[:,param].values == m_list[mi])[0]
        if inds.shape[0]>0:
            ax.scatter(adjusted.loc[:,'DOXY'].values[inds],
                        adjusted.loc[:,'DOXY_MOORING'].values[inds],
                        c = cmm(np.ones(inds.shape[0],dtype=int)*mi),
                    label = m_list[mi])
    ts = 'ADJUSTED\nMEDIAN DIF: '+\
        str(np.round(np.nanmedian(adjusted.loc[:,'DOXY_MOORING'].values-adjusted.loc[:,'DOXY'].values),2))
    ax.set_title(ts)
    ax.set_xlabel('ARGO DOXY (µmol/kg)')
    ax.set_ylabel('MOORING DOXY (µmol/kg)')
    
    ax = fig.add_subplot(1,3,3)
    ax.grid()
    ax.plot([260,320],[260, 320],'k-')
    
    for mi in np.arange(len(m_list)):
        inds = np.where(delayed.loc[:,param].values == m_list[mi])[0]
        if inds.shape[0]>0:
            ax.scatter(delayed.loc[:,'DOXY'].values[inds],
                        delayed.loc[:,'DOXY_MOORING'].values[inds],
                        c = cmm(np.ones(inds.shape[0],dtype=int)*mi),
                    label = m_list[mi])
    ts = 'DELAYED\nMEDIAN DIF: '+\
        str(np.round(np.nanmedian(delayed.loc[:,'DOXY_MOORING'].values-delayed.loc[:,'DOXY'].values),2))
    ax.set_title(ts)
    ax.set_xlabel('ARGO DOXY (µmol/kg)')
    ax.set_ylabel('MOORING DOXY (µmol/kg)')
    
    fig.subplots_adjust(wspace = 0.4)
    fig.savefig('fig/'+param_list[pi]+'.jpg')

fig = plt.figure(figsize = (15, 8))

ax1 = fig.add_subplot(1,3,1)
x = df_all.groupby('WMO').median().index.values.astype(str)
y = df_all.groupby('WMO').median().loc[:,'MOORING_ARGO_DIF']
y_minus =  df_all.groupby('WMO').quantile(0.25).loc[:,'MOORING_ARGO_DIF']
y_plus =  df_all.groupby('WMO').quantile(0.75).loc[:,'MOORING_ARGO_DIF']

ax1.bar(x, y, label = 'MEDIAN')
ax1.errorbar(x,y,yerr=[y_minus, y_plus],
             linestyle = 'None', color = 'k',
             capsize = 5, label = '25-75th Q')
ax1.tick_params(axis='x', labelrotation=90)
ax1.set_title('ALL')

ax2 = fig.add_subplot(1,3,2,sharey=ax1)
x = adjusted.groupby('WMO').median().index.values.astype(str)
y = adjusted.groupby('WMO').median().loc[:,'MOORING_ARGO_DIF']
y_minus =  adjusted.groupby('WMO').quantile(0.25).loc[:,'MOORING_ARGO_DIF']
y_plus =  adjusted.groupby('WMO').quantile(0.75).loc[:,'MOORING_ARGO_DIF']

ax2.bar(x, y, label = 'MEDIAN')
ax2.errorbar(x,y,yerr=[y_minus, y_plus],
             linestyle = 'None', color = 'k',
             capsize = 5, label = '25-75th Q')
ax2.tick_params(axis='x', labelrotation=90)
ax2.set_title('ADJ')

ax3 = fig.add_subplot(1,3,3,sharey=ax1)
x = delayed.groupby('WMO').median().index.values.astype(str)
y = delayed.groupby('WMO').median().loc[:,'MOORING_ARGO_DIF']
y_minus =  delayed.groupby('WMO').quantile(0.25).loc[:,'MOORING_ARGO_DIF']
y_plus =  delayed.groupby('WMO').quantile(0.75).loc[:,'MOORING_ARGO_DIF']

ax3.bar(x, y, label = 'MEDIAN')
ax3.errorbar(x,y,yerr=[y_minus, y_plus],
             linestyle = 'None', color = 'k',
             capsize = 5, label = '25-75th Q')
ax3.tick_params(axis='x', labelrotation=90)
ax3.set_title('DMQC')

###


fig = plt.figure(figsize = (15,6))

param = 'MOORING_ARGO_ERROR'
m_list = ['AIR','CTD_NCEP','CTD_WOA','WOA']

for mi in np.arange(len(m_list)):
    
    if mi == 0:
        ax = fig.add_subplot(1,len(m_list), int(mi+1))
    else:
        ax = fig.add_subplot(1,len(m_list), int(mi+1), sharey = ax)
        
    subset = df_all.loc[df_all.loc[:,'QC_TYPE']==m_list[mi]]
    x = subset.groupby('WMO').median().index.values.astype(str)
    y = subset.groupby('WMO').median().loc[:,param]
    y_minus =  subset.groupby('WMO').quantile(0.25).loc[:,param]
    y_plus =  subset.groupby('WMO').quantile(0.75).loc[:,param]

    ax.bar(x, y, label = 'MEDIAN')
    ax.errorbar(x,y,yerr=[y_minus, y_plus],
                 linestyle = 'None', color = 'k',
                 capsize = 5, label = '25-75th Q')
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_title(m_list[mi])
    
    if mi == 0:
        ax.set_ylabel(param)
    
    
########
#######
plt.figure(figsize = (6.5, 4))

x = delayed.groupby('QC_TYPE').median().index.values
y = delayed.groupby('QC_TYPE').median().loc[:,'MOORING_ARGO_ERROR']
y_minus = y- delayed.groupby('QC_TYPE').quantile(0.25).loc[:,'MOORING_ARGO_ERROR']
y_plus =  delayed.groupby('QC_TYPE').quantile(0.75).loc[:,'MOORING_ARGO_ERROR']-y

plt.bar(x, y, label = 'MEDIAN')
plt.errorbar(x,y,yerr=[y_minus, y_plus],
             linestyle = 'None', color = 'k',
             capsize = 5, label = '25-75th Q')
plt.legend()
plt.ylabel('RELATIVE ERROR (%)')

ts = 'Median Relative Error: '+str(np.round(delayed.loc[:,'MOORING_ARGO_ERROR'].median(),2))+'%'
plt.title(ts)

plt.figure(figsize = (6.5, 4))
x = delayed.groupby('QC_TYPE').median().index.values
y = delayed.groupby('QC_TYPE').median().loc[:,'MOORING_ARGO_DIF']
y_minus =  y-delayed.groupby('QC_TYPE').quantile(0.25).loc[:,'MOORING_ARGO_DIF']
y_plus =  delayed.groupby('QC_TYPE').quantile(0.75).loc[:,'MOORING_ARGO_DIF']-y

plt.bar(x, y, label = 'MEDIAN')
plt.errorbar(x,y,yerr=[y_minus, y_plus],
             linestyle = 'None', color = 'k',
             capsize = 5, label = '25-75th Q')
plt.legend()
plt.ylabel('MOORING - FLOAT (µmol/kg)')
ts = 'Median Difference: '+str(np.round(delayed.loc[:,'MOORING_ARGO_DIF'].median(),2))+' µmol/kg'
plt.title(ts)


