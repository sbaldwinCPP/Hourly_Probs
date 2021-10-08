# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:41:44 2021

@author: sbaldwin

new tool to calculate probabilities from hourly met data
many functions adapted from HVAC_Emissions_post.py
"""

#%% Import
print('Importing modules...')

try:
    #default packages
    import sys
    import os
    import datetime
    #non default packages
    import numpy as np
    import pandas as pd
    import easygui
    
    print('All modules imported')

except ModuleNotFoundError as err:
    input(str(err) +' \n Press enter to exit')
    
#%% Functions

#%% File selections

def Get_Met():
    #starts file selection at location of this script
    inidir = os.getcwd()    #this is better for .exe
    txt='Select met data file(s) to process...'
    ftyp= '*.csv'
    dflt=inidir + "\\" + ftyp
    filepaths = easygui.fileopenbox(default=dflt,msg=txt,filetypes=ftyp,multiple=True)
    try: Targdir=os.path.dirname(filepaths[0])
    except TypeError:
        easygui.msgbox(msg='Nothing selected, nothing to do. Goodbye!')
        sys.exit()
    return filepaths, Targdir

def Get_Fit(Targdir):
    inidir = Targdir    #look at met data location
    txt='Select fit file to process...'
    ftyp= '*.xls'
    dflt=inidir + "\\" + ftyp
    filepath = easygui.fileopenbox(default=dflt,msg=txt,filetypes=ftyp,multiple=False)
    if filepath is None:
        easygui.msgbox(msg='Nothing selected, nothing to do. Goodbye!')
        sys.exit()
    return filepath

def Get_Crit(Targdir):
    inidir = Targdir    #look at met data location
    txt='Select crit file to process...'
    ftyp= '*.xlsx'
    dflt=inidir + "\\" + ftyp
    filepath = easygui.fileopenbox(default=dflt,msg=txt,filetypes=ftyp,multiple=False)
    if filepath is None:
        easygui.msgbox(msg='Nothing selected, nothing to do. Goodbye!')
        sys.exit()
    return filepath


#%% Read files

def Read_Met(paths):
    '''
    reads single or multiple met data .csv files and combines if mulpiple
    '''
    print('Reading met file...')
    met=pd.DataFrame()
    for file in paths:
        data = pd.read_csv(file,skiprows=2,header=0) 
        met = met.append(data[['Year','Month','Day','Hour','Wind Direction','Wind Speed']].copy())
        
    met.columns=('Y','M','D','H','WD','WS')
    return met

def Read_Crit(path):
    '''
    reads crit .xlsx file, only takes values in the first column, ignore others 
    '''
    print('Reading crit file...')
    data=pd.read_excel(path,skiprows=1,header=None)
    crit=data[0].copy()
    return crit

def Read_Fit(path):
    '''
    reads either .xls or .xlsx version of a fit file
    to use .xlsx select 'all files' option in gui file selction
    '''
    
    print('Reading fit file...')
    fit_ext=os.path.splitext(path)[1]
    
    if fit_ext=='.xls':
        try:
            fits=pd.read_table(path, 
                     skiprows=4, 
                     header=None,
                     encoding='latin-1')
        except Exception as err:
            easygui.msgbox(msg="Issue loading fit file, check special character format: {}".format(err))       
    elif fit_ext=='.xlsx': 
        try:
            fits=pd.read_excel(path, 
                     skiprows=4, 
                     header=None,
                     usecols=range(17))
        except Exception as err:
            easygui.msgbox(msg="Issue loading fit file: {}".format(err))
    else: 
        easygui.msgbox(msg='File extension not recognized, program will now crash')  
        
    fits.columns=('ProjNum',
                  'RunNum',
                  'RunLet',
                  'Cm',
                  'WDc',
                  'WSc',
                  'A',
                  'B',
                  'RMSE',
                  'Bias',
                  'WDmin',
                  'WDmax',
                  'PltNum',
                  'Stack',
                  'Ht',
                  'Rec',
                  'DateTime')
    
    fits['RunID']=fits.RunNum.astype(str).copy() + fits.RunLet.copy()
    
    return fits

#%% Calculations

def Hourly_Cm(fit,met,runid):
    '''
    Generates hourly normalized concentrations for all fits with
    a given run ID (e.g. 101A) and takes the max values from any corresponding fits

    '''
    print('Calculating concentrations for run {}'.format(runid))
    hrly=met[['WD','WS']].copy()
    conc=pd.DataFrame()
    f = fit[fit.RunID==runid]
    
    for i in f.index:
        Cmax = f.Cm[i]
        WDc = f.WDc[i]
        Uc = f.WSc[i]
        A = f.A[i]
        B = f.B[i]
        pnum=f.PltNum[i]
        fitid=runid+str(pnum)

        conc[fitid]=Calc_Cm(Cmax,WDc,Uc,A,B,hrly.WD,hrly.WS)
        
    conc[runid]=conc.max(axis=1)
    return conc[runid]
    
def Calc_Cm(Cmax,WDc,Uc,A,B,WD,U):
    '''
    Calculate a single normalized concentration given fit parameters and 
    any Wind Speed and Wind Direction (single pair or series)
    '''
    try:
        #if a single WD and WS are passsed (tests)
        WD_Bias = WD-WDc
        if WD_Bias > 180: WD_Bias = 360 - WD_Bias
        elif WD_Bias < -180: WD_Bias = WD_Bias + 360
        Cm = Cmax * np.exp((-A)*((1/U)-(1/Uc))**2) * np.exp(-(((WD_Bias)/B)**2))
        return Cm
    except ValueError:
        #if a series of WD and WS are passsed
        Cm=np.zeros(len(WD))   
        count = 0
        for i in range(len(WD)):
            if (U[i] == 0) or (WD[i] == 999) or (U[i] == 999 and WD[i] == 999) or (np.isnan(U[i]))  or (np.isnan(WD[i])): 
                Cm[i] = 0
                count+=1
            elif U[i] != 0 and WD[i] != 999:
                WD_Bias = WD[i]-WDc
                if WD_Bias > 180: WD_Bias = 360 - WD_Bias
                elif WD_Bias < -180: WD_Bias = WD_Bias + 360
                Cm[i] = Cmax * np.exp((-A)*((1/U[i])-(1/Uc))**2) * np.exp(-(((WD_Bias)/B)**2))
        if count != 0: print('Number of zeroed concentrations: {}'.format(count))        
        return Cm 

#%% Operations

def Hrly_Runs(fit,met):
    '''
    Calculate max concentrations for all run numbers
    '''
    runs=fit.RunID.unique()
    data=pd.DataFrame()
    for r in runs:
        data[r]=Hourly_Cm(fit,met,r)
        
    return data
        


#%% Extras and QA
    
def Calc_DV(series, DV):
    """
    Calculate design value for a series of values given DV in percent.
    e.g. for 1% design value DV=1
    """
    print('Computing {}% value...'.format(DV))
    series=cleanup(series)  #may want to cleanup outside/before this function
    series.sort_values(inplace=True)
    series=series.reset_index(drop=True)
    val=round(series.index.size*(1-(DV/100)))
    return series[val]  

def cleanup(series):
    if 999 in series.values:
        print("Series has 999 in it:")
        series=series.replace(999,0)
    if series.isnull().values.any():
        print("Series has NaN in it:")
        series=series.fillna(0)
    #add any other cleanup item here
    return series
    

#%% Startup & File selection
print('Use GUI to select files...')    

metpaths,td=Get_Met()
fitpath=Get_Fit(td)
critpath=Get_Crit(td)

#%% Load data from files
t0 = datetime.datetime.now()

print('Reading input files...')    

met=Read_Met(metpaths)
fit=Read_Fit(fitpath)
crit=Read_Crit(critpath)

data=Hrly_Runs(fit,met)

#%% done
print('Done!')
t1=datetime.datetime.now()
dt= t1-t0
dt=dt.seconds
easygui.msgbox(msg="Done!\n Process took: {} seconds".format(dt))  
