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
    met=pd.DataFrame()
    for file in paths:
        data = pd.read_csv(file,skiprows=2,header=0) 
        met = met.append(data[['Year','Month','Day','Hour','Minute','Wind Direction','Wind Speed']].copy())
        
    met.columns=('Y','Mo','D','H','Min','WD','WS')
    return met

def Read_Fit(path):
    '''
    reads either .xls or .xlsx version of a fit file
    to use .xlsx select 'all files' option in gui file selction
    '''
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
    return fits

def Read_Crit(path):
    data=pd.read_excel(path,skiprows=1,header=None)
    crit=data[0].copy()
    return crit

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
print('Reading input files...')    

met=Read_Met(metpaths)
fit=Read_Fit(fitpath)
crit=Read_Crit(critpath)

