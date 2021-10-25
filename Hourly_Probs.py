# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:41:44 2021

@author: sbaldwin

new tool to calculate probabilities from hourly met data
many functions adapted from HVAC_Emissions_post.py
and extended_outfiles.py
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
    added option to load .txt format
    '''
    print('Reading met file...')
    
    met_ext=os.path.splitext(paths[0])[1]
    met=pd.DataFrame()
    
    if met_ext=='.csv':
        try:
            for file in paths:
                data = pd.read_csv(file,skiprows=2,header=0) 
                met = met.append(data[['Year','Month','Day','Hour','Wind Direction','Wind Speed']].copy())
            met.columns=('Y','M','D','H','WD','WS')
        except Exception as err:
            easygui.msgbox(msg="Issue loading met file: {}".format(err))
            sys.exit()
            
    elif met_ext=='.txt':
        try:
            data = pd.read_table(paths[0],skiprows=3,header=None)
            data.columns=['Day',
                          'Month',
                          'Year',
                          'Hour',
                          'Temp',
                          'Hmds',
                          'Wind Direction',
                          'Wind Speed',
                          'WB',
                          'CT_LR',
                          'CT_T',
                          ]
            met=data[['Year','Month','Day','Hour','Wind Direction','Wind Speed']].copy()
            met.columns=('Y','M','D','H','WD','WS')  
        except Exception as err:
            easygui.msgbox(msg="Issue loading met file: {}".format(err))
            sys.exit()
    else: 
        easygui.msgbox(msg='File extension not recognized, program will now crash')
        sys.exit()
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
            sys.exit()
    elif fit_ext=='.xlsx': 
        try:
            fits=pd.read_excel(path, 
                     skiprows=4, 
                     header=None,
                     usecols=range(17))
        except Exception as err:
            easygui.msgbox(msg="Issue loading fit file: {}".format(err))
            sys.exit()
    else: 
        easygui.msgbox(msg='File extension not recognized, program will now crash')
        sys.exit()
        
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
    Calculate normalized concentration given fit parameters and 
    any Wind Speed and Wind Direction (single pair or series)
    '''
    try:
        #if a single WD and WS are passsed (tests)
        WD_Bias = WD-WDc
        if WD_Bias > 180: WD_Bias = 360 - WD_Bias
        elif WD_Bias < -180: WD_Bias = WD_Bias + 360
        if U==0: Cm=0
        else: Cm = Cmax * np.exp((-A)*((1/U)-(1/Uc))**2) * np.exp(-(((WD_Bias)/B)**2))
        return Cm
    except ValueError:
        #if a series of WD and WS are passsed
        Cm=np.zeros(len(WD))   
        for i in range(len(WD)):
            if U[i] == 0: 
                Cm[i] = 0
            else:
                WD_Bias = WD[i]-WDc
                if WD_Bias > 180: WD_Bias = 360 - WD_Bias
                elif WD_Bias < -180: WD_Bias = WD_Bias + 360
                Cm[i] = Cmax * np.exp((-A)*((1/U[i])-(1/Uc))**2) * np.exp(-(((WD_Bias)/B)**2))      
        return Cm 
    
    
def Calc_Prob(series, crit):
    """
    Calculate probability for a series of values exceeding given criteria in Cm.
    """
    #run=series.name
    n=len(series)
    c=sum(series>crit)
    prob=c/n
    return prob  

#%% Operations

def Hrly_Runs(fit,met):
    '''
    Calculate max concentrations for all run numbers
    '''
    print('Calculating hourly Cm...')
    runs=fit.RunID.unique()
    data=pd.DataFrame()
    for r in runs:
        data[r]=Hourly_Cm(fit,met,r)
    return data
        
def All_Probs(fit,crit,hrly):
    '''
    Create results dataframe with probs for each run/crit combo

    '''
    print('Calculating probs...')
    runs=fit.RunID.unique()
    crits=crit.unique()
    IDs=[]
    cols=['RunNum','RunLet','Cmax','WDc','WSc','Crit','Prob','Hrs']
    
    # first loop to create list of IDs
    for c in crits:
        for r in runs:
            ID=str(r)+str(c)
            IDs.append(ID)
            
    # create empty dataframe to fill 
    data=pd.DataFrame(index=IDs,columns=cols)      
            
    # second loop to fill out results
    for r in runs:
        runnum=r[:3]
        runlet=r[len(r)-1]
        cmax=fit[fit.RunID==r].Cm.values.astype(int)
        wdc=fit[fit.RunID==r].WDc.values.astype(int)
        wsc=fit[fit.RunID==r].WSc.values
        
        series=hrly[r]
        
        for c in crits:
            ID=str(r)+str(c)
            data.loc[ID,'Prob']=Calc_Prob(series,c)
            data.loc[ID,'RunNum']=runnum
            data.loc[ID,'RunLet']=runlet
            data.loc[ID,'Crit']=c
            data.loc[ID,'Cmax']=cmax
            data.loc[ID,'WDc']=wdc
            data.loc[ID,'WSc']=wsc
            
    data.Hrs=data.Prob*365*24 
    return data
    

#%% Extras and QA
    
def Calc_DV(s, DV):
    """
    Calculate design value for a series of values given DV in percent.
    e.g. for 1% design value DV=1
    """
    #print('Computing {}% value...'.format(DV))
    #series=cleanup(series)  #may want to cleanup outside/before this function
    series=s.copy()
    series.sort_values(inplace=True)
    series=series.reset_index(drop=True)
    val=round(series.index.size*(1-(DV/100)))
    return round(series[val],2)


def MetQA(met):
    df=met.copy()
    n=len(df)
    print('{} hours in met file(s) ({} years)'.format(n,round(n/8760,1)))
    
    #data QA
    print('\nThe following data has been converted zero WS:')
    # check for 999 or WD greater than 360 
    c=sum(df.WD>360)
    p=round(c/n*100,2)
    print("{} 999's in WD ({}%)".format(c,p))
    df.loc[df[df.WD==999].index,['WS','WD']]=0,999
    
    # check for empty cells in WD
    c=sum(df.WD.isnull())
    p=round(c/n*100,2)
    print("{} NaN's in WD ({}%)".format(c,p))
    df.loc[df[df.WD.isnull()].index,['WS','WD']]=0,999
    
    # check for 999 in WS
    c=sum(df.WS==999)
    p=round(c/n*100,2)
    print("{} 999's in WS ({}%)".format(c,p))
    df.loc[df[df.WS==999].index,'WS']=0
    
    # check for empty cells in WS
    c=sum(df.WS.isnull())
    p=round(c/n*100,2)
    print("{} NaN's in WS ({}%)".format(c,p))
    df.WS=df.WS.fillna(0)
    
    # check for outliers in WS
    c=sum(df.WS>45)
    p=round(c/n*100,2)
    print("{} hours WS exceeds 45 m/s (100 mph)".format(c))
    df.loc[df[df.WS>45].index,['WS','WD']]=0,999
    
    # Stats
    print('\nFinal met data stats:')
    # Zero Wind
    c=sum(df.WS==0)
    p=round(c/n*100,2)
    print('{} zero hours (WS=0) ({}%)'.format(c,p))
    
    # Calm Winds
    c=sum(df.WS<1)
    p=round(c/n*100,2)
    print('{} calm hours (WS<1) ({}%)'.format(c,p))
    
    # Max WS
    print('Max WS is: {} m/s'.format(max(df.WS)))
    
    # 1% Wind
    DV=1
    print('{}% WS is: {} m/s'.format(DV,Calc_DV(df.WS,DV)))
    # 5% Wind
    DV=5
    print('{}% WS is: {} m/s'.format(DV,Calc_DV(df.WS,DV)))

    df.reset_index(drop=True,inplace=True)
    
    YN=easygui.ynbox('See met data stats in command window.\nDo you want to continue?')
    if YN: return df
    else: sys.exit()
    

#%% Startup & File selection
print('Use GUI to select met data file(s)...')    

metpaths,td=Get_Met()
met=Read_Met(metpaths)
met_QA= MetQA(met)

print('Use GUI to select fit and crit files...') 
fitpath=Get_Fit(td)
critpath=Get_Crit(td)

label=easygui.enterbox('Enter a label for output file:')
if label is None: sys.exit()

#%% Load data from files
t0 = datetime.datetime.now() 
print('Reading fit and crit files...')    
fit=Read_Fit(fitpath)
crit=Read_Crit(critpath)

fitname=os.path.basename(fitpath)
proj=fitname[:fitname.find('fit')]

#%% Calculate results
hrly=Hrly_Runs(fit,met_QA)
results=All_Probs(fit,crit,hrly)

#%% Save
t1=datetime.datetime.now()

savepath=os.path.join(os.getcwd(),proj+'_probs'+label+'.csv')
results.to_csv(savepath)
print('Saved...')

#%% done
print('Done!')
dt= t1-t0
dt=dt.seconds
easygui.msgbox(msg="Done!\nProcess took: {} seconds\nResults saved here: {}".format(dt,savepath))  
