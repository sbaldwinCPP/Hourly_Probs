# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:41:44 2021

@author: sbaldwin

new tool to calculate probabilities from hourly met data
many functions adapted from HVAC_Emissions_post.py and extended_outfiles.py

"""

#%% Import
print('Importing modules...')

try:
    #default packages
    import sys
    import os
    import datetime
    import pickle
    #non default packages
    import numpy as np
    import pandas as pd
    import easygui
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    
    print('All modules imported')

except ModuleNotFoundError as err:
    input(str(err) +' \n Press enter to exit')
    
#%% File selections & GUIs

def Get_Met():
    #starts file selection at location of this script
    inidir = os.getcwd()    #this is better for .exe
    txt='Select met data file(s) to process...'
    ftyp='*.txt'        # File type to look for
    dflt=os.path.join(inidir, ftyp)
    filepaths=easygui.fileopenbox(txt, txt, dflt, ftyp, True)
    
    try: Targdir=os.path.dirname(filepaths[0])
    except TypeError:
        easygui.msgbox(msg='Nothing selected, nothing to do. Goodbye!')
        sys.exit()
    return filepaths, Targdir

def Get_Fit(Targdir):
    inidir = Targdir    #look at met data location
    txt='Select fit file to process...'
    ftyp= '*.xls'
    dflt=os.path.join(inidir, ftyp)
    filepath = easygui.fileopenbox(default=dflt,msg=txt,filetypes=ftyp,multiple=False)
    if filepath is None:
        easygui.msgbox(msg='Nothing selected, nothing to do. Goodbye!')
        sys.exit()
    return filepath

def Get_Crit(Targdir):
    inidir = Targdir    #look at met data location
    txt='Select crit file to process...'
    ftyp= '*.xlsx'
    dflt=os.path.join(inidir, ftyp)
    filepath = easygui.fileopenbox(default=dflt,msg=txt,filetypes=ftyp,multiple=False)
    if filepath is None:
        easygui.msgbox(msg='Nothing selected, nothing to do. Goodbye!')
        sys.exit()
    return filepath

def Get_Op_Hrs():
    """
    GUI to select operating hours to consider
    """
    choices=range(24)
    msg='Select Operating Hours'
    print('Select operating hours from list:')
    op_hrs= easygui.multchoicebox(msg,msg,choices,preselect=choices)
    if  op_hrs is None: sys.exit()
    # convert to int
    op_hrs = [int(i) for i in op_hrs]
    return op_hrs

#%% Read files

def Read_Met(paths):
    '''
    reads single or multiple met data .csv files and combines if mulpiple.
    added option to load .txt format, changed to default
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
    data=data.dropna(how='all') 
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
            easygui.msgbox(msg="Issue loading fit file, check special character format:\n{}".format(err))   
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
        easygui.msgbox(msg='File extension not recognized, program will now close')
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
    
    # pre-processing
    fits=fits.dropna(how='all')             # sometimes empty rows get picked up
    fits.RunNum=fits.RunNum.astype(int)     # force to int for consistency
    fits.PltNum=fits.PltNum.astype(int)     # ^
    fits=fits.drop(columns='DateTime')      # never gets used, drop when reading
    
    # create run and fit IDs for lookups
    fits['RunID']=fits.RunNum.astype(str).copy() + fits.RunLet.copy()
    fits['FitID']= fits.RunID.copy() + fits.PltNum.astype(str).copy()

    return fits

#%% Calculations

def Hourly_Cm(fit,met,runid):
    '''
    Generates hourly normalized concentrations for all fits with
    a given run ID (e.g. 101A) and takes the max values by hour

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
    c=sum(series>=crit)
    prob=c/n
    return prob 

def Calc_DV(series, DV):
    """
    Calculate design value for a series of values given DV in percent.
    e.g. 1% design value DV=1
    """
    s=series.copy()
    s.sort_values(inplace=True)
    s=s.reset_index(drop=True)
    val=round(s.index.size*(1-(DV/100)))
    return round(s[val],2)

def Calc_prob_WS(WS, WD, met, wd_band):
    """
    Calc % time is exceeded for a specific WD
    +- bounds
    """
    n=len(met)                          # total hours

    # take +- deg on either side of WD
    wdmin=WD-wd_band
    wdmax=WD+wd_band

    d=met.copy()
    
    # correct wrap accross 0 if needed
    if wdmax >= 360: d=d[~d.WD.between(wdmax-360,wdmin)]
    elif wdmin <= 0: d=d[~d.WD.between(wdmax,wdmin+360)]
    else: d=d[d.WD.between(wdmin,wdmax)] 

    c=sum(d.WS>=WS)                     # count # of hrs at or above WS
    p=c/n*100                           # calc percent of total hours
        
    return p   

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
        
def All_Probs(fit,crit,hrly,met):
    '''
    Create results dataframe with probs for each run/crit combo

    '''
    print('Calculating probs...')
    runs=fit.RunID.unique()
    crits=crit.unique()
    IDs=[]
    cols=['RunNum','RunLet','Cmax','WDc','WSc','Crit','Prob','Hrs']
    
    # TODO: work in progress, on the fence between two methods 
    ####vvvv    
    # specify operating hours to consider
    op_hrs=Get_Op_Hrs()
    h=hrly.copy()
    
# =============================================================================
#     # 1:    set Cm outside operating hours to 0
#     #       includes all hours in probs
#     #       Cannot make probs worse
#     h[~met.H.isin(op_hrs)]=0
# =============================================================================
    
    # 2:    limit data to only operating hou6-18rs
    #       includes only op hours in probs
    #       can make probs worse
    h=h[met.H.isin(op_hrs)]
    
    ####^^^^
    
    # first loop to create list of IDs
    for c in crits:
        for r in runs:
            ID=str(r)+str(c)
            IDs.append(ID)
            
    # create empty dataframe to fill 
    data=pd.DataFrame(index=IDs,columns=cols)      
            
    # second loop to fill out results
    for r in runs:
        r=str(r)
        runnum=r[:3]
        runlet=r[len(r)-1]
        cmax=fit[fit.RunID==r].Cm.values.astype(int)
        wdc=fit[fit.RunID==r].WDc.values.astype(int)
        wsc=fit[fit.RunID==r].WSc.values
        
        series=h[r]
        
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
    return data, op_hrs


#%% QA & plots

def MetQA(met):
    """
    Perform all QA checks and data filtering
    """
    df=met.copy()
    n=len(df)
    print('{} hours in met file(s) ({} years)'.format(n,round(n/8760,1)))
    
    ### data QA
    print('\nThe following data has been converted to zero WS:')
    
    # check for 999 or WD greater than 360 
    c=sum(df.WD>360)
    p=round(c/n*100,2)
    print("{} bad hrs in WD ({}%)".format(c,p))
    df.loc[df[df.WD>360].index,['WS','WD']]=0,999
    
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
    
    ### stats
    print('\nFinal met data stats:')
    # Zero Wind
    c=sum(df.WS==0)         #count
    p=round(c/n*100,2)      #percent
    print('{} zero hours (WS=0) ({}%)'.format(c,p))
    
    # Calm Winds
    c=sum(df.WS<1)
    p=round(c/n*100,2)
    print('{} calm hours (WS<1) ({}%)'.format(c,p))
    
    # Outliers in WS
    c=sum(df.WS>45)
    p=round(c/n*100,2)
    print("{} hours WS exceeds 45 m/s (100 mph)".format(c))
    #df.loc[df[df.WS>45].index,['WS','WD']]=0,999
    
    # Max WS
    print('Max WS is: {} m/s'.format(max(df.WS)))
    
    # 1% Wind
    print('{}% WS is: {} m/s'.format(1,Calc_DV(df.WS,1)))
    
    # 5% Wind
    print('{}% WS is: {} m/s'.format(5,Calc_DV(df.WS,5)))

    df.reset_index(drop=True,inplace=True)
    
    if not os.path.exists(plot_path):
        if easygui.ynbox('Generate wind rose plot?'):
            print('Creating wind rose...')
            wr=WindRose(df)
            return df, wr
    
    elif easygui.ynbox('See met data stats in command window.\nDo you want to continue?'): 
        plt.close()
        return df, None
    
    else: sys.exit()
    
    
def cbScale(bounds):
    """
    Set up scalar-mappable (sm) for colorbar in plots 
    """
    # plot settings
    mpl.rcdefaults()            # reset to defaults
    styles=plt.style.available  # save all plot styles to  list
    plt.style.use(styles[14])   # set style
    
    #cmap='jet'
    #cmap='turbo'
    #cmap='gist_ncar'
    cmap='nipy_spectral'
    
    s=np.arange(bounds[0],bounds[1]+1,1)
    norm = mpl.colors.Normalize()
    norm.autoscale(s)
    sm = mpl.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    return sm    
    
def WindRose(met_QA):
    """
    Generate windrose from data for QA comparison
    """
    #generate a windrose for QA comparison
    data = met_QA[met_QA.WD!=999].copy() 
    
    #probs stuff
    WDu=data.WD.unique()
    WDu.sort()
    wd_band=11.25  # set deg +- for probability calcs
    
    
    data['prob']=''
    
    # XXX: setup to update command line in-place - doesnt work well in spyder
    # end arg sets cursor at begininning of line
    # text will be replaced 
    print("DUMMY TEXT", end="")
    
    # hack to only calc all unique pairs of WS & WD, instead of all hours (slow)
    for wd in WDu:
        ws_u=data[data.WD==wd].WS.unique()
        ws_u.sort()
        for ws in ws_u:
            #generate string to display
            disp='WD:{} WS:{}'.format(round(wd),round(ws))
            # pad with spaces out to 30 chars, keeps display line clean
            d=f"{disp:<30}"
            # clear previous line and write display string
            print("\r", end="")
            print(d, end="")
            p=Calc_prob_WS(ws,wd,data,wd_band)
            valid=data[(data.WS==ws) & (data.WD==wd)]
            data.loc[valid.index,'prob']=p
            
    # start on new line and stop updating in place
    print('\nWindrose complete...')
    
    ### test to limit to just unique wd/ws pairs
    data=data.drop_duplicates(subset=['WD','WS'])
    
    # sort to plot high probs last to show on 'top'
    data=data.sort_values(by=['prob'])
    
    # polar maths
    theta=data.WD*np.pi/180
    r=data.WS
    colors=data.prob
    sm=cbScale([min(colors),max(colors)])
    s=2

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rorigin(-1)

    ax.scatter(theta, r, c=colors, alpha=1, cmap=sm.cmap, s=s)
    cb=plt.colorbar(sm)
    title='% Probability that WS is exceeded at WD +-{} degrees'.format(round(wd_band))
    cb.ax.set_title(title,
                    fontsize=9,
                    rotation='vertical',
                    va='center',
                    ha='center',
                    y= .5,
                    x= 2.5,
                    )
    
    if max(r)>25: ax.set_rmax(25)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    return data 


#%% Startup & File selection
print('Use GUI to select met data file(s)...') 

metpaths,td=Get_Met()
plot_path=os.path.join(td,'QA_windrose.png')
met=Read_Met(metpaths)
met_QA,wr=MetQA(met)

print('Use GUI to select fit and crit files...') 
fitpath=Get_Fit(td)
critpath=Get_Crit(td)

print('Use GUI to enter a label for output file...')
label=easygui.enterbox('Enter a label for output file (if desired):')
if label is None: sys.exit()

#%% Load data from saved files if selected

# look for previous save data
pklpath=os.path.join(td,'PROBS.pkl')    #path of pickled (saved) dataframe

if os.path.isfile(pklpath):             #check if .pkl exists already   
    LoadSave=easygui.ynbox("Saved data found, do you want to append & update it?")
else: LoadSave=False

if LoadSave is None: sys.exit()
elif LoadSave:
    print('Loading saved data...')
    try:
        with open(pklpath, 'rb') as handle:
            old_dict = pickle.load(handle)
    except:
        print('Issue loading saved data...')
        cont=easygui.msgbox(msg="Issue loading save file.\nAll concentrations will be re-calculated, press ok to continue...")
        if cont==None: sys.exit()
        LoadSave=False
        
t0 = datetime.datetime.now()   
fit=Read_Fit(fitpath)
crit=Read_Crit(critpath)

fitname=os.path.basename(fitpath)
proj=fitname[:fitname.find('fit')]

#%% Append old hourly data if needed
if LoadSave:
    # unpack saved data
    fit_old=old_dict['fit']
    met_old=old_dict['met']
    hrly_old=old_dict['hourly']
    
    # if anything in met data has changed, re-run all fits
    if not all(met_old==met_QA):
        hrly=Hrly_Runs(fit,met_QA)
        
    else:
        # find new runs/fits
        new_fits=fit[~fit.FitID.isin(fit_old.FitID)].set_index('FitID')
        new_runs=new_fits.RunID.drop_duplicates()
        
        # find fits that exist in both old and new
        repeats=fit[fit.FitID.isin(fit_old.FitID)].set_index('FitID')
        check=fit_old[fit_old.FitID.isin(fit.FitID)].set_index('FitID')
        
        # find fits that have changed
        comp=repeats[repeats!=check].dropna(how='all')
        updated_fitIDs=comp.index
        updated_runs=repeats.loc[updated_fitIDs].RunID.drop_duplicates()
        
        # check for dropped fits in repeat runs
        repeat_runs=repeats.RunID.drop_duplicates()
        old_repeats=fit_old[fit_old.RunID.isin(repeat_runs)].set_index('FitID')
        good_IDs=repeats[~repeats.RunID.isin(updated_runs)].index
        dropped=old_repeats[~old_repeats.index.isin(good_IDs)]
        dropped_runs=dropped.RunID.drop_duplicates() 
        
        # Append list of updated run to include any runs where a fit was dropped
        updated_runs=updated_runs.append(dropped_runs)
        
        # find runs that have not changed
        good=repeats[~repeats.RunID.isin(updated_runs)]
        good_runs=good.RunID.drop_duplicates()
        
        # pull hourly data for runs that havent changed
        good_hrs=hrly_old[good_runs]
        
        # calc new and updated hrly data
        calc_runs=new_runs.append(updated_runs)
        calc_fits=fit[fit.RunID.isin(calc_runs)]
        new_hrly=Hrly_Runs(calc_fits,met_QA)
        
        # combine old and new hrly data
        hrly=good_hrs.copy()
        hrly[new_hrly.columns]=new_hrly.copy()
    
else:
    # if no save data found, run all fits
    hrly=Hrly_Runs(fit,met_QA)


#%% Calculate probs

results, op_hrs = All_Probs(fit,crit,hrly,met_QA)

#%% Save
savefolder=os.path.join(td,'Prob_Out')
if not os.path.exists(savefolder): os.makedirs(savefolder)

# PKL 
new_dict={}
new_dict['hourly']=hrly
new_dict['fit']=fit
new_dict['met']=met_QA

with open(pklpath, 'wb') as handle:
    pickle.dump(new_dict, handle)
print('PKL file Saved...')

# hourly Cm 
hrly_out=met_QA.copy()
hrly_out[hrly.columns]=hrly
save_hrly=os.path.join(savefolder,proj+'_hourly_'+label+'.csv')
hrly_out.to_csv(save_hrly, index=False)
print('Hourly Cm Saved...')

# probs
save_probs=os.path.join(savefolder,proj+'_probs_'+label+'.csv')
results.to_csv(save_probs)
print('Probs Saved...')

# operating hours
op_path=os.path.join(savefolder,proj+'_OpHours_'+label+'.txt')
with open(op_path, 'w') as text:
    for hr in op_hrs:
        text.write("{}\n".format(hr))


#%% done
t1=datetime.datetime.now()
print('Done!')
dt= t1-t0
dt=dt.seconds
easygui.msgbox(msg="Done!\nProcess took: {} seconds\nResults saved here: {}".format(dt,savefolder))  

sys.exit() #remove me when barplots are ready
#%% NEW Cell to test bar plotting
fig, ax = plt.subplots()

# =============================================================================
# import random
# vals = random.sample(range(1, 50), 24)
# =============================================================================

bins = op_hrs
n=len(hrly_out)
run='205A'
crit=500


vals=[]
for h in op_hrs:
    count=sum(hrly_out[hrly_out.H==h][run]>=crit)
    vals.append(count/n*100)

ax.bar(bins,vals)







