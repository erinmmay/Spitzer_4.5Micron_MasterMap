bpath='/Volumes/emay_spitzer/'
dpath='Data/Spitzer/Spitzer_Map_Ch2/S19.2.0/'
rpath='Analyses/bd067bo21/run/'
ch = 2

plt_savepath = '/Users/mayem1/Desktop/AOR_frames/'

apdir='ap4000715'
apsize=4.00

tpath=bpath

######################

import numpy as np
import pickle

import os, sys
import poet_run as run
from datetime import datetime

sys.path.insert(0,'/Users/mayem1/Documents/Code/POET/code/lib/models_c/py_func/')
sys.path.insert(0,'/Users/mayem1/Documents/Code/POET/code/lib/')

os.environ['OMP_NUM_THREADS']='4'
isinteractive = True 

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams

params = {'font.family' : 'serif'}
matplotlib.rcParams.update(params)

from m0_Iteration_Functions import *


###############
# Set up grid to bin on

# grid_step = 0.007
# npoints=4
grid_step = 0.01
npoints=2

ymin=14.4
ymax=16.0
xmin=14.4
xmax=16.0

ysize=int((ymax-ymin)/grid_step + 1)
xsize=int((xmax-xmin)/grid_step + 1)

ygrid = np.linspace(ymin,ymax,ysize)
xgrid = np.linspace(xmin,xmax,xsize)

xfull, yfull = np.meshgrid(ygrid,xgrid)
yfull = yfull.flatten()
xfull = xfull.flatten()

###############

AORs=[name.split('r')[1] for name in os.listdir(bpath+dpath) if os.path.isdir(bpath+dpath+name)]
print('Number of AORs: ', len(AORs))
print(AORs)

AORs_act = np.copy(AORs)

if ch == 2:
    dup_aors = np.array([45493248,45493504,45493760,45494016,45494272,45494528,45494784,45495040,45495296,45495552],dtype=int)

    for ai, a in enumerate(AORs_act):
        if int(a) in dup_aors:
            adel = np.where(AORs_act == a)[0]
            AORs_act = np.delete(AORs_act, adel)
            print('SKIP: ', ai, a, len(AORs_act))
            continue
        
###############
#### open full X, Y, F arrays
savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+'FullArrays_unscaled.npz','rb')
FullFs,FullXs,FullYs,FullFe,aornar,t0_arr,aorgrp = pickle.load(savefile_name)
savefile_name.close()

#### open index arrays for binning and interpolation
savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+'BinBilin_Inds_grouped.npz','rb')     
BinInds, BilinInds, BilinDist = pickle.load(savefile_name)
savefile_name.close()

#################
#### set up iterations
#AORs_todo = np.copy(AORs_act)
nloop = 0
rstrt = False
ndone = 0
rdiff = 0.9e-5  #10ppm precision in flux
maxloops = 100


if nloop>0:
    savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+'filesave_'+str(nloop).zfill(4)+'.npz','rb')
    rdiff_arr,trescfc = pickle.load(savefile_name)
    savefile_name.close()
else:
    rdiff_arr = np.ones(np.nanmax(aorgrp)+1)*100.0  #holds rescales from last loop
    trescfc = np.ones_like(AORs_act.astype(float))*1.0      #total rescales (multiplicative)
    
t_group = np.ones(np.nanmax(aorgrp)+1)*1.

FullFs /= np.nanmean(FullFs)   #initial scaling so median of map is 1 -> makes things easier later

os.chdir(bpath+rpath)


if rstrt == True:
    #rescaling FullFs based on current loop restart
    for ai in range(0,len(AORs_act)):
        AOR_fih = np.append(AOR_fih, np.where(aornar == int(AORs_act[ai]))[0])
        FullFs[AOR_fih] *= trsecfc[ai]
        FullFe[AOR_fih] *= trsecfc[ai]
    # for ni in range(0,np.nanmax(aorgrp)+1):
    #     group_AORs = np.where(aorgrp == ni) [0]
    #
    #     AOR_fih = np.array([],dtype = int)  #current group of AORs indexes
    #     for gai in group_AORs:
    #         AOR_fih = np.append(AOR_fih, np.where(aornar == int(AORs_act[gai]))[0])
    #         #print(len(AOR_fih))
    #
    #     FullFs[AOR_fih] *= trescfc[ni]    

#AORs_act=AORs_act[:50]
while np.round(np.sqrt(np.mean((rdiff_arr-1.0)**2.)),5) > rdiff and nloop <= maxloops: 
    total_aors = 0
    AORs_todo = np.copy(AORs_act)
    rescfc = np.ones(np.nanmax(aorgrp)+1)*1.            #gets updated in current loop
    print('#######################')
    print('# RUNNING LOOP: ', nloop)
    print('# Current rdiff:', np.round(np.sqrt(np.mean((rdiff_arr-1.0)**2.)),5))
    print('# rdiff limit:  ', rdiff)
    print('# med FullFs:   ', np.nanmean(FullFs))
    print('#######################')
    for ni in range(0,np.nanmax(aorgrp)+1):
    
        group_AORs = np.where(aorgrp == ni) [0]
        total_aors += len(group_AORs)

        time0=datetime.now()
    
        print('--->',ni, total_aors, len(group_AORs))
        
        if nloop == 0:
            plot = False
        else:
            plot = False
     
        #get indexes for this AOR
        AOR_fih = np.array([],dtype = int)  #current group of AORs indexes
        for gai in group_AORs:
            hold_inds = np.where(aornar == int(AORs_act[gai]))[0]
            AOR_fih = np.append(AOR_fih, hold_inds)
        AOR_fib = np.ones_like(aornar, dtype = bool)    #boolean array for rest of map
        AOR_fib[AOR_fih] = False
        
        if nloop == 0 or rstrt == True:
            # if len(group_AORs)>1:
            #     print(t0_arr[group_AORs])
            #     print(np.diff(np.sort(t0_arr[group_AORs]))*24.)
            firstt = np.nanargmin(t0_arr[group_AORs])
            t_group[ni] = t0_arr[group_AORs][firstt]
    
        mm_flux = FullFs[AOR_fib]
        ca_flux = FullFs[AOR_fih]
        # print(len(mm_flux),len(ca_flux), len(mm_flux)+len(ca_flux))
        #### BINNING #####
        time1 = datetime.now()
        bin_flux_hold = np.empty([ysize*xsize])*np.nan
        for i in range(xsize*ysize):
            bin_flux_hold[i] = np.mean(mm_flux[BinInds[ni][i]])
        
        #### INTERPOLATING #####
        time2 = datetime.now()
        interp_flux = np.empty([len(ca_flux)])*np.nan
    
        for i in range(0,len(ca_flux)):
            f11 = bin_flux_hold[BilinInds[ni][i][0]]
            f21 = bin_flux_hold[BilinInds[ni][i][1]]
            f22 = bin_flux_hold[BilinInds[ni][i][2]]
            f12 = bin_flux_hold[BilinInds[ni][i][3]]
        
            if any(np.isnan([f11,f21,f22,f12])):
                interp_flux[i] = np.nan
                continue
            
            d1 = BilinDist[ni][i][0]
            d2 = BilinDist[ni][i][1]
            d3 = BilinDist[ni][i][2]
            d4 = BilinDist[ni][i][3]
        
            interp_flux[i] = (f11 * d1) + (f21 * d2) + (f22 * d3) + (f12 * d4)
        time3 = datetime.now()
    
        #### CALCULATING RESCALE ####
        rescale_fac,  where_full, where_fnew = rescale_easy(interp_flux, ca_flux)
        rescale_fac2, where_full, where_fnew = rescale_easy(interp_flux, ca_flux*rescale_fac)
    
    
        if np.isfinite(rescale_fac):
            if plot == True:
                overlap_plot(ni, a, ndone, FullXs[AOR_fib], FullYs[AOR_fib], FullXs[AOR_fih], FullYs[AOR_fih], xfull, yfull, FullXs, FullYs, where_full, where_fnew, plt_savepath)
        
            for gai in group_AORs:
                adel = np.where(AORs_todo == (AORs_act[gai]))[0]
                AORs_todo = np.delete(AORs_todo, adel)
        
            ndone += len(group_AORs)
            print('       ',ndone, '/', (1+nloop)*len(AORs_act), len(AORs_todo), np.round(rescale_fac,5), np.round(rescale_fac2,5),'time to bin:', time2-time1,'time to interp:', time3-time2)
        
        
            FullFs[AOR_fih]      = np.copy(rescale_fac*ca_flux)
            FullFe[AOR_fih]      = np.copy(rescale_fac*FullFe[AOR_fih])
            rescfc[ni]           = rescale_fac
            trescfc[group_AORs]  = np.copy(rescale_fac*trescfc[group_AORs])
    
        del bin_flux_hold
        del interp_flux
        del where_full
        del where_fnew
        
    rdiff_arr = np.copy(rescfc)

    # renormalize the map! 
    rdiff_arr /= np.nanmean(FullFs)
    trescfc /= np.nanmean(FullFs)
    FullFe /= np.nanmean(FullFe)
    FullFs /= np.nanmean(FullFs)
    
    if nloop ==0 or rstrt == True:
        if ch == 1:
            timecut0 = 2455930.0
            timecut1 = 2456500.0
            timecut2 = 2456500.0
            timecut3 = 2457250.0
        if ch == 2:
            timecut0 = 2455930.0
            timecut1 = 2456500.0
            timecut2 = 2456500.0
            timecut3 = 2457250.0

        indexes0 = np.where(t_group <  timecut0)[0]
        indexes1 = np.where((t_group >  timecut0) & (t_group <  timecut1)) [0]
        indexes2 = np.where((t_group >  timecut1) & (t_group <  timecut2)) [0]
        indexes3 = np.where((t_group >  timecut2) & (t_group <  timecut3)) [0]
        indexes4 = np.where(t_group >  timecut3)[0]
        
        tindexes0 = np.where(t0_arr <  timecut0)[0]
        tindexes1 = np.where((t0_arr >  timecut0) & (t0_arr <  timecut1)) [0]
        tindexes2 = np.where((t0_arr >  timecut1) & (t0_arr <  timecut2)) [0]
        tindexes3 = np.where((t0_arr >  timecut2) & (t0_arr <  timecut3)) [0]
        tindexes4 = np.where(t0_arr >  timecut3)[0]

    rescfc0 = np.nanmedian(rdiff_arr[indexes0])
    rescfc1 = np.nanmedian(rdiff_arr[indexes1])
    rescfc2 = np.nanmedian(rdiff_arr[indexes2])
    rescfc3 = np.nanmedian(rdiff_arr[indexes3])
    rescfc4 = np.nanmedian(rdiff_arr[indexes4])

    trescfc0 = np.nanmedian(trescfc[tindexes0])
    trescfc1 = np.nanmedian(trescfc[tindexes1])
    trescfc2 = np.nanmedian(trescfc[tindexes2])
    trescfc3 = np.nanmedian(trescfc[tindexes3])
    trescfc4 = np.nanmedian(trescfc[tindexes4])

    plt.figure(int(ndone+1),figsize=(14,8))

    plt.hlines(y=rescfc0,xmin=np.nanmin(t0_arr), xmax=timecut0,color='black',lw=5.0)
    plt.hlines(y=rescfc1,xmin=timecut0, xmax=timecut1,color='black',lw=5.0)
    plt.hlines(y=rescfc2,xmin=timecut1, xmax=timecut2,color='black',lw=5.0)
    plt.hlines(y=rescfc3,xmin=timecut2, xmax=timecut3,color='black',lw=5.0)
    plt.hlines(y=rescfc4,xmin=timecut3, xmax=np.nanmax(t0_arr),color='black',lw=5.0)

    tc='lightsteelblue'
    tlw=3
    plt.hlines(y=trescfc0,xmin=np.nanmin(t0_arr), xmax=timecut0,color=tc,lw=tlw)
    plt.hlines(y=trescfc1,xmin=timecut0, xmax=timecut1,color=tc,lw=tlw)
    plt.hlines(y=trescfc2,xmin=timecut1, xmax=timecut2,color=tc,lw=tlw)
    plt.hlines(y=trescfc3,xmin=timecut2, xmax=timecut3,color=tc,lw=tlw)
    plt.hlines(y=trescfc4,xmin=timecut3, xmax=np.nanmax(t0_arr),color=tc,lw=tlw)

    plt.axhline(y=1.0,color='black', lw=2.0, ls='--', alpha=0.5)

    plt.plot(t0_arr, trescfc, '.', ms=15,color='mediumslateblue',alpha=0.9)

    plt.plot(t_group,rdiff_arr,'.',ms=15,mec='tomato',mfc='none',mew=2.0)


    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Caclulated/Cumulative Rescale Factor',fontsize=20)
    plt.title(np.round(np.sqrt(np.mean((rdiff_arr-1.0)**2.)),6),fontsize=20)
    plt.figtext(0.8,0.8,str(nloop),fontsize=35)
    plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes0]-trescfc0)**2.)),6),xy=(np.mean(t0_arr[tindexes0]),1.003),fontsize=15,ha='center',rotation='vertical')
    plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes1]-trescfc1)**2.)),6),xy=(np.mean(t0_arr[tindexes1]),1.003),fontsize=15,ha='center',rotation='vertical')
    plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes2]-trescfc2)**2.)),6),xy=(np.mean(t0_arr[tindexes2]),1.003),fontsize=15,ha='center',rotation='vertical')
    plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes3]-trescfc3)**2.)),6),xy=(np.mean(t0_arr[tindexes3]),1.003),fontsize=15,ha='center',rotation='vertical')
    plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes4]-trescfc4)**2.)),6),xy=(np.mean(t0_arr[tindexes4]),1.003),fontsize=15,ha='center',rotation='vertical')

    plt.ylim(0.996,1.004)

    plt.savefig(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+str(nloop).zfill(4)+'.png')
    #plt.show()
    plt.close(int(ndone+1))
    
    savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+'filesave_'+str(nloop).zfill(4)+'.npz','wb')
    pickle.dump([rdiff_arr,trescfc],savefile_name)
    savefile_name.close()
    
    nloop += 1
 
print('#######################')
print('# End LOOP: ', nloop-1)
print('# Current rdiff:', np.round(np.sqrt(np.mean((rdiff_arr-1.0)**2.)),5))
print('# rdiff limit:  ', rdiff)
print('# med FullFs:   ', np.nanmedian(FullFs))
print('#######################')  
      
                
