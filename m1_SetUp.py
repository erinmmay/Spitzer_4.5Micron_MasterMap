bpath='/Volumes/emay_spitzer/'
dpath='Data/Spitzer/Spitzer_Map_Ch2/S19.2.0/'
rpath='Analyses/bd067bo21/run/'
ch = 2

apdir='ap2000715'
apsize=2.00

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

# open all AORs and create arrays of X,Y,F 

FullXs = np.array([])
FullYs = np.array([])
FullFs = np.array([])
FullFe = np.array([])
aornar = np.array([])

groupN = 0

t0_arr = np.ones_like(AORs_act.astype(float))*np.nan
aorgrp = np.ones_like(AORs_act.astype(int))*groupN

print(' ')
print('########################')
print('# Creating full lists  #')
print('########################')
os.chdir(bpath+rpath)
for ai, a in enumerate(AORs_act):
    # if ai>50:
    #     continue
    ind_act = np.where(AORs_act == a)[0]
    print('--->', ai, a, '----')
    if not os.path.exists(a+'/'):
        print('not ran')
        continue

    tpath=bpath+rpath+str(a)+'/fgc/'+apdir+'/'
    xV, yV, fV, fe, ts = open_file(tpath, ch)
    
    FullXs = np.append(FullXs,xV)
    FullYs = np.append(FullYs,yV)
    FullFs = np.append(FullFs,fV)
    FullFe = np.append(FullFe,fe)
    aornar = np.append(aornar,np.ones_like(fV,dtype=int)*int(a))
    
    # if ai == 0:
    #     aorgrp[ai] = groupN
    # else:
    #     closest = np.nanargmin(np.abs(t0_arr-ts))  #closesest in time. units of days
    #     print('   ', 1.0/24.0, ts, t0_arr[closest], t0_arr[closest] - ts, groupN)
    #     if np.abs(t0_arr[closest] - ts) <= (1.0/24.0): #1 hour in units of days
    #           aorgrp[ai] = aorgrp[closest] # asign same group number as closest if first frame within an hour (either direction)
    #     else:
    #         groupN += 1
    #         aorgrp[ai] = groupN
            
    t0_arr[ai] = ts
    

time_sort = np.argsort(t0_arr)
aors_sort = AORs_act[time_sort]

for ai, a in enumerate(aors_sort):
    ind_act = np.where(AORs_act == a)[0]
    
    tsc = t0_arr[time_sort][ai]
    if ai == 0:
        aorgrp[ind_act] = groupN
    else:
        tsb = t0_arr[time_sort][ai-1]
        if (tsc - tsb) <= (1.0/24.0): # if current aor started <= 1 hr after previous
            aorgrp[ind_act] = groupN
        else:
            groupN += 1
            aorgrp[ind_act] = groupN
    

# print(aorgrp)
# print(len(FullFs))
savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+'FullArrays_unscaled.npz','wb')      
pickle.dump([FullFs,FullXs,FullYs,FullFe,aornar,t0_arr,aorgrp],savefile_name)
savefile_name.close()

#############

#list of arrays to save indexes for each bin for each mastermap minus a given AOR
BinInds = []
BilinInds = []
BilinDist = []


# ### FOR INDIVIDUAL AORS
# print(' ')
# print('########################################################')
# print('# Creating lists of binning and interpolation indexes  #')
# print('########################################################')
# os.chdir(bpath+rpath)
# ningroup = 0
# ngroup = 0
# for ai, a in enumerate(AORs_act):
#     # if ai>50:
#     #     continue
#     ind_act = np.where(AORs_act == a)[0]
#     print('--->', ai, a, '----')
#     if not os.path.exists(a+'/'):
#         print('not ran')
#         continue
#
#     #get indexes for this AOR
#     AOR_fih = np.where(aornar == int(a))[0]         #current AOR indexes
#     AOR_fib = np.ones_like(aornar, dtype = bool)    #boolean array for rest of map
#     AOR_fib[AOR_fih] = False
#
#     time1=datetime.now()
#     ca_binds = POET_bin_inds(FullXs[AOR_fib],FullYs[AOR_fib],xgrid,ygrid,npoints)
#     time2=datetime.now()
#     ca_bilininds, ca_bilindist = BiLinInterp_inds(xfull,yfull,FullXs[AOR_fih],FullYs[AOR_fih])
#     time3=datetime.now()
#
#     BinInds.append(ca_binds)
#     BilinInds.append(ca_bilininds)
#     BilinDist.append(ca_bilindist)
#
#     print('     ', len(ca_binds),len(ca_bilininds),len(ca_bilindist),len(FullXs[AOR_fib]),len(FullXs[AOR_fih]))
#     print('      time to bin: ', time2-time1,'      time to interp: ', time3-time2)
 
##### FOR GROUPED AORS ####
print(' ')
print('########################################################')
print('# Creating lists of binning and interpolation indexes  #')
print('########################################################')
os.chdir(bpath+rpath)
total_aors = 0
for ni in range(0,np.nanmax(aorgrp)+1):
    group_AORs = np.where(aorgrp == ni) [0]
    total_aors += len(group_AORs)

    print('--->',ni, total_aors, group_AORs,AORs_act[group_AORs])
    if not os.path.exists(a+'/'):
        print('not ran')
        continue

    AOR_fih = np.array([],dtype = int)
    for gai in group_AORs:
        hold_inds = np.where(aornar == int(AORs_act[gai]))[0]
        AOR_fih = np.append(AOR_fih, hold_inds)

    AOR_fib = np.ones_like(aornar, dtype = bool)    #boolean array for rest of map
    AOR_fib[AOR_fih] = False

    time1=datetime.now()
    ca_binds = POET_bin_inds(FullXs[AOR_fib],FullYs[AOR_fib],xgrid,ygrid,npoints)
    time2=datetime.now()
    ca_bilininds, ca_bilindist = BiLinInterp_inds(xfull,yfull,FullXs[AOR_fih],FullYs[AOR_fih])
    time3=datetime.now()

    BinInds.append(ca_binds)
    BilinInds.append(ca_bilininds)
    BilinDist.append(ca_bilindist)

    print('     ', len(ca_binds),len(ca_bilininds),len(ca_bilindist),len(FullXs[AOR_fib]),len(FullXs[AOR_fih]))
    print('      time to bin: ', time2-time1,'      time to interp: ', time3-time2)


savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+'BinBilin_Inds_grouped.npz','wb')
pickle.dump([BinInds, BilinInds, BilinDist],savefile_name)
savefile_name.close()


