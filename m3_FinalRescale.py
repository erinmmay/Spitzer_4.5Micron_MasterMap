bpath='/Volumes/emay_spitzer/'
dpath='Data/Spitzer/Spitzer_Map_Ch2/S19.2.0/'
rpath='Analyses/bd067bo21/run_ap2000715/'
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

###############

AORs=[name.split('r')[1] for name in os.listdir(bpath+dpath) if os.path.isdir(bpath+dpath+name)]
print('Number of AORs: ', len(AORs))
print(AORs)
#
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

###############
#### open rescale files
nloop = 14
do_rescale = 2              #1 = time averaged, 2 = group averaged

savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+'filesave_'+str(nloop).zfill(4)+'.npz','rb')
rdiff_arr,trescfc = pickle.load(savefile_name)
savefile_name.close()

# print(trescfc.shape, AORs_act.shape)
# for ni in range(0,max(aorgrp)+1):
#     hold_inds = np.where(aorgrp == ni)[0]
#     print(ni, hold_inds, AORs_act[hold_inds], trescfc[hold_inds])
frescfc = np.ones_like(trescfc)

################
### divide up

timecut0 = 2455930.0
timecut1 = 2456500.0#2456154.0
timecut2 = 2456500.0
timecut3 = 2457250.0

tindexes0 = np.where(t0_arr <  timecut0)[0]
tindexes1 = np.where((t0_arr >  timecut0) & (t0_arr <  timecut1)) [0]
tindexes2 = np.where((t0_arr >  timecut1) & (t0_arr <  timecut2)) [0]
tindexes3 = np.where((t0_arr >  timecut2) & (t0_arr <  timecut3)) [0]
tindexes4 = np.where(t0_arr >  timecut3)[0]

trescfc0 = np.nanmedian(trescfc[tindexes0])
trescfc1 = np.nanmedian(trescfc[tindexes1])
trescfc2 = np.nanmedian(trescfc[tindexes2])
trescfc3 = np.nanmedian(trescfc[tindexes3])
trescfc4 = np.nanmedian(trescfc[tindexes4])

frescfc[tindexes0] = trescfc0
frescfc[tindexes1] = trescfc1
frescfc[tindexes2] = trescfc2
frescfc[tindexes3] = trescfc3
frescfc[tindexes4] = trescfc4

###########
## Plot it up
plt.figure(1,figsize=(14,8))

tc='lightsteelblue'
tlw=3
plt.hlines(y=trescfc0,xmin=np.nanmin(t0_arr), xmax=timecut0,color=tc,lw=tlw)
plt.hlines(y=trescfc1,xmin=timecut0, xmax=timecut1,color=tc,lw=tlw)
plt.hlines(y=trescfc2,xmin=timecut1, xmax=timecut2,color=tc,lw=tlw)
plt.hlines(y=trescfc3,xmin=timecut2, xmax=timecut3,color=tc,lw=tlw)
plt.hlines(y=trescfc4,xmin=timecut3, xmax=np.nanmax(t0_arr),color=tc,lw=tlw)

plt.axhline(y=1.0,color='black', lw=2.0, ls='--', alpha=0.5)

plt.plot(t0_arr, trescfc, '.', ms=15,color='mediumslateblue',alpha=0.9)
plt.plot(t0_arr, frescfc,'.',ms=15,mec='tomato',mfc='none',mew=2.0)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel('Time',fontsize=20)
plt.ylabel('Caclulated/Cumulative Rescale Factor',fontsize=20)
plt.figtext(0.7,0.2,'FINAL',fontsize=35)

plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes0]-trescfc0)**2.)),6),xy=(np.mean(t0_arr[tindexes0]),1.003),fontsize=15,ha='center',rotation='vertical')
plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes1]-trescfc1)**2.)),6),xy=(np.mean(t0_arr[tindexes1]),1.003),fontsize=15,ha='center',rotation='vertical')
plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes2]-trescfc2)**2.)),6),xy=(np.mean(t0_arr[tindexes2]),1.003),fontsize=15,ha='center',rotation='vertical')
plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes3]-trescfc3)**2.)),6),xy=(np.mean(t0_arr[tindexes3]),1.003),fontsize=15,ha='center',rotation='vertical')
plt.annotate(np.round(np.sqrt(np.mean((trescfc[tindexes4]-trescfc4)**2.)),6),xy=(np.mean(t0_arr[tindexes4]),1.003),fontsize=15,ha='center',rotation='vertical')

plt.ylim(0.996,1.004)

plt.savefig(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'Final_Rescale.png')
#plt.show()
plt.close(1)

##############
## Do the rescale and save the file
for ai,a in enumerate(AORs_act):
    #get indexes for this AOR
    AOR_fih = np.where(aornar == int(a))[0]         #current AOR indexes
    if do_rescale==1:
        FullFs[AOR_fih] = np.copy(FullFs[AOR_fih]*frescfc[ai])
        FullFe[AOR_fih] = np.copy(FullFe[AOR_fih]*frescfc[ai])
        savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_final_rescale_avgtime.npz','wb')
    if do_rescale==2:
        FullFs[AOR_fih] = np.copy(FullFs[AOR_fih]*trescfc[ai])
        FullFe[AOR_fih] = np.copy(FullFe[AOR_fih]*trescfc[ai])
        savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_final_rescale_avggrp.npz','wb')

pickle.dump([FullFs,FullXs, FullYs, FullFe],savefile_name)
savefile_name.close()




