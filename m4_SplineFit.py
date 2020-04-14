bpath='/Volumes/emay_spitzer/'
dpath='Data/Spitzer/Spitzer_Map_Ch2/S19.2.0/'
rpath='Analyses/TY4212o11/run/'
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

import scipy
from scipy import interpolate

from m0_Iteration_Functions import *

###############

# AORs=[name.split('r')[1] for name in os.listdir(bpath+dpath) if os.path.isdir(bpath+dpath+name)]
# print('Number of AORs: ', len(AORs))
# print(AORs)
#
# dup_aors = np.array([45493248,45493504,45493760,45494016,45494272,45494528,45494784,45495040,45495296,45495552],dtype=int)
#
# AORs_act = np.copy(AORs)
# for ai, a in enumerate(AORs_act):
#     if int(a) in dup_aors:
#         adel = np.where(AORs_act == a)[0]
#         AORs_act = np.delete(AORs_act, adel)
#         print('SKIP: ', ai, a, len(AORs_act))
#         continue
        
###############
#### open rescaled X, Y, F arrays
do_rescale = 1 

if do_rescale==1:
    savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_final_rescale_avgtime.npz','rb')
if do_rescale==2:
    savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_final_rescale_avggrp.npz','rb')

FullFs,FullXs,FullYs, FullFe = pickle.load(savefile_name)
savefile_name.close()

###############
# Set up grid to calc spline fit on

grid_step = 0.001
npoints=1

minval = np.nanmin([np.nanmin(FullXs),np.nanmin(FullYs)])
maxval = np.nanmax([np.nanmax(FullXs),np.nanmax(FullYs)])

minval = np.round(minval,2) - 0.01
maxval = np.round(maxval,2) + 0.01

size = int((maxval-minval)/grid_step + 1)
print(size)

ygrid = np.linspace(minval,maxval,size)
xgrid = np.linspace(minval,maxval,size)

xfull, yfull = np.meshgrid(ygrid,xgrid)
yfull = yfull.flatten()
xfull = xfull.flatten()
print(len(xfull))

##################
# first we gotta bin it down because there are too many points!

print('... binning onto defined grid')
BFullFs, BFullFe, numpts = POET_bin(FullXs, FullYs, FullFs, FullFe, xgrid,ygrid,npoints)

##################
#### Do Spline Fit

# knot grid 
knot_steps = 0.020
knot_size = int((maxval-minval)/knot_steps + 1)

knot_grid = np.linspace(minval,maxval,knot_size)

# remove nan - Fluxes
nan_inds = np.where(np.isfinite(BFullFs))[0]
print(len(nan_inds))

############
### interpolate spline fit onto OG grid -> including between gaps, but first remove isolated points from the calc
gap = 1.0*knot_steps/grid_step
frac=0.1

# remove nan - Fluxes
BFullFs_rm = np.copy(BFullFs)
# print('... remove isolated points (3 sided)')
# skip_inds3a = remove_indexes(BFullFs_rm, ygrid, xgrid, FullYs, FullXs, size, 3)
# BFullFs_rm[skip_inds3a] = np.nan
# print('... remove isolated points (4 sided)')
# skip_inds4a = remove_indexes(BFullFs_rm, ygrid, xgrid, FullYs, FullXs, size, 4)
# BFullFs_rm[skip_inds4a] = np.nan

nan_inds = np.where(np.isfinite(BFullFs_rm))[0]
print(len(nan_inds))

print('... running spline fit')
time1 = datetime.now()
SplineFit = interpolate.LSQBivariateSpline(xfull[nan_inds], yfull[nan_inds], BFullFs_rm[nan_inds], knot_grid, knot_grid, kx=3, ky=3 )
time2 = datetime.now()
print('     spline time: ', time2 - time1)



####      
## fill in gaps
#round 1
print('~~ finding gaps, round 1')
interp_addi = interp_indexes(BFullFs_rm, ygrid, xgrid, FullYs, FullXs, size, gap,frac)
interp_inds = np.sort(np.append(nan_inds, interp_addi))

TempGrid = SplineFit.ev(xfull[interp_inds],yfull[interp_inds])
TempFull = np.empty([size*size])*np.nan
TempFull[interp_inds] = TempGrid
print('... remove isolated points (3 sided)')
skip_inds3b = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 3)
TempFull[skip_inds3b] = np.nan
print('... remove isolated points (4 sided)')
skip_inds4b = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 4)
TempFull[skip_inds4b] = np.nan

interp_inds_rm = np.where(~np.isnan(TempFull))[0]

#round 2
print('~~ finding gaps, round 2')
TempGrid = SplineFit.ev(xfull[interp_inds_rm],yfull[interp_inds_rm])
TempFull = np.empty([size*size])*np.nan
TempFull[interp_inds_rm] = TempGrid
interp_addi2 = interp_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, gap,frac)
interp_inds = np.sort(np.append(interp_inds, interp_addi2))

TempGrid = SplineFit.ev(xfull[interp_inds],yfull[interp_inds])
TempFull = np.empty([size*size])*np.nan
TempFull[interp_inds] = TempGrid
print('... remove isolated points (3 sided)')
skip_inds3b = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 3)
TempFull[skip_inds3b] = np.nan
print('... remove isolated points (4 sided)')
skip_inds4b = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 4)
TempFull[skip_inds4b] = np.nan

interp_inds_rm = np.where(~np.isnan(TempFull))[0]

#round 3
print('~~ finding gaps, round 3')
TempGrid = SplineFit.ev(xfull[interp_inds_rm],yfull[interp_inds_rm])
TempFull = np.empty([size*size])*np.nan
TempFull[interp_inds_rm] = TempGrid
interp_addi3 = interp_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, gap,frac)
interp_inds = np.sort(np.append(interp_inds, interp_addi3))

TempGrid = SplineFit.ev(xfull[interp_inds],yfull[interp_inds])
TempFull = np.empty([size*size])*np.nan
TempFull[interp_inds] = TempGrid
print('... remove isolated points (3 sided)')
skip_inds3b = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 3)
TempFull[skip_inds3b] = np.nan
print('... remove isolated points (4 sided)')
skip_inds4b = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 4)
TempFull[skip_inds4b] = np.nan

interp_inds_rm = np.where(~np.isnan(TempFull))[0]

#round 4
print('~~ finding gaps, round 4')
TempGrid = SplineFit.ev(xfull[interp_inds_rm],yfull[interp_inds_rm])
TempFull = np.empty([size*size])*np.nan
TempFull[interp_inds_rm] = TempGrid
interp_addi4 = interp_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, gap,frac)
interp_inds = np.sort(np.append(interp_inds, interp_addi4))

TempGrid = SplineFit.ev(xfull[interp_inds],yfull[interp_inds])
TempFull = np.empty([size*size])*np.nan
TempFull[interp_inds] = TempGrid

print('... remove edges (1 sided)')
skip_indsE1 = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 1)
TempFull[skip_indsE1] = np.nan
print('... remove edges (1 sided)')
skip_indsE2 = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 1)
TempFull[skip_indsE2] = np.nan
print('... remove edges (1 sided)')
skip_indsE2 = remove_indexes(TempFull, ygrid, xgrid, FullYs, FullXs, size, 1)
TempFull[skip_indsE2] = np.nan

interp_inds_rm = np.where(~np.isnan(TempFull))[0]

##########

# do the interpolation
OutGrid = SplineFit.ev(xfull[interp_inds_rm],yfull[interp_inds_rm]) 
OutFull = np.empty([size*size])*np.nan
OutFull[interp_inds_rm] = OutGrid       

OutFullP = OutFull.reshape(size,size)

# calculate reduced chi sq
redchisq = np.nansum(((OutFull[nan_inds] - BFullFs[nan_inds])**2.)/(BFullFe[nan_inds]**2.))/(len(nan_inds)-knot_size**2.)
print(redchisq)

##########

if do_rescale==1:
    saveplot_name = 'Smoothed_AvgTime_master_map_G'+str(int(grid_step*10**4.))+'_P'+str(int(npoints))+'_K'+str(int(knot_steps*10**4.))
if do_rescale==2:
    saveplot_name = 'Smoothed_AvgGrp_master_map_G'+str(int(grid_step*10**4.))+'_P'+str(int(npoints))+'_K'+str(int(knot_steps*10**4.))


fig=plt.figure(1,figsize=(5.,4.))
plt.clf()
plt.cla()
plt.subplots_adjust(bottom=0.05,left=0.06,top=0.99)

diff = (OutFullP-BFullFs.reshape(size,size))/BFullFs.reshape(size,size)
if np.nanmax(diff) >1:
    vmin = -0.1
    vmax = 0.1
else:
    vmin = -1.0*np.nanmax(diff)
    vmax = np.nanmax(diff)

p1=plt.imshow(diff, extent = (min(xfull), max(xfull), max(yfull), min(yfull)), aspect = 'equal', cmap = 'BrBG', vmin = vmin, vmax = vmax)  

plt.vlines(x=14.85, ymin=14.85,ymax=15.35,lw=4.0)
plt.vlines(x=15.35, ymin=14.85,ymax=15.35,lw=4.0)
plt.hlines(y=14.85, xmin=14.85,xmax=15.35,lw=4.0)
plt.hlines(y=15.35, xmin=14.85,xmax=15.35,lw=4.0)

plt.ylim(14.4,15.7)
plt.xlim(14.4,15.7)

c=plt.colorbar(p1)

plt.figtext(0.1,0.85, '$\chi^2_{red}$ = '+str(np.round(redchisq,3)), fontsize=15, color='black', weight='bold')
plt.savefig(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+saveplot_name+'_Plot1.png')

#########

fig=plt.figure(2,figsize=(5.,4.))
plt.clf()
plt.cla()
plt.subplots_adjust(bottom=0.05,left=0.06,top=0.99)

p2=plt.imshow(OutFullP/np.nanmax(BFullFs), extent = (min(xfull), max(xfull), max(yfull), min(yfull)), aspect = 'equal', cmap = 'terrain', vmin = np.nanmin(BFullFs)/np.nanmax(BFullFs), vmax = 1.0)  
# plt.scatter(xfull[interp_addi], yfull[interp_addi], s= 4, color='red')

plt.vlines(x=14.85, ymin=14.85,ymax=15.35,lw=4.0)
plt.vlines(x=15.35, ymin=14.85,ymax=15.35,lw=4.0)
plt.hlines(y=14.85, xmin=14.85,xmax=15.35,lw=4.0)
plt.hlines(y=15.35, xmin=14.85,xmax=15.35,lw=4.0)

plt.ylim(14.4,15.7)
plt.xlim(14.4,15.7)

plt.axvline(x=15.0)
plt.axhline(y=15.0)

c=plt.colorbar(p2)

plt.figtext(0.1,0.85, '$\chi^2_{red}$ = '+str(np.round(redchisq,3)), fontsize=15, color='black', weight='bold')
plt.savefig(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+saveplot_name+'_Plot2.png')


#########

fig=plt.figure(5,figsize=(5.,4.))
plt.clf()
plt.cla()
plt.subplots_adjust(bottom=0.05,left=0.06,top=0.99)

p2=plt.imshow(TempFull.reshape(size,size)/np.nanmax(BFullFs), extent = (min(xfull), max(xfull), max(yfull), min(yfull)), aspect = 'equal', cmap = 'terrain', vmin = np.nanmin(BFullFs)/np.nanmax(BFullFs), vmax = 1.0)  
plt.scatter(xfull[interp_addi], yfull[interp_addi], s= 3, color='tomato')
plt.scatter(xfull[interp_addi2], yfull[interp_addi2], s= 2, color='green')
plt.scatter(xfull[interp_addi3], yfull[interp_addi3], s= 2, color='blue')
plt.scatter(xfull[interp_addi4], yfull[interp_addi4], s= 2, color='orange')

plt.vlines(x=14.85, ymin=14.85,ymax=15.35,lw=4.0)
plt.vlines(x=15.35, ymin=14.85,ymax=15.35,lw=4.0)
plt.hlines(y=14.85, xmin=14.85,xmax=15.35,lw=4.0)
plt.hlines(y=15.35, xmin=14.85,xmax=15.35,lw=4.0)

plt.ylim(14.4,15.7)
plt.xlim(14.4,15.7)

c=plt.colorbar(p2)

plt.figtext(0.1,0.85, '$\chi^2_{red}$ = '+str(np.round(redchisq,3)), fontsize=15, color='black', weight='bold')
plt.savefig(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+saveplot_name+'_Plot5.png')

#########

fig=plt.figure(6,figsize=(5.,4.))
plt.clf()
plt.cla()
plt.subplots_adjust(bottom=0.05,left=0.06,top=0.99)

p2=plt.imshow(BFullFs.reshape(size,size)/np.nanmax(BFullFs), extent = (min(xfull), max(xfull), max(yfull), min(yfull)), aspect = 'equal', cmap = 'terrain', vmin = np.nanmin(BFullFs)/np.nanmax(BFullFs), vmax = 1.0)  
# plt.scatter(xfull[skip_inds3a], yfull[skip_inds3a], s= 3, color='red')
# plt.scatter(xfull[skip_inds4a], yfull[skip_inds4a], s= 3, color='blue')

plt.scatter(xfull[skip_inds3b], yfull[skip_inds3b], s= 3, color='orange')
plt.scatter(xfull[skip_inds4b], yfull[skip_inds4b], s= 3, color='green')

plt.vlines(x=14.85, ymin=14.85,ymax=15.35,lw=4.0)
plt.vlines(x=15.35, ymin=14.85,ymax=15.35,lw=4.0)
plt.hlines(y=14.85, xmin=14.85,xmax=15.35,lw=4.0)
plt.hlines(y=15.35, xmin=14.85,xmax=15.35,lw=4.0)

plt.ylim(14.4,15.7)
plt.xlim(14.4,15.7)

c=plt.colorbar(p2)

plt.figtext(0.1,0.85, 'Removed Points', fontsize=15, color='black', weight='bold')
plt.savefig(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+saveplot_name+'_Plot6.png')

#######
val=15.0
ind= np.argmin(np.abs(ygrid-val))
print(ygrid[ind])

fig = plt.figure(3, figsize=(5.,4.))
plt.clf()
plt.cla()
plt.subplots_adjust(bottom=0.05,left=0.10,top=0.99)

plt.fill_between(ygrid,(BFullFs-BFullFe).reshape(size,size)[ind,:],(BFullFs+BFullFe).reshape(size,size)[ind,:], color='grey', alpha=0.4)
plt.plot(xgrid,BFullFs.reshape(size,size)[ind,:], color = 'black', lw = 2.0, alpha = 0.7)   
plt.plot(xgrid,OutFullP.reshape(size,size)[ind,:], color = 'blue', lw = 5.0, alpha = 0.5)  
plt.figtext(0.15,0.85, 'Y - SLICE', fontsize = 20, fontweight = 'bold', color = 'blue' ,alpha = 0.8)
plt.savefig(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+saveplot_name+'_Plot3.png')

#######
val=15.0
ind= np.argmin(np.abs(ygrid-val))
print(xgrid[ind])

fig = plt.figure(4, figsize=(5.,4.))
plt.clf()
plt.cla()
plt.subplots_adjust(bottom=0.05,left=0.10,top=0.99)

plt.fill_between(ygrid,(BFullFs-BFullFe).reshape(size,size)[:,ind],(BFullFs+BFullFe).reshape(size,size)[:,ind], color='grey', alpha=0.4)
plt.plot(ygrid,BFullFs.reshape(size,size)[:,ind], color = 'black', lw = 2.0, alpha = 0.7)   
plt.plot(ygrid,OutFullP.reshape(size,size)[:,ind], color = 'red', lw = 5.0, alpha = 0.5)  
plt.figtext(0.15,0.85, 'X - SLICE', fontsize = 20, fontweight = 'bold', color = 'red' ,alpha = 0.8)

plt.savefig(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_'+saveplot_name+'_Plot4.png')

print(OutFull[nan_inds].shape, xfull[nan_inds].shape, yfull[nan_inds].shape)

### save the file
if do_rescale==1:
    savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_master_map_AvgTime_G'+str(int(grid_step*10**4.))+'_P'+str(int(npoints))+'_K'+str(int(knot_steps*10**4.))+'.npz','wb')
if do_rescale==2:
    savefile_name = open(bpath+rpath+'/ch'+str(ch)+'_'+apdir+'_master_map_AvgGrp_G'+str(int(grid_step*10**4.))+'_P'+str(int(npoints))+'_K'+str(int(knot_steps*10**4.))+'.npz','wb')

pickle.dump([OutFull,xfull, yfull],savefile_name)
savefile_name.close()
 