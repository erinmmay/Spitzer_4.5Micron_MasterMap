import numpy as np
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams

params = {'font.family' : 'serif'}
matplotlib.rcParams.update(params)



def open_file(tpath, ch):
    if ch == 1:
        fname='TY4212o11_p5c.dat'
    if ch == 2:
        fname='bd067bo21_p5c.dat'
    
    handle = open(tpath+fname, 'rb')
    ev5=(pickle.load(handle,encoding='latin1'))

    xV=ev5.x[np.where(ev5.good == 1)]
    yV=ev5.y[np.where(ev5.good == 1)]
    fV=ev5.aplev[np.where(ev5.good == 1)]#/(ev5.exptime)
    fe=ev5.aperr[np.where(ev5.good == 1)]
    ts=ev5.bjdtdb[np.where(ev5.good == 1)]

    inds=np.where(fV>0.0)[0]
    xV=xV[inds]
    yV=yV[inds]
    fV=fV[inds]
    fe=fe[inds]
    ts=ts[inds]

    liml=14.0
    limu=16.0
    inds=np.where((xV>liml) & (xV<limu) & (yV>liml) & (yV<limu))[0]
    xV=xV[inds]
    yV=yV[inds]
    fV=fV[inds]
    fe=fe[inds]
    ts=ts[inds]
    
    return xV, yV, fV, fe, ts[0]
    
def POET_bin_inds(xi,yi,ygrid,xgrid,minnumpts):
    wherebinflux=[]
    wbfipmask=[]
    
    ####
    xgrid = xgrid
    ygrid = ygrid

    ysize=len(ygrid)
    xsize=len(xgrid)
    ystep = np.abs(ygrid[1]-ygrid[0])
    xstep = np.abs(xgrid[1]-xgrid[0])

    for m in range(ysize):
        wbftemp = np.where(np.abs(yi-ygrid[m])-ystep/2. <= 1e-16)[0]
        for n in range(xsize):
            wbf = wbftemp[np.where(np.abs(xi[wbftemp]-xgrid[n])-xstep/2. <= 1e-16)[0]]
                
            if len(wbf)>= minnumpts:
                wherebinflux.append(wbf)
                wbfipmask.append(wbf)
            else:
                wherebinflux.append([])
                wbfipmask.append([])
                
    return wbfipmask
    
def POET_bin(xi,yi,fi,fe,ygrid,xgrid,minnumpts):
    wherebinflux=[]
    wbfipmask=[]
    
    ####
    xgrid = xgrid
    ygrid = ygrid

    ysize=len(ygrid)
    xsize=len(xgrid)
    ystep = np.abs(ygrid[1]-ygrid[0])
    xstep = np.abs(xgrid[1]-xgrid[0])
    
    numpts = np.zeros((ysize,xsize),dtype=int)*np.nan
    for m in range(ysize):
        wbftemp = np.where(np.abs(yi-ygrid[m])-ystep/2. <= 1e-16)[0]
        for n in range(xsize):
            wbf = wbftemp[np.where(np.abs(xi[wbftemp]-xgrid[n])-xstep/2. <= 1e-16)[0]]
                
            if len(wbf)>= minnumpts:
                numpts[m,n]=len(wbf)
                wherebinflux.append(wbf)
                wbfipmask.append(wbf)
            else:
                wherebinflux.append([])
                wbfipmask.append([])
    
    binflux=np.zeros(len(wbfipmask))*np.nan
    binerrs=np.zeros(len(wbfipmask))*np.nan
    for i in range(xsize*ysize):
        binflux[i] = np.mean(fi[wbfipmask[i]])
        binerrs[i] = np.sqrt(np.nansum(fe[wbfipmask[i]]**2.))/len(wbfipmask[i])
                
    return binflux, binerrs, numpts.flatten()
    
def BiLinInterp_inds(oX,oY,nX,nY):
    
    indexes = []
    distances = []
    
    Fout=np.zeros_like(nX)*np.nan
    minx=min(oX)
    maxx=max(oX)
    miny=min(oY)
    maxy=max(oY)
    
    for xi,x in enumerate(nX):
        yi=xi
        y=nY[yi]
        
        if x<minx or x>maxx or y<miny or y>maxy:
            indexes.append([0,0,0,0])
            distances.append([0,0,0,0])
            continue  

        distX=(x-oX)
        distY=(y-oY)
    
        f11=np.nan
        f12=np.nan
        f21=np.nan
        f22=np.nan
        
        p1ind=np.where((distX>0) & (distY>0))[0]
        p1=np.argmin(np.sqrt(distX*distX+distY*distY)[p1ind])

        p2ind=np.where((distX<0) & (distY>0))[0]
        p2=np.argmin(np.sqrt(distX*distX+distY*distY)[p2ind])

        p3ind=np.where((distX<0) & (distY<0))[0]
        p3=np.argmin(np.sqrt(distX*distX+distY*distY)[p3ind])

        p4ind=np.where((distX>0) & (distY<0))[0]
        p4=np.argmin(np.sqrt(distX*distX+distY*distY)[p4ind])
        
        indexes.append([p1ind[p1],p2ind[p2],p3ind[p3],p4ind[p4]])

        x1=oX[p1ind[p1]]
        #x1=oX[p4ind[p4]]

        x2=oX[p2ind[p2]]
        #x2=oX[p3ind[p3]]

        y1=oY[p1ind[p1]]
        #y1=oY[p2ind[p2]]

        y2=oY[p3ind[p3]]
        #y2=oY[p4ind[p4]]
        
        dist1 = ((x2-x)/(x2-x1))*((y2-y)/(y2-y1))
        dist2 = ((x-x1)/(x2-x1))*((y2-y)/(y2-y1))
        dist4 = ((x2-x)/(x2-x1))*((y-y1)/(y2-y1))
        dist3 = ((x-x1)/(x2-x1))*((y-y1)/(y2-y1))
        
        distances.append([dist1,dist2,dist3,dist4])
        
    return indexes,distances
    
def rescale_easy(Ffull,Fnew):
    Where_Full = ~np.isnan(Ffull)
    Where_FNew = ~np.isnan(Fnew)
    overlap_indexes = Where_Full*Where_FNew
    if len(np.where(overlap_indexes)[0])>5:
        rescale_fac = np.nanmean(Ffull[overlap_indexes])/np.nanmean(Fnew[overlap_indexes])
    else:
        rescale_fac = np.nan
    #print(rescale_fac, overlap_indexes, xfull[overlap_indexes],yfull[overlap_indexes])
    return rescale_fac, Where_Full, Where_FNew
    
    
def overlap_plot(ai, a, ndone, FXin, FYin, xVin, yVin, xfull,yfull, FullXs, FullYs, where_full,where_fnew, savepath):
    
    plt.figure(int(ndone+1),figsize=(8,8))
    plt.plot(FullXs, FullYs, '.',color='black', ms=3, alpha=0.1,zorder=0)
    plt.plot(FXin, FYin, '.',color='black', ms=6, alpha=0.5,zorder=1)
    plt.plot(xVin, yVin, '.',color='blue', ms=6, alpha=0.5,zorder=2)
    
    #plt.scatter(xVin,yVin)
    plt.plot(xVin[where_full*where_fnew],yVin[where_full*where_fnew],'.', color='red', ms=12, alpha=0.8,zorder=3)

    plt.axvline(x = np.nanmin(xVin), color='blue', linewidth=0.5)
    plt.axvline(x = np.nanmax(xVin), color='blue', linewidth=0.5)
    plt.axhline(y = np.nanmin(yVin), color='blue', linewidth=0.5)
    plt.axhline(y = np.nanmax(yVin), color='blue', linewidth=0.5)

    plt.axvline(x = np.nanmin(FXin), color='black', linewidth=0.5)
    plt.axvline(x = np.nanmax(FXin), color='black', linewidth=0.5)
    plt.axhline(y = np.nanmin(FYin), color='black', linewidth=0.5)
    plt.axhline(y = np.nanmax(FYin), color='black', linewidth=0.5)


    plt.title(str(int(ai)) + '_' + str(int(a)) + '_' + str(int(ndone)))

    plt.ylim(14.75,15.45)
    plt.xlim(14.75,15.45)

    plt.savefig(savepath+str(ndone).zfill(4)+'.png')

    #plt.show(block=False)
    plt.close(int(ndone+1))
    
def interp_indexes(in_flux, in_y, in_x, fully, fullx, size, gap,frac):
    new_indexes = np.array([])
    
    miny = np.nanmin(fully)
    maxy = np.nanmax(fully)
    minx = np.nanmin(fullx)
    maxx = np.nanmax(fullx)
    
    for yi in range(size):
        if yi % 100 == 0:
            print('     -->', yi, np.round(in_y[yi],2))
        if in_y[yi] < miny or in_y[yi] > maxy:
            continue
        #print('*** ',yi, in_y[yi],' ***')
        for xi in range(size):
            if in_x[xi] < minx or in_x[xi] > maxx:
                continue
                
            fi = (yi*size) + xi
            f_p = in_flux[fi]
            if np.isfinite(f_p):
                continue
            else:
                #find all places in same row within 5 bins
                yd = int(np.nanmax([0,yi-gap]))
                yu = int(np.nanmin([yi+gap, size]))
                yd_ar = np.arange(yd,yi,1)
                yu_ar = np.arange(yi,yu,1)
                fy_d = (yd_ar*size) + xi
                fy_u = (yu_ar*size) + xi
                fluxes_near_yD = in_flux[fy_d]  
                fluxes_near_yU = in_flux[fy_u]
                # for hi, h in enumerate(fy_d):
                #     if h in skip_inds:
                #         fluxes_near_yD[hi] = np.nan
                # for hi, h in enumerate(fy_u):
                #     if h in skip_inds:
                #         fluxes_near_yU[hi] = np.nan
                
                xd = int(np.nanmax([0,xi-gap]))
                xu = int(np.nanmin([xi+gap, size]))
                fx_d = (yi*size) + xd
                fx_u = (yi*size) + xu
                fluxes_near_xD = in_flux[fx_d:fi]
                fluxes_near_xU = in_flux[fi:fx_u]
                # for hi, h in enumerate(np.arange(fx_d,fi, 1)):
                #     if h in skip_inds:
                #         fluxes_near_xD[hi] = np.nan
                # for hi, h in enumerate(np.arange(fi,fx_u, 1)):
                #     if h in skip_inds:
                #         fluxes_near_xU[hi] = np.nan
                
                nmatch = frac*gap
                if np.sum(np.isfinite(fluxes_near_yD))>=nmatch or np.sum(np.isfinite(fluxes_near_yU))>=nmatch:            # find 1
                    if np.sum(np.isfinite(fluxes_near_xD))>=nmatch and np.sum(np.isfinite(fluxes_near_xU))>=nmatch:       # need to find 2
                        new_indexes = np.append(new_indexes,fi)
                if np.sum(np.isfinite(fluxes_near_yD))>=nmatch and np.sum(np.isfinite(fluxes_near_yU))>=nmatch:           # find 2
                    if np.sum(np.isfinite(fluxes_near_xD))>=nmatch or np.sum(np.isfinite(fluxes_near_xU))>=nmatch:        # need to find 1
                        new_indexes = np.append(new_indexes,fi)
    
    return new_indexes.astype(int)
    
def remove_indexes(in_flux, in_y, in_x, fully, fullx, size, edge):
    rm_indexes = np.array([])
    
    miny = np.nanmin(fully)
    maxy = np.nanmax(fully)
    minx = np.nanmin(fullx)
    maxx = np.nanmax(fullx)
    
    for yi in range(size):
        if yi % 100 == 0:
            print('     -->', yi, np.round(in_y[yi],2))
        # if in_y[yi] < miny or in_y[yi] > maxy:
        #     continue
        #print('*** ',yi, in_y[yi],' ***')
        for xi in range(size):
            # if in_x[xi] < minx or in_x[xi] > maxx:
            #     continue
                
            fi = (yi*size) + xi
            f_p = in_flux[fi]
            if np.isfinite(f_p):
                #check suroundings -> if more than or equal to "edge" are nan remove this point
                yd = int(np.nanmax([0,yi-1]))
                yu = int(np.nanmin([yi+1, size]))
                fy_d = (yd*size) + xi
                fy_u = (yu*size) + xi
                fn_yD = in_flux[fy_d]  
                fn_yU = in_flux[fy_u]
                
                xd = int(np.nanmax([0,xi-1]))
                xu = int(np.nanmin([xi+1, size]))
                fx_d = (yi*size) + xd
                fx_u = (yi*size) + xu
                fn_xD = in_flux[fx_d]
                fn_xU = in_flux[fx_u]
                
                if np.sum(np.isnan([fn_yD, fn_yU, fn_xD, fn_xU])) >= edge:
                    # print(fn_yD, fn_yU, fn_xD, fn_xU)
                    rm_indexes = np.append(rm_indexes, fi)        
    
    return rm_indexes.astype(int)
              


