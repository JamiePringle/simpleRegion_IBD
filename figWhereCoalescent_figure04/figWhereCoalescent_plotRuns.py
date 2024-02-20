from pylab import *
from numpy import *
import zarr
import bisect
import time
import multiprocessing as mp
from itertools import islice
import gc
import random as randomModule
from numba import jit
from collections import Counter
import os
import shutil
import glob
from sklearn.neighbors import KernelDensity
import pickle 

#this code plots the output of figWhereCoalescent_makeData.py

__spec__=None
if __name__=='__main__':

    runNameVec=['extrasmallDomain_Ladv','verysmallDomain_Ladv','smallDomain_Ladv','mediumDomain_Ladv']
    runNameSize=[64,128,256,512,1024] #size of domain

    nRunName=-1
    for runName in runNameVec:
        nRunName+=1

        #if first run, make figures and subplots
        if nRunName==0:
            figure(1,figsize=(8.5,8.5))
            clf()
            style.use('ggplot')
            fig1ax221=subplot(2,2,1)
            fig1ax222=subplot(2,2,2)
            fig1ax223=subplot(2,2,3)

            figure(2,figsize=(8.5,8.5))
            clf()
            style.use('ggplot')
            #fig2ax221=subplot(2,2,1)
            fig2ax222=subplot(1,1,1)
            #fig2ax223=subplot(2,2,3)

            figure(3,figsize=(8.5,8.5))


        #get data
        with open(runName+'.pickle', 'rb') as handle:
            dataIn=pickle.load(handle)
        answerVec=dataIn['answerVec']
        startPoints=dataIn['startPoints']
        param1vec=dataIn['param1vec']
        param1name=dataIn['param1name']
        scaleName=dataIn['scaleName']
        nPatch=dataIn['nPatch']
        Np=dataIn['Np']

        #loop over data and plot
        for nParam1 in range(len(answerVec)):

            #now gather answers for all runs so far.
            if nParam1==0:
                fracCoalescVec=zeros((len(param1vec),len(startPoints)))+nan
                meanTimeVec=zeros((len(param1vec),len(startPoints)))+nan
                medianTimeVec=zeros((len(param1vec),len(startPoints)))+nan
            for n in range(len(startPoints)):
                fracCoalescVec[nParam1,n]=answerVec[nParam1][n][0]
                meanTimeVec[nParam1,n]=answerVec[nParam1][n][2]
                medianTimeVec[nParam1,n]=answerVec[nParam1][n][3]

            #now make data for plots of where coalescent occurs
            nOutputs=len(startPoints)

            #if first parameter, make arrays to contain PDF of where coalescent occurs
            if nParam1==0:
                plotAxis=arange(Np)
                whereCoalesc=nan+zeros((len(plotAxis),len(param1vec),nOutputs))

            #now for each output, use KDE to make smooth plot of where coalescent
            for n in range(nOutputs):
                bandwidth=3.0
                output=answerVec[nParam1]
                thisWhere=output[n][1]
                del thisWhere[-1] # get rid of points that never coalesced
                nPnts=len(thisWhere)
                thisWhereMat=zeros((nPnts,1))
                thisWhereMat[:,0]=array([p for p in thisWhere])
                thisWeights=array([thisWhere[p] for p in thisWhere])
                kde=KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(thisWhereMat,
                                                                              sample_weight=thisWeights)
                toFit=arange(Np)+0.0; toFit=toFit.reshape(-1,1)
                whereCoalesc[:,nParam1,n]=exp(kde.score_samples(toFit))
            


        #now plot
        figure(1)
        
        sca(fig1ax221)
        for n in range(len(startPoints)):
            plot(param1vec,fracCoalescVec[:,n],'*-',label=startPoints[n][2])
        xlabel(param1name)
        ylabel('fracCoalesc')
        legend(fontsize='small')

        sca(fig1ax222)
        for n in range(len(startPoints)):
            plot(param1vec,meanTimeVec[:,n],'*-',label=startPoints[n][2])
        xlabel(param1name)
        ylabel('mean time to MRCA')

        sca(fig1ax223)
        for n in range(len(startPoints)):
            plot(param1vec,medianTimeVec[:,n],'*-',label=startPoints[n][2])
        xlabel(param1name)
        ylabel('median time to MRCA')


        draw()
        ion()
        show()

        #now plot SCALED
        figure(2)

        #COMPUTE scaling
        if param1name=='Ldiff':
            scaleValue=((param1vec**2)/5.0)*nPatch; scaleName='Ldiff**2/Ladv'
        elif param1name=='Ladv':
            scaleValue=(50.0**2/(param1vec))*nPatch; scaleName='Ldiff**2/Ladv'

        # sca(fig2ax221)
        # for n in range(len(startPoints)):
        #     plot(param1vec,fracCoalescVec[:,n],'*-',label=startPoints[n][2])
        # xlabel(param1name)
        # ylabel('fracCoalesc')
        # legend(fontsize='small')

        sca(fig2ax222)
        for n in range(len(startPoints)):
            plot(2*scaleValue,meanTimeVec[:,n],'*-',label=startPoints[n][2])
        xlabel(r'2*$L_{reten}$')
        ylabel('mean time to MRCA')

        #draw a 1:1 line a the very end, over everything
        if nRunName==(len(runNameVec)-1):
            jnk=arange(100.0,2000.0)
            plot(jnk,jnk,':',alpha=0.7)

        #plot some lines indicating something about Np
        xmax=12000
        plot([0.0,xmax],5*array([runNameSize[nRunName],runNameSize[nRunName]]),'k:',alpha=0.4)
            
        # sca(fig2ax223)
        # for n in range(len(startPoints)):
        #     plot(2*scaleValue,medianTimeVec[:,n],'*-',label=startPoints[n][2])
        # xlabel(r'2*$L_{reten}$')
        # ylabel('median time to MRCA')

        suptitle('All scaled by '+scaleName)

        draw()
        ion()
        show()

        #========================================================================
        #now plot where coalesncent occurs

        figure(3)

        #choose whichpoint to plot
        whichPoint=0

        subplot(1,len(runNameVec),nRunName+1)

        #normalize so each Ladv has max PDF of 1
        toPlot=whereCoalesc[:,:,whichPoint].T
        for nn in range(toPlot.shape[0]):
            toPlot[nn,:]=toPlot[nn,:]/toPlot[nn,:].max()

        pcolormesh(arange(Np),param1vec,toPlot)
        colorbar(location='bottom',shrink=0.7,fraction=0.02,pad=0.1)
        title('where coalesc, start=%s'%(startPoints[n][2],),fontsize='medium')
        if n==0:
            ylabel(param1name)
        xlabel('distance')

        plot(50.0**2/param1vec,param1vec,'r-*')
        axis(xmax=Np-1)

        draw; pause(0.1)
        
        draw()
        ion()
        show()
        pause(0.2)

    if True:
        #save plots
        figure(3); savefig('figWhere_coalescent.png',dpi=300)
        figure(1); savefig('figWhere_coalescent_details.png',dpi=300)
        figure(2); savefig('figWhere_coalescent_details_scaled.png',dpi=300)
        
