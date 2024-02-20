from pylab import *
from numpy import *
import zarr
import bisect
import time
import multiprocessing as mp
from itertools import islice
import gc
import random as randomModule
#from numba import jit
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
            figure(1,figsize=(16.25,6))
            clf()
            style.use('ggplot')
            fig1ax222=subplot(1,3,2)
            fig2ax222=subplot(1,3,3)
            fig3ax111=subplot(1,3,1)

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
        
        sca(fig1ax222)
        for n in [1]: #range(len(startPoints)):
            plot(param1vec,meanTimeVec[:,n],'*-',
                 label=r'$L_{diff}$=50km, domain size=%dkm'%(runNameSize[nRunName]))
        #xlabel(param1name+', km')
        xlabel(r'$L_{adv}$, km'); assert param1name=='Ladv','this is a hack'
        ylabel('mean generations to MRCA')
        title('time to MRCA')
        if nRunName==(len(runNameVec)-1):
            legend()

        #now plot SCALED

        #COMPUTE scaling
        if param1name=='Ldiff':
            scaleValue=((param1vec**2)/5.0)*nPatch; scaleName='Ldiff**2/Ladv'
        elif param1name=='Ladv':
            scaleValue=(50.0**2/(param1vec))*nPatch; scaleName='Ldiff**2/Ladv'

        sca(fig2ax222)
        for n in [1]: #range(len(startPoints)):
            plot(2*scaleValue,meanTimeVec[:,n],'*-',label=startPoints[n][2])
        xlabel(r'2*$H_{dens}L_{reten}$')
        ylabel('mean time to MRCA')
        title('time to MRCA')
        axis(xmax=6000)

        #draw a 1:1 line a the very end, over everything
        if nRunName==(len(runNameVec)-1):
            jnk=arange(100.0,2500.0)
            plot(jnk,jnk,'k--',alpha=0.7)

        #plot some lines indicating something about Np
        xmax=12000
        plot([-500.0,xmax],5*array([runNameSize[nRunName],runNameSize[nRunName]]),'k:',alpha=0.4)
            
        # sca(fig2ax223)
        # for n in range(len(startPoints)):
        #     plot(2*scaleValue,medianTimeVec[:,n],'*-',label=startPoints[n][2])
        # xlabel(r'2*$L_{reten}$')
        # ylabel('median time to MRCA')

        #suptitle('All scaled by '+scaleName)

        #========================================================================
        #now plot where coalesncent occurs

        #choose whichpoint to plot
        whichPoint=0

        if nRunName==1:
            sca(fig3ax111)

            #normalize so each Ladv has max PDF of 1
            toPlot=whereCoalesc[:,:,whichPoint].T
            for nn in range(toPlot.shape[0]):
                toPlot[nn,:]=toPlot[nn,:]/toPlot[nn,:].max()

            pcolormesh(arange(Np),param1vec,toPlot)
            colorbar(location='bottom',shrink=0.7,fraction=0.02,pad=0.15)
            title('where coalescent occurs')
            if n==0:
                ylabel(param1name)
            xlabel('alongshore distance')
            ylabel(r'$L_{adv}$')

            plot(50.0**2/param1vec,param1vec,'r-*')
            axis(xmax=Np-1)

        
        draw()
        ion()
        show()
        pause(0.2)

    if True:
        #save plots
        tight_layout()
        savefig('figWhere_coalescent_paperPlot.png',dpi=300)
        
