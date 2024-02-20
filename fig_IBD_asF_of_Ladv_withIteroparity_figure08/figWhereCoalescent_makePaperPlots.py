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


__spec__=None
if __name__=='__main__':

    if True:
        xmax=100.0; xmin=0.6
        figName='figIBD_plot.png' 
        titleStr=r'change in $N_{maxAge}$'
        fileInVec=[
            'largeDomain_Ladv_dOff_200_dOffInit_50_Ldiff_10_nPatch_5_maxAge_1.pickle',
            'largeDomain_Ladv_dOff_200_dOffInit_50_Ldiff_10_nPatch_5_maxAge_2.pickle',
            'largeDomain_Ladv_dOff_200_dOffInit_50_Ldiff_10_nPatch_5_maxAge_5.pickle',
            #
            #'largeDomain_Ladv_dOff_200_Ldiff_5_nPatch_5.pickle',
            #'largeDomain_Ladv_dOff_200_Ldiff_10_nPatch_5.pickle',
            #'largeDomain_Ladv_dOff_200_Ldiff_25_nPatch_5.pickle',
        ]
        descriptVec=[r'Maximum Age of 1 generation',
                     r'Maximum Age of 2 generations',
                     r'Maximum Age of 5 generations'
                     ]

    nFile=-1
    for file in fileInVec:
        nFile+=1
        
        with open(file,'rb') as fid:
            data=pickle.load(fid)
            
        #unpack dictionary
        param1vec=data['param1vec']
        answerVec=data['answerVec']
        Np=data['Np']
        param1name=data['param1name']
        param2name=data['param2name']
        startPoints=data['startPoints']
        param2=data['param2']
        dOff=data['dOff']
        nPatch=data['nPatch']
        maxAge=data['maxAge']

        #now gather answers for all runs so far.
        for nParam1 in range(len(param1vec)):
            if nParam1==0:
                fracCoalescVec=zeros((len(param1vec),len(startPoints)))+nan
                meanTimeVec=zeros((len(param1vec),len(startPoints)))+nan
                medianTimeVec=zeros((len(param1vec),len(startPoints)))+nan
            for n in range(len(startPoints)):
                fracCoalescVec[nParam1,n]=answerVec[nParam1][n][0]
                meanTimeVec[nParam1,n]=answerVec[nParam1][n][2]
                medianTimeVec[nParam1,n]=answerVec[nParam1][n][3]

        #========================================================================
        figure(1,figsize=[8.5,8.5]); clf(); style.use('ggplot')
        subplot(2,2,2)
        for n in range(len(startPoints)):
            plot(param1vec,meanTimeVec[:,n],'*-',label=startPoints[n][2])
        xlabel(param1name)
        ylabel('mean time to MRCA')

        subplot(2,2,3)
        for n in range(len(startPoints)):
            plot(param1vec,medianTimeVec[:,n],'*-',label=startPoints[n][2])
        xlabel(param1name)
        ylabel('median time to MRCA')

        subplot(2,2,4)
        ibdVec=(meanTimeVec[:,1]-meanTimeVec[:,0])
        plot(param1vec,ibdVec,'*-')
        xlabel(param1name)
        ylabel('change in mean time to MRCA')

        suptitle(file)

        #========================================================================
        if file==fileInVec[0]:
            figure(2,figsize=[8.5,8.5]); clf(); style.use('ggplot')
        else:
            figure(2)

        ibdVec=(meanTimeVec[:,1]-meanTimeVec[:,0])
        plot(log10(param1vec),log10(ibdVec),'*-',label=file)
        xlabel('log10('+param1name+')')
        ylabel('log 10 change in mean time to MRCA')

        suptitle('IBD for Np=%d and dOff=%d'%(Np,dOff))

        #========================================================================
        if file==fileInVec[0]:
            figure(3,figsize=array([13.8,7.4])); clf(); style.use('ggplot')
            ax1=subplot(1,2,1)
            ax2=subplot(1,2,2)
        else:
            figure(3)

        ibdVec=(meanTimeVec[:,1]-meanTimeVec[:,0])
        scaleValue=(dOff/2)/(((maxAge+1)/2)*param1vec)
        scaleValue_noAge=(dOff/2)/(param1vec)
        if False:
            plot(log10(param1vec),log10(ibdVec/scaleValue),'*-',label=file)
            #plot(log10(param1vec),log10(scaleValue),'o-k')        
            #plot(log10(param1vec),log10(0.5*scaleValue),'o-k')        
            xlabel('log10('+param1name+')/scaleValue')
            ylabel('log 10 change in mean time to MRCA/scaleValue')
            suptitle('IBD for Np=%d and dOff=%d'%(Np,dOff))
        else:
            if False: #linear plots
                plot(scaleValue,ibdVec,'-*',label=file)
                plot(scaleValue,scaleValue,'k-',alpha=0.2)
                plot(scaleValue,1.333*scaleValue,'k-',alpha=0.2)
                axis(xmin=-1.0,ymin=-1.0,xmax=xmax,ymax=xmax)
            else:
                #log log plots
                sca(ax1)
                loglog(scaleValue_noAge,ibdVec,'-*',label=descriptVec[nFile])
                loglog(scaleValue_noAge,scaleValue_noAge,'k-',alpha=0.2)
                loglog(scaleValue,1.333*scaleValue,'k-',alpha=0.2)
                axis(xmin=xmin,ymin=xmin,xmax=xmax,ymax=xmax)    
                xlabel('scaled change in time to MRCA from equation 8')
                ylabel('modeled change in time to MRCA')
                title('no correction for lifespan')

                sca(ax2)
                loglog(scaleValue_noAge,ibdVec/((1+maxAge)*0.5),'-*',label=descriptVec[nFile])
                loglog(scaleValue_noAge,scaleValue_noAge,'k-',alpha=0.2)
                loglog(scaleValue_noAge,1.333*scaleValue_noAge,'k-',alpha=0.2)
                axis(xmin=xmin,ymin=xmin,xmax=xmax,ymax=xmax)    
                xlabel('scaled change in time to MRCA from equation 8')
                ylabel('modeled change in time to MRCA scaled by $(N_{age}+1)/2$')
                title('corrected for lifespan')

            suptitle(titleStr)

            
        #========================================================================
        if file==fileInVec[0]:
            figure(4,figsize=[8.5,8.5]); clf(); style.use('ggplot')
        else:
            figure(4)

        ibdVec=(meanTimeVec[:,1]-meanTimeVec[:,0])
        plot(scaleValue,1-fracCoalescVec[:,1],'-*',label=file)
        xlabel('scaled for IBD')
        ylabel('1-fracCoalesc')
        suptitle('frac Coalesc')
            



    figure(3); sca(ax1); legend(fontsize='large') ; tight_layout(); savefig(figName,dpi=300)
    figure(4); legend(fontsize='small')
    draw()
    show()
    
               
