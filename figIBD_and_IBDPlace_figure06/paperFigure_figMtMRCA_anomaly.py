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

#this plots mean time to MRCA as a function of
# A) vrs Ldiff for La=0, B) vrs Ldiff for Ladv.ne.0
# C) ? or 3 panel plot   D) vrs Ladv for Ladv.ne.0 (or all Ladv?)

figure(1,figsize=(8.5,8.5)); clf(); style.use('ggplot')

#MULTIPLY STARTVEC BY 2 BECAUSE RELEASE AT CENTER-STARTVEC AND CENTER+STARTVEC

#==========================================
subplot(2,2,1)
iterateVec=[]

#nameTemplate='bySeperation_centerPoint_256_Ladv_0_Ldiff_%d.pickle'
#iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=0, L_{diff}=%d$, start at 256'%(p,)) for p in [10,20,40]]
#iterateVec=iterateVec+iterateVecNow

nameTemplate='bySeperation_centerPoint_512_Ladv_0_Ldiff_%d.pickle'
iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=0, L_{diff}=%d$'%(p,)) for p in [10,20,40]]
iterateVec=iterateVec+iterateVecNow

#nameTemplate='bySeperation_centerPoint_768_Ladv_0_Ldiff_%d.pickle'
#iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=0, L_{diff}=%d$, start at 768'%(p,)) for p in [10,20,40]]
#iterateVec=iterateVec+iterateVecNow

for thisRun in iterateVec:
    plotLabel=thisRun[1]
    fileName=thisRun[0]
    
    with open('MRCA_pickleShed/'+fileName,'rb') as fid:
        dataIn=pickle.load(fid)
    startPoints=dataIn['startPoints']
    startVec=dataIn['startVec']
    output=dataIn['output']
    scaleName=dataIn['scaleName']
    
    #now gather answers
    fracCoalescVec=zeros(len(startPoints))+nan
    meanTimeVec=zeros(len(startPoints))+nan
    medianTimeVec=zeros(len(startPoints))+nan

    for n in range(len(startPoints)):
        fracCoalescVec[n]=output[n][0]
        meanTimeVec[n]=output[n][2]
        medianTimeVec[n]=output[n][3]

    #now plot
    #the times 2 is because two points are launched at centerpoint +/- pnt where pnt is element of startVec
    plot(startVec*2,meanTimeVec-meanTimeVec[0],label=plotLabel)
    xlabel(scaleName)
    ylabel('mean time to MRCA')
legend(fontsize='medium')

#==========================================
subplot(2,2,2)
iterateVec=[]

# nameTemplate='bySeperation_centerPoint_256_Ladv_16_Ldiff_%d.pickle'
# iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=16, L_{diff}=%d$, start at 256'%(p,)) for p in [10,20,40]]
# iterateVec=iterateVec+iterateVecNow

nameTemplate='bySeperation_centerPoint_512_Ladv_16_Ldiff_%d.pickle'
iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=16, L_{diff}=%d$'%(p,)) for p in [10,20,40]]
iterateVec=iterateVec+iterateVecNow

# nameTemplate='bySeperation_centerPoint_768_Ladv_16_Ldiff_%d.pickle'
# iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=16, L_{diff}=%d$, start at 768'%(p,)) for p in [10,20,40]]
# iterateVec=iterateVec+iterateVecNow

for thisRun in iterateVec:
    #param=thisRun[0]
    plotLabel=thisRun[1]
    fileName=thisRun[0]
    
    with open('MRCA_pickleShed/'+fileName,'rb') as fid:
        dataIn=pickle.load(fid)
    startPoints=dataIn['startPoints']
    startVec=dataIn['startVec']
    output=dataIn['output']
    scaleName=dataIn['scaleName']
    
    #now gather answers
    fracCoalescVec=zeros(len(startPoints))+nan
    meanTimeVec=zeros(len(startPoints))+nan
    medianTimeVec=zeros(len(startPoints))+nan

    for n in range(len(startPoints)):
        fracCoalescVec[n]=output[n][0]
        meanTimeVec[n]=output[n][2]
        medianTimeVec[n]=output[n][3]

    #now plot
    #the times 2 is because two points are launched at centerpoint +/- pnt where pnt is element of startVec
    plot(startVec*2,meanTimeVec-meanTimeVec[0],label=plotLabel)
    xlabel(scaleName)
    ylabel('mean time to MRCA')
legend(fontsize='medium')

#==========================================
subplot(2,2,3)
iterateVec=[]

nameTemplate='bySeperation_centerPoint_256_Ladv_%d_Ldiff_20.pickle'
iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=%d, L_{diff}=20$, start at 256'%(p,)) for p in [16]]
iterateVec=iterateVec+iterateVecNow

nameTemplate='bySeperation_centerPoint_512_Ladv_%d_Ldiff_20.pickle'
iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=%d, L_{diff}=20$, start at 512'%(p,)) for p in [16]]
iterateVec=iterateVec+iterateVecNow

nameTemplate='bySeperation_centerPoint_768_Ladv_%d_Ldiff_20.pickle'
iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=%d, L_{diff}=20$, start at 768'%(p,)) for p in [16]]
iterateVec=iterateVec+iterateVecNow

for thisRun in iterateVec:
    #param=thisRun[0]
    plotLabel=thisRun[1]
    fileName=thisRun[0]
    
    with open('MRCA_pickleShed/'+fileName,'rb') as fid:
        dataIn=pickle.load(fid)
    startPoints=dataIn['startPoints']
    startVec=dataIn['startVec']
    output=dataIn['output']
    scaleName=dataIn['scaleName']
    
    #now gather answers
    fracCoalescVec=zeros(len(startPoints))+nan
    meanTimeVec=zeros(len(startPoints))+nan
    medianTimeVec=zeros(len(startPoints))+nan

    for n in range(len(startPoints)):
        fracCoalescVec[n]=output[n][0]
        meanTimeVec[n]=output[n][2]
        medianTimeVec[n]=output[n][3]

    #now plot
    #the times 2 is because two points are launched at centerpoint +/- pnt where pnt is element of startVec
    plot(startVec*2,meanTimeVec-meanTimeVec[0],label=plotLabel)
    xlabel(scaleName)
    ylabel('mean time to MRCA')
legend(fontsize='medium')

#==========================================
subplot(2,2,4)
iterateVec=[]

# nameTemplate='bySeperation_centerPoint_256_Ladv_%d_Ldiff_20.pickle'
# iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=%d, L_{diff}=20$, start at 256'%(p,)) for p in [8,16,32]]
# iterateVec=iterateVec+iterateVecNow

nameTemplate='bySeperation_centerPoint_512_Ladv_%d_Ldiff_20.pickle'
iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=%d, L_{diff}=20$'%(p,)) for p in [8,16,32]]
iterateVec=iterateVec+iterateVecNow

# nameTemplate='bySeperation_centerPoint_768_Ladv_%d_Ldiff_20.pickle'
# iterateVecNow=[(nameTemplate%(p,),r'$L_{adv}=%d, L_{diff}=20$, start at 768'%(p,)) for p in [8,16,32]]
# iterateVec=iterateVec+iterateVecNow

for thisRun in iterateVec:
    #param=thisRun[0]
    plotLabel=thisRun[1]
    fileName=thisRun[0]
    
    with open('MRCA_pickleShed/'+fileName,'rb') as fid:
        dataIn=pickle.load(fid)
    startPoints=dataIn['startPoints']
    startVec=dataIn['startVec']
    output=dataIn['output']
    scaleName=dataIn['scaleName']
    
    #now gather answers
    fracCoalescVec=zeros(len(startPoints))+nan
    meanTimeVec=zeros(len(startPoints))+nan
    medianTimeVec=zeros(len(startPoints))+nan

    for n in range(len(startPoints)):
        fracCoalescVec[n]=output[n][0]
        meanTimeVec[n]=output[n][2]
        medianTimeVec[n]=output[n][3]

    #now plot
    #the times 2 is because two points are launched at centerpoint +/- pnt where pnt is element of startVec
    plot(startVec*2,meanTimeVec-meanTimeVec[0],label=plotLabel)
    xlabel(scaleName)
    ylabel('mean time to MRCA')
legend(fontsize='medium')



suptitle('Mean time to MRCA')
tight_layout()
ion()
draw()
show()


savefig('figMtMRCA_anomaly.png',dpi=300)
savefig('figMtMRCA_anomaly.svg',dpi=300)