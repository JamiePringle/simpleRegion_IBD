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

#this code calculates a connectivity matrix E for a 1D domain Np in
#size and calculates a large number of backwards trajectories that are
#used in a subsequent code to calculate the time to MRCA and fraction
#of coalescence for this idealized domain.

#the units of distance are the size of each individual patch, and
#all lengthscales should be larger than the patch. 

#define dispersal kernel
def dispKernel(nxFrom,nxTo,param1,param2,param1name,param2name):
    '''nxFrom is the starting location of offspring nxTo is where
    they end up. nxFrom is a scaler, nxTo can be a vector.

    The kernel must be normalized so that if nxTo went from -infinity
    to infinity by 1, the sum of the kernel would be 1.0. This means
    that particles that leave the domain die.

    param1 and param2 are used to vary dispersal in iterated runs, and
    you will have to look below to see what they do (if anything). 

    '''

    if (param1name=='Ladv') and (param2name=='Ldiff'):
        scale=1
        Ladv=param1*scale
        Ldiff=param2*scale
        if Ladv>0.0:
            print('Lreten is',Ldiff**2/Ladv,'La,Ld=',Ladv,Ldiff)
        else:
            print('Ladv is',Ladv,'Ldiff is',Ldiff)
    else:
        assert False,'pick something'

    K=(1/(Ldiff*sqrt(2.0*pi)))*exp(-(nxTo-(nxFrom+Ladv))**2/(2*Ldiff**2))

    #if True, then for a distance distSticky around midSticky, have a
    #different Ladv and Ldiff to simulate a retention zone
    if False:
        midSticky=236
        distSticky=40
        if abs(nxFrom-midSticky)<distSticky/2:
            Ladv=0.0
            Ldiff=1.0/4
            K=(1/(Ldiff*sqrt(2.0*pi)))*exp(-(nxTo-(nxFrom+Ladv))**2/(2*Ldiff**2))

    #if True, then if Nfrom>nBreak, only eps of the individuals that
    #would get to <nBreak survive, and if Nfrom<Nbreak, only eps of
    #the individuals that would get to >nBreak survive.
    if False:
        nBreak=256
        eps=0.01/5
        if nxFrom>nBreak:
            K[:nBreak]=K[:nBreak]*eps
        else:
            K[nBreak:]=K[nBreak:]*eps
    
    #normalize K DON'T NORMALIZE BECAUSE THEN NOTHING CAN BE LOST OFF OF BOUNDARIES 
    #K=K/sum(K)
    
    return K

def visualizeMat(E,nStart,nTime):
    figure(1)
    style.use('ggplot')
    clf()

    subplot(1,3,1)
    pcolormesh(E)
    colorbar()

    ax2=subplot(1,3,2)
    ax3=subplot(1,3,3)

    #initial condition
    popVec=zeros((Np,)); popVec[Np//2]=1.0
    popVec=zeros((Np,)); popVec[nStart]=1.0

    totPop=[]
    for ng in range(nTime):
        popVec=E@popVec
        totPop.append(sum(popVec))

        if ng==(nTime-1):
            sca(ax2)
            cla()
            plot(popVec,'b-')
            title('nGen '+str(ng))
            
            sca(ax3)
            plot(totPop,'r*')    
            draw(); show(); pause(0.01)

    return(ax2)

#now make a function that takes a list or vector nStart of starting
#positions and nGen for the number of generations to run, and returns
#a matrix of (len(nStart),nGen) backwards paths of the particles.
def backPaths(EbackCDF,startVec,nGen):
    paths=zeros((len(startVec),nGen),dtype=dtype('int16'))
    paths[:,0]=startVec
    for ng in range(1,nGen):
        theRands=random.uniform(size=(len(startVec),1))
        for np in range(len(startVec)):
            paths[np,ng]=bisect.bisect_left(EbackCDF[:,paths[np,ng-1]],theRands[np])
            #assert paths[np,ng]<EbackCDF.shape[1],'think'
    return(paths)
    

# This code computes coalecent estimates between all pairs of starting points. 
# now make functio to estimate time to MRCA for a series of individuals in the
# simulation, all that start at the same location. These are the
# parameters beyond nPath and nGen:
#
# nPatch; the nominal population of a patch, so that if two lineages
# are in the same patch at the same time, they have a 1/nPatch chance
# of having coalesced. After coalescent, the lineage ends? Need to
# think about this.
#
# results of this will be two sets of information; a vector of the
# time it took for the individual to coaslesce, and a counter whose
# key is (nx,ny) of where a coalescent event occured and whose value
# is how many events occured there.  The vector of times will record a
# -1 if there is no coalescent in a time nGen.

@jit(nopython=True)
def innerCore(paths1,paths2,nGen,nPatch):
    #set up output vectors; if I don't set the type in the offset of -1, numba
    #will promote the arrays to int, which is unhelpful...
    nPaths1=paths1.shape[0]
    nPaths2=paths2.shape[0]
    nTimeVec=zeros((nPaths1,nPaths2),dtype=dtype('int16')) -int16(1)
    nWhereVec=zeros((nPaths1,nPaths2),dtype=dtype('int16')) -int16(1)

    #we will check if nothing has been found in nThresh paths in
    #Paths1.  if it has not, then do not continue. The variable
    #foundAny is part of that, and is only true if some coalescense
    #has occured
    nThresh=100
    foundAny=False
    
    #we loop over all pairs of paths
    for nTrace1 in arange(paths1.shape[0]):
        for nTrace2 in arange(paths2.shape[0]):

            nTime=-1 #what it will be if no coalescent occurs
            nWhere=-1  #what it will be if no coalescent occurs

            samePlaceSameTime=paths1[nTrace1,:]==paths2[nTrace2,:]

            #only do inner loop if some chance of coalescense this
            #shortcut only saves time if looping in numba is still
            #expensive. Removing if test below will not change result;
            #perhaps it will make things faster. 
            if samePlaceSameTime.any():
                #now run except in case where n1==n2 and nTrace1==nTrace2
                #(because you can't coalesc with yourself. NOTE, there is a
                #slight bug in this because it will count the same point and
                #same patch as a failure to coalesc, which is not right
                if not ((nTrace1==nTrace2) and samePlaceSameTime.all()):
                    for ng in arange(1,nGen+1): #start after one generation of dispersing
                        #if both tracks are at the same place, the odds of a coalescent is 1/nPatch
                        #THE FORM OF THE RANDOM GENERATOR NEEDS THOUGHT
                        if samePlaceSameTime[ng] and (random.rand()<=1.0/nPatch):
                            nTime=ng
                            nWhere=paths1[nTrace1,ng]
                            foundAny=True
                            break #bail out of the rest of the loop

            #record data for this pair
            nTimeVec[nTrace1,nTrace2]=nTime
            nWhereVec[nTrace1,nTrace2]=nWhere

        #check if at nTrace1==nThresh, if none have been found, then bail
        if (nTrace1==nThresh) and (not foundAny):
            break

    return nTimeVec,nWhereVec

#@profile
def twoTrackCheck(point1,point2,nPatch,nGens,nPaths,EbackCDF,nRun):
    '''def oneTrackCheck(nTrace,nStart)

    This code calculates where and when a set of paths from two
    starting points coalesce

    point1 and point2 are the starting locations of the individuals to
    compare

    nPatch is the population of each deme -- this effects the
    likelyhood of coalescence

    nGens is the number of generations to calculate the backward path for

    nPaths is the number of backwards paths to calculate

    EbackCDF is the backward population matrix turned into a CDF, to
    quickly calculate backwards paths. It is defined below
    
    nRun is just a number passed in to be printed out below; it does
    nothing else in the code


    A coalescent event is assumed to occur for two individuals from
    the two runs in the same patch at the same time with probability
    1/nPatch.

    The question we are asking is if we have two individuals, one
    specified by nTrace, the other one starting in the same patch as
    the nTrace one (call it nOther), where are they likely to
    coalesce?

    So at time ng, both particle nTrace and another (call it nP) are
    in the same patch. What is the likelyhood that nP is the same as
    nOther? 1/nPath!  And if it is nOther, what is the likelyhood that
    it will coalesce at this patch at this generation? 1/nPatch. So
    the chance that a track nP is nOther and will coalesce in this
    place and this generation is 1/(nPath*nPatch).

    Of course, we will compare nTrace against all other particles in
    nPath, the chance that there will be coalescent event between
    nTrace and all the other particles will be
    Prob_i*nPath*(1/(nPath*nPatch))=Prob_i*(1/nPatch), where Prob_i is
    the likelyhood that a particle come from the starting point to the
    i-th patch.

    '''

    tic=time.time()

    print('Starting run',nRun,'for',point1,point2,flush=True)

    #make paths for point1 and point2. Don't parallelize so that this function can be called in parallel
    startVec=point1+zeros((nPaths,),dtype=dtype('int16'))
    paths1=backPaths(EbackCDF,startVec,nGens)

    startVec=point2+zeros((nPaths,),dtype=dtype('int16'))
    paths2=backPaths(EbackCDF,startVec,nGens)

    
    #call inner core (the commented code above) which has been jited
    #in numba.  this is a seperate code to make as simple and able to
    #be done with numba as possible.
    nTimeVec,nWhereVec=innerCore(paths1,paths2,nGens,nPatch)

    #sigh, pay attention to memory
    del paths1
    del paths2
    gc.collect()

    fracCoalesc=sum(nTimeVec>0)/prod(nWhereVec.shape)
    whereCoalesce=Counter(nWhereVec.flatten())
    indx=nTimeVec>0
    meanTime=mean(nTimeVec[indx])
    medianTime=median(nTimeVec[indx])

    print('   Done with run',nRun,'for',point1,point2,
          'in %5.2fs'%(time.time()-tic,),
          'and %4.2f coalesced'%(fracCoalesc,),
          flush=True)
    
    return fracCoalesc,whereCoalesce,meanTime,medianTime


def makeConnectivityMatrices(Np,param1,param2,param1name,param2name):
    #make E, Eback and EbackCDF

    #now lets make connectivity matrix E.shape=(Np,Np), where
    #E[nTo,nFrom]
    E=zeros((Np,Np))
    nAxis=arange(Np+0.0)
    for n in range(Np):
        E[:,n]=dispKernel(n,nAxis,param1,param2,param1name,param2name)

    #now, if true, visualize the matrix and the evolution in time of an
    #introduction somewhere in the domain.
    if False:
        visualizeMat(E,4,300)
        assert False,'asdf'

    #now make a backwards in time matrix. This matrix
    #Eback*popVec(time=2)=Eback*popVec(time=1) for LINEAGES. Remember that
    #lineages cannot leave the domain, since each child must have a
    #parent.  Eback is aranged as Eback[nTo,nFrom] where "from" is before
    #in time "to", as above.
    #
    #We normalize it so that sum(Eback[:,nFrom])=1.0 because ALL offspring
    #must have a parent
    Eback=E.T.copy() #transpose so Eback[nTo,nFrom]
    for n in range(Np):
        #Eback[n,:]=Eback[n,:]/sum(Eback[n,:])
        Eback[:,n]=Eback[:,n]/sum(Eback[:,n])

    #given a particle at nTo, how can we pick an nFrom it came from,
    #randomly, with the appriopriate odds? Make each Column of Eback into
    #a cumilative probability function by summing allong the rows, to make
    #EbackCDF. Then pick a random number between 0 and 1 inclusive and use
    #bisection to find where the particle came from. So if the
    #CDF=[a,b,c,d,e], where 0<=a<=b<=c<=d<=e<=1.... (e should be 1).  Pick
    #a random number x and if x<=a, then it goes into location 0, if
    #a<x<=b it goes in 1, and if d<x<=e then it goes to location 4.
    EbackCDF=nan*Eback
    for n in range (Np):
        EbackCDF[:,n]=cumsum(Eback[:,n])

    #finite math is imperfect; enforce E[-1,:]=1.0
    EbackCDF[-1,:]=1

    return E,Eback,EbackCDF

__spec__=None
if __name__=='__main__':

    #we can iterate over param1 and param2
    #what they do can only be determined by reading the code below
    domainEnbiggen=2
    Np=512*domainEnbiggen ;

    #LaMult=0,1,2 ; LdMult=0.5,1,2
    #LaMult=1; LdMult=1
    for LdMult in [0.5,1,2]:
        for LaMult in [0,1,2,4]:


            if True:
                centerPoint=Np//2;
                centerPoint=Np//4;
                centerPoint=Np//2+Np//4;
                Ladv=2.0*4*LaMult; Ldiff=5.0*4*LdMult
                runName='bySeperation_centerPoint_%d_Ladv_%d_Ldiff_%d'%(centerPoint,Ladv,Ldiff)
                scaleName='Seperation'

            #WARNING, LOGIC ON THIS NEEDS TO BE IMPROVED SO YOU CAN USE ANY CENTER POINT!
            startVec=linspace(0,centerPoint-2,8*domainEnbiggen).astype(int); 
            startVec=linspace(0,Np//4-2,8*domainEnbiggen).astype(int); 

            #if we are not using param2, make nan
            param1=Ladv; param1name='Ladv'
            param2=Ldiff; param2name='Ldiff'

            #define domain and duration
            #now define parameters for computing backwards in time paths
            nPaths=1200 #number of paths to create
            nPatch=5   #population of each habitat patch

            #number of generations to run
            if param1name=='nGens':
                nGens=param1
            else:
                if Ladv==0:
                    nGens=12000*3 #ok for Ladv=0, Ldiff=5
                else:
                    nGens=6000*2


            #make dispersal matrices
            E,Eback,EbackCDF=makeConnectivityMatrices(Np,param1,param2,param1name,param2name)

            #============================================================================================================
            #ok, now compute the backwards in time for some set of pairs

            #make list of pairs in tuple list of tuples (point1,point2,'label')
            if True:
                startPoints=[]
                for pnt in startVec:
                    startPoints.append((centerPoint-pnt,centerPoint+pnt,'start at %d'%pnt))

            #make pairs to analyze and make arguements (in argVec) for
            #function that analyzes location of coalescense
            argVec=[]
            nRun=0
            for n1 in range(len(startPoints)):
                point1=startPoints[n1][0]
                point2=startPoints[n1][1]
                nRun+=1
                argVec.append((point1,point2,nPatch,nGens,nPaths,EbackCDF,nRun))

            print('THE NUMBER OF RUNS WILL BE',nRun,'for',param1name,'set to',param1)
            #assert False,'as'

            #run analysis
            tic=time.time()
            print('starting with coalescent analysis')
            if False: #serial for debugging. Warning, will not work with rest of code below
                for n in [0]:
                    outputTuple=twoTrackCheck(*argVec[n])
                    outputTuple=(outputTuple,)
            else:
                #parallel run
                nProcs=mp.cpu_count()
                with mp.Pool(nProcs,maxtasksperchild=10) as pool:
                    output=pool.starmap(twoTrackCheck,argVec) 

            #now gather answers
            fracCoalescVec=zeros(len(startPoints))+nan
            meanTimeVec=zeros(len(startPoints))+nan
            medianTimeVec=zeros(len(startPoints))+nan

            for n in range(len(startPoints)):
                fracCoalescVec[n]=output[n][0]
                meanTimeVec[n]=output[n][2]
                medianTimeVec[n]=output[n][3]


            #now plot
            figure(1,figsize=(8.5,8.5)); style.use('ggplot'); clf()

            subplot(2,1,1)
            plot(startVec,fracCoalescVec,'r*-')
            xlabel(scaleName)
            ylabel('mean time to MRCA')
            title('fraction coalesce')

            subplot(2,1,2)
            plot(startVec,meanTimeVec,'r*-')
            xlabel(scaleName)
            ylabel('mean time to MRCA')
            title('fraction coalesce')

            draw()
            ion()
            show()
            pause(0.2)
            savefig('MRCA_pngShed/'+runName+'.png',dpi=300)
            print('done with',runName)


            #save output each parameter iterations. This is done so you can plot the output partway through the
            #run
            if True:
                #save output
                dictToSave={'output':output,'startPoints':startPoints,'startVec':startVec,'param1name':param1name,
                            'param2name':param2name,
                            'scaleName':scaleName,'nPatch':nPatch,'Np':Np}
                with open('MRCA_pickleShed/'+runName+'.pickle', 'wb') as handle:
                    pickle.dump(dictToSave, handle, protocol=pickle.HIGHEST_PROTOCOL)


            # if False:
            #     #save plots
            #     figure(3); savefig('figWhere_coalescent.png',dpi=300)
            #     figure(1); savefig('figWhere_coalescent_details.png',dpi=300)
            #     figure(2); savefig('figWhere_coalescent_details_scaled.png',dpi=300)
