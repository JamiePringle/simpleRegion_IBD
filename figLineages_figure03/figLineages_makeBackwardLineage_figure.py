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

#this code calculates a connectivity matrix E for a 1D domain Np in
#size and calculates a large number of backwards trajectories that are
#used in a subsequent code to calculate the time to MRCA and fraction
#of coalescence for this idealized domain.

#the units of distance are the size of each individual patch, and
#all lengthscales should be larger than the patch. 

#size of domain Np
Np=512

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

    if False:
        assert param1name=='nGens','param change missmatch' 
        Ladv=0.0
        Ldiff=5.0
    elif False:
        Ladv=param1; assert param1name=='Ladv','param change missmatch' #positive means goes to bigger N
        Ldiff=5.0*1
    elif False:
        assert param1name=='time','param change missmatch'
        Ladv=param1*1.0
        Ldiff=sqrt(param1)*5.0
    else:
        Ladv=param1; assert param1name=='Ladv','param change missmatch' #positive means goes to bigger N
        Ldiff=param2; assert param2name=='Ldiff','param change missmatch' 

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
    print('CJMP',paths.shape)
    return(paths)
    
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

    #define domain and duration
    #now define parameters for computing backwards in time paths
    nPaths=1200*10 #number of paths to create
    nGens=300
    nGens=201

    #make dispersal matrices for two sets of parameters

    movieName='compareParams'
    param1_1=0.0; param1name1='Ladv'
    param2_1=5.0; param2name1='Ldiff' #look at Kernel code to see what these params do
        
    param1_2=2.0; param1name2='Ladv'
    param2_2=5.0; param2name2='Ldiff' #look at Kernel code to see what these params do

    print('staring making connectivity')
    E1,Eback1,EbackCDF1=makeConnectivityMatrices(Np,param1_1,param2_1,param1name1,param2name1)
    E2,Eback2,EbackCDF2=makeConnectivityMatrices(Np,param1_2,param2_2,param1name2,param2name2)
    print('   done')
    
    #==============================================================================================================
    #ok, now compute the backwards in time coaslescents

    #first, make list of points to start points at.
    #there will be (n+1)*n/2 total runs
    points2list1=[Np//2] #start close together
    #points2list1=[Np//2-200,Np//2+200] #start far appart together

    if False:
        #make different points2list for two runs
        points2list1=[Np//2-25,Np//2+25]
        points2list2=[Np//2-200,Np//2+200]
    else:
        points2list2=points2list1.copy()

    argVec1=[]
    argVec2=[]
    for n in range(len(points2list1)):
        argVec1.append((EbackCDF1,zeros((nPaths,))+points2list1[n],nGens))
        argVec2.append((EbackCDF2,zeros((nPaths,))+points2list2[n],nGens))
        
    with mp.Pool() as pool:
        print('makeing paths')
        output1=pool.starmap(backPaths,argVec1)
        output2=pool.starmap(backPaths,argVec2)
    print('   done making paths')

    #==============================================================================================================
    #animate output
    figure(1,figsize=(12,8))
    style.use('ggplot')
    subplots_adjust(left=0.1,bottom=0.05,
                    right=0.95,top=0.95,
                    wspace=0.1,hspace=0.1)

    def makePDFs(thisPop1):
        bandwidth=4.0/2
        kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(thisPop1)
        #
        xvec=arange(Np)
        thisSmooth1=exp(kde1.score_samples(xvec.reshape(-1, 1)))
        #
        return thisSmooth1

    nPlots=0
    clf()
    for nFrame in range(nGens):
        thisPop1=output1[0][:,nFrame].reshape((nPaths,1)) #KernelDensity expects 2D array
        thisSmooth1=makePDFs(thisPop1)

        thisPop1=output2[0][:,nFrame].reshape((nPaths,1)) #KernelDensity expects 2D array
        thisSmooth2=makePDFs(thisPop1)

        #if True, plot every frame, if False, make paper plot
        if False:
            clf()
            plot(thisSmooth1,'r--')
            plot(thisSmooth2,'r-')
            plot(points2list1,0*array(points2list1),'r*',markersize=10)
            ylabel('population PDF')
            title('Generation %d computed from %d individuals'%(nFrame,nPaths),fontsize='medium')
            axis(xmin=0.0,xmax=Np-1,ymin=-0.005,ymax=0.065)
        else:
            frames2plot=[0,50,100,150,200]

            if nFrame in frames2plot:
                nPlots+=1
                subplot(len(frames2plot),2,2*(nPlots-1)+2)
                plot(thisSmooth1,'r--')
                plot(thisSmooth2,'r-')
                plot(points2list1,0*array(points2list1),'r*',markersize=10)
                #ylabel('gen %d'%(nFrame))
                axis(xmin=0.0,xmax=Np-1,ymin=-0.005,ymax=0.065)

                if nFrame==0:
                    title('backwards in time')

                jnk=gca()
                jnk.set_yticklabels(' ')
                if nFrame!=200:
                    jnk.set_xticklabels(' ')
                


            # if nPlots==len(frames2plot):
            #     suptitle('Generation %d computed from %d individuals'%(nFrame,nPaths),fontsize='medium')
            #     suptitle('Left, forward in time, Right, backwards in time',fontsize='medium')


        


        draw()
        show()
        pause(0.01)
        
        #=====================================================================================


