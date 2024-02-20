from pylab import *
from numpy import *

L=1000.0
Ladv=150.0
Ldiff=50.0
dx=1.0

x=arange(0.0,L+dx,dx)

def backKernel(x,childLoc):
    x=exp(-(x-(childLoc-Ladv))**2/(2*Ldiff**2))
    x=x/(sum(x*dx))
    indx=x<1e-9 #this stops plotting kernel when too small for graphic effect
    x[indx]=nan
    return x


figure(1,figsize=1.8*array((6.5,2.0)))
clf(); style.use('ggplot')

xC=0.75*L; textLoc=0.0; colorStr='r'
plot(x,backKernel(x,xC),colorStr+'-',linewidth=2,label='lineages 1')
plot(xC,textLoc,colorStr+'*',markersize=20,mec='k')
text(xC,textLoc,'   P1 Child',rotation=45,color=colorStr)
xlabel('distance')
ylabel('PDF')

xC=0.15*L; textLoc=0.0; colorStr='c'
plot(x,backKernel(x,xC),colorStr+'-',linewidth=2,label='lineages 1')
plot(xC,textLoc,colorStr+'*',markersize=20,mec='k')
text(xC,textLoc,'   P2 Child',rotation=45,color=colorStr)
xlabel('distance, km')
ylabel('PDF')

arrow(600.0,0.01,Ladv,0.0,length_includes_head=True,color='k',width=2.5e-4,head_length=20,head_width=0.0015)
text(600+0.5*Ladv,0.011,'$L_{adv}$',horizontalalignment='center',fontsize='large')

arrow(600.0-Ldiff,0.004,1.75*Ldiff,0.0,length_includes_head=False,color='k',width=2.5e-4,head_length=20,head_width=0.0015)
arrow(600.0+Ldiff,0.004,-1.75*Ldiff,0.0,length_includes_head=False,color='k',width=2.5e-4,head_length=20,head_width=0.0015)
text(600,0.005,'$L_{diff}$',horizontalalignment='center',fontsize='large')

tight_layout()
draw()
show()
savefig('fig_BackwardsInTimeKernel.png',dpi=300)

