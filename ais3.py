# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 17:43:02 2017

@author: O679510
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.cm as cm
import time
import copy
from matplotlib.collections import PatchCollection
import matplotlib.colorbar as cbar



T = 1000
def truep(x):
    return  np.exp(-0.5*(0.03*x[0]**2+(x[1]+0.03*(x[0]**2-100))**2))
  
explorefactor = 3
threshold = 10
M = 20


L_max = 1000
total = np.zeros((L_max,L_max))        #total is an estimate of Z: sum(p/u)
total_log = np.zeros((L_max,L_max))    #total_log is sum(p/u * log(p/u))
w2 = np.zeros((L_max,L_max))        #total is an estimate of Z: sum((p/u * log(p/u))^2)

nn = np.zeros((L_max,L_max))
dd = np.zeros((L_max,L_max)) #np.ones  #indicator of whether a parent/child node, initially all are parent nodes
dd[0,0] = 0 #1
mm = np.zeros((L_max,L_max))

timer1 = np.zeros(T)
timer2 = np.zeros(T)
xx = np.zeros((2,T))
pp = np.zeros((T))
RR = np.zeros((1,T))
q = np.zeros((L_max,L_max))

q[0,0] = 1;



for tt in range(T):
    ii=[int(0),int(0)]
    ll=[0,0]
    ss=[-M,-M]
    
    ii_parent = [int(0),int(0)]
    
    while(1):
        #sample along the tree
        ind = 1- (ll[0] == ll[1])
        
        if dd[ii[0],ii[1]] == 1:
            cc = [2*ii[ind]+1, 2*ii[ind]+2]
            ll[ind] = ll[ind] + 1    
            
            if ind==0:
                mm[cc,ii[1]] = total[cc,ii[1]]/nn[cc,ii[1]]
                exploreboost = explorefactor*np.sqrt(np.log(nn[ii[0],ii[1]]+1)/nn[cc,ii[1]])
                rr = mm[cc,ii[1]] + exploreboost
                q[cc,ii[1]] = rr/sum(rr)*q[ii[0],ii[1]]
            
            else:
                mm[ii[0],cc] = total[ii[0],cc]/nn[ii[0],cc]
                exploreboost = explorefactor*np.sqrt(np.log(nn[ii[0],ii[1]]+1)/nn[ii[0],cc])
                rr = mm[ii[0],cc] + exploreboost
                q[ii[0],cc] = rr/sum(rr)*q[ii[0],ii[1]]
            
            jj = 1 - float((np.random.uniform(0,1,1)< (rr/sum(rr))[0])) 
            
            ii[ind] = 2*ii[ind] + jj + 1
            ss[ind] = ss[ind] + jj/(2**ll[ind])*2*M
        
        
        elif (dd[ii[0],ii[1]] == 0):  #nn[ii[0],ii[1]]<threshold or (ii[ind]>(2**maxlevel)) 
            ii_parent = copy.deepcopy(ii)
            cc = [2*ii_parent[ind]+1, 2*ii_parent[ind]+2]
            start = time.time()
            rr = np.random.uniform(0,1,2)
            xx[:,tt] = ss + 2*M*rr/(np.power(2,ll))
            timer1[tt] = time.time()-start
            pp[tt] = truep(xx[:,tt])
            
            ii[ind] = int(2*ii[ind] + (rr[ind] > 0.5) + 1)
            ll[ind] = int(ll[ind] + 1)
            #update statisitcs
            kk = ll
            while (np.max(kk) > 0) & (np.min(kk)>=0):
                ii = [int(ii[0]),int(ii[1])]
                v = pp[tt]*(2*M/2**kk[0])*(2*M/2**kk[1])
                total[ii[0],ii[1]] = total[ii[0],ii[1]] +  v
                
                total_log[ii[0],ii[1]] = total_log[ii[0],ii[1]] + v* np.log(v)
                nn[ii[0],ii[1]] = nn[ii[0],ii[1]] + 1
                
                index = 1*(kk[0]==kk[1])
                ii[index] = np.floor((ii[index]-1)/2)
                kk[index] = kk[index]-1
            
            total[0,0] =  total[0,0] +  v
            total_log[0,0] =  total_log[0,0] + v* np.log(v)
            nn[0,0] = nn[0,0] + 1
            
            start = time.time()
            if (ind==0) & (np.prod(nn[cc,ii_parent[1]]) == 0):
                break
            elif (ind==1) & (np.prod(nn[ii_parent[0],cc]) == 0):
                break
            elif (ind==0) & (np.prod(nn[cc,ii_parent[1]])>0):
                mm[cc,ii_parent[1]] = total[cc,ii_parent[1]]/nn[cc,ii_parent[1]]
                exploreboost = explorefactor*np.sqrt(np.log(nn[ii_parent[0],ii_parent[1]]+1)/nn[cc,ii_parent[1]])
                rr = mm[cc,ii_parent[1]] + exploreboost
                q[cc,ii_parent[1]] = rr/sum(rr)*q[ii_parent[0],ii_parent[1]]
            
            elif (ind==1) & (np.prod(nn[ii_parent[0],cc])>0):
                mm[ii_parent[0],cc] = total[ii_parent[0],cc]/nn[ii_parent[0],cc]
                exploreboost = explorefactor*np.sqrt(np.log(nn[ii_parent[0],ii_parent[1]]+1)/nn[ii_parent[0],cc])
                rr = mm[ii_parent[0],cc] + exploreboost
                q[ii_parent[0],cc] = rr/sum(rr)*q[ii_parent[0],ii_parent[1]]
            
            jj = 1 - float((np.random.uniform(0,1,1)< (rr/sum(rr))[0])) 
            timer2[tt] = time.time()-start
            
            #turn child node to parent if KL is small enough
            KL1 = total_log[ii_parent[0],ii_parent[1]]/total[ii_parent[0],ii_parent[1]]-np.log(q[ii_parent[0],ii_parent[1]])
            if ind==0:
                KL2 = 0.5/total[ii_parent[0],ii_parent[1]]*(total_log[ii_parent[0],ii_parent[1]]-sum(np.log(q[cc,ii_parent[1]])*total[cc,ii_parent[1]]))      
            else:
                KL2 = 0.5/total[ii_parent[0],ii_parent[1]]*(total_log[ii_parent[0],ii_parent[1]]-sum(np.log(q[ii_parent[0],cc])*total[ii_parent[0],cc]))
            
            if (np.log(np.mean(timer1[:tt])) + KL1) > (np.log(np.mean(timer1[:tt]+timer2[:tt])) + KL2):
                dd[ii_parent[0],ii_parent[1]] = 1
            elif (ind==0) & (np.prod(nn[cc,ii_parent[1]])>0):
                q[cc,ii_parent[1]] = [0,0]
            elif (ind==1) & (np.prod(nn[ii_parent[0],cc])>0):
                q[ii_parent[0],cc] = [0,0]
            break


Index = np.matrix.nonzero(q)
qqq=[]
for id in range(len(Index[0])):
    i = Index[0][id]
    j = Index[1][id]
    if (dd[i,j] == 0) & (nn[i,j] >0) & ([i*j >0]):
        qqq.append(q[i,j])
      
normal = plt.normalize(min(qqq), max(qqq))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim((-M,M))
ax.set_ylim((-M,M))
ax.set_title("proposal dsitribution")
Index = np.matrix.nonzero(q)
qqq=[]
for id in range(len(Index[0])):
    i = Index[0][id]
    j = Index[1][id]
    if (dd[i,j] == 0) & (nn[i,j] >0) & ([i*j >0]):
        qqq.append(q[i,j])
        ll_i = np.floor(np.log2(i+1))
        ll_j = np.floor(np.log2(j+1))
       # if ((ll_i==ll_j) & (nn[2*i+1,j]*nn[2*i+2,j]>0)) or ((ll_i > ll_j) & (nn[i,2*j+1]*nn[i,2*j+2]>0)):
        ss_i = (i+1-2**ll_i)/2**ll_i*(2*M)-M
        ss_j = (j+1-2**ll_j)/2**ll_j*(2*M)-M
        ax.add_patch(
         patches.Rectangle((ss_i, ss_j),  2*M/(2**ll_i),   2*M/(2**ll_j) ,edgecolor="blue", facecolor = cm.Blues(normal(q[i,j]))))
#cax, _ = cbar.make_axes(ax) 
#cb1 = cbar.ColorbarBase(cax, cmap=plt.cm.Blues,norm=normal) 
plt.show()  


plt.figure()
H, xedges, yedges = np.histogram2d(xx[1,:],xx[0,:],bins=50)
plt.title("heatmap of samples")
im = plt.imshow(H, interpolation='nearest',origin='lower')
#plt.colorbar()
plt.show()
