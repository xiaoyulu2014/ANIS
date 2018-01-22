import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import matplotlib.cm as cm
import time

T = 10000
def truep(x):
    return  np.exp(-0.5*(0.03*x[0]**2+(x[1]+0.03*(x[0]**2-100))**2))
  
explorefactor = 0.5
threshold = 100
M = 20


L_max = 1000
total = np.zeros((L_max,L_max))        #total is an estimate of Z: sum(p/u)
total_log = np.zeros((L_max,L_max))    #total_log is sum(p/u * log(p/u))
total_log2 = np.zeros((L_max,L_max))        #total is an estimate of Z: sum((p/u * log(p/u))^2)

nn = np.zeros((L_max,L_max))
dd = np.ones((L_max,L_max))   #indicator of whether a parent/child node, initially all are parent nodes
dd[0,0] = 1
mm = np.zeros((L_max,L_max))

timer1 = np.zeros(T)
timer2 = np.zeros(T)
xx = np.zeros((2,T))
pp = np.zeros((T))
maxlevel = 10
RR = np.zeros((1,T))
q = np.zeros((L_max,L_max))
q[0,0] = 1;



for tt in range(T):
    ii=[int(0),int(0)]
    ll=[0,0]
    ss=[-M,-M]

    while(1):
        #sample along the tree
        ind = 1- (ll[0] == ll[1])
        if (nn[ii[0],ii[1]]<threshold or ii[ind]>(2**maxlevel) or dd[ii[0],ii[1]] == 0):
            
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
                total_log2[ii[0],ii[1]] = total_log2[ii[0],ii[1]] + (v* np.log(v))**2
                nn[ii[0],ii[1]] = nn[ii[0],ii[1]] + 1
              
                ind = 1*(kk[0]==kk[1])
                ii[ind] = np.floor((ii[ind]-1)/2)
                kk[ind] = kk[ind]-1

            total[0,0] =  total[0,0] +  v
          
            total_log[0,0] =  total_log[0,0] + v* np.log(v)
            total_log2[0,0] =  total_log2[0,0] + (v* np.log(v))**2
            nn[0,0] = nn[0,0] + 1
            
            break

        else:
            #dd[ii[0],ii[1]] = 0
            cc = [2*ii[ind]+1, 2*ii[ind]+2]
            ll[ind] = ll[ind] + 1

            start = time.time()
            if ind==0:
                mm[cc,ii[1]] = total[cc,ii[1]]/nn[cc,ii[1]]
                exploreboost = explorefactor*np.sqrt(np.log(nn[ii[0],ii[1]]+1))/nn[cc,ii[1]]
                rr = mm[cc,ii[1]] + exploreboost
                q[cc,ii[1]] = rr/sum(rr)*q[ii[0],ii[1]]

            else:
                mm[ii[0],cc] = total[ii[0],cc]/nn[ii[0],cc]
                exploreboost = explorefactor*np.sqrt(np.log(nn[ii[0],ii[1]]+1))/nn[ii[0],cc]
                rr = mm[ii[0],cc] + exploreboost
                q[ii[0],cc] = rr/sum(rr)*q[ii[0],ii[1]]

            jj = float((np.random.uniform(0,1,1)< rr[1]/sum(rr))[0])
            timer2[tt] = time.time()-start
            
            #turn child node to parent if KL is small enough
           # KL1 = (total_log[ii[0],ii[1]] - total[ii[0],ii[1]]*np.log(q[ii[0],ii[1]]))/total[ii[0],ii[1]]
           # if ind==0:
           #     KL2 = sum(0.5*(total_log[cc,ii[1]] - total_log[cc,ii[1]]*np.log(q[cc,ii[1]])))/total[ii[0],ii[1]]
           # else:
           #     KL2 = sum(0.5*(total_log[ii[0],cc] - total_log[ii[0],cc]*np.log(q[ii[0],cc])))/total[ii[0],ii[1]]
            
           # if np.log(np.mean(timer1)) + KL1 < 0.9*(np.log(np.mean(timer1+timer2)) + KL2):
           #     dd[ii[0],ii[1]] = 0
           #     print(ii)
                
            if ind==0:    
                w_mean = -(0.5*total_log[ii[0],ii[1]]/total[ii[0],ii[1]] + np.sum((nn[cc,ii[1]]*np.log(rr/sum(rr)))/total[ii[0],ii[1]]))
            else:
                w_mean = -(0.5*total_log[ii[0],ii[1]]/total[ii[0],ii[1]] + np.sum((nn[ii[0],cc]*np.log(rr/sum(rr)))/total[ii[0],ii[1]]))
            
            var = total_log2[ii[0],ii[1]]/nn[ii[0],ii[1]] - (total_log[ii[0],ii[1]]/nn[ii[0],ii[1]])**2
            if (w_mean-np.log(np.mean(timer1)/np.mean(timer1+timer2)))/np.sqrt(var) < -1.96:
                dd[ii[0],ii[1]] = 0
                print(ii)
            
            ii[ind] = 2*ii[ind] + jj + 1
            ss[ind] = ss[ind] + jj/(2**ll[ind])*2*M
            

            


    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim((-M,M))
ax.set_ylim((-M,M))
Index = np.matrix.nonzero(q)

for ind in range(len(Index[0])):
    i = Index[0][ind]
    j = Index[1][ind]
    if (dd[i,j] == 1) & (nn[i,j] >0):
        ll_i = np.floor(np.log2(i+1))
        ll_j = np.floor(np.log2(j+1))
        ss_i = (i+1-2**ll_i)/2**ll_i*(2*M)-M
        ss_j = (j+1-2**ll_j)/2**ll_j*(2*M)-M
        ax.add_patch(
         patches.Rectangle((ss_i, ss_j),  2*M/(2**ll_i),   2*M/(2**ll_j) ,edgecolor="blue", facecolor = cm.Blues(q[i,j]/0.02)))

                
    
