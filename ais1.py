 
 
import matplotlib.pyplot as plt
import numpy as np
import time

N=1000
X = np.zeros(N); D = np.zeros(N); R = np.zeros(N)
X[0]=0.1
for t in range(1,1000):
  X[t] = X[t-1] + 0.1 + np.random.normal(0,0.1,1)
  D[t] = X[t] - X[t-1]
  R[t] = (X[t] - X[t-1])/X[t-1]
    
      
plt.figure()
plt.hist(D,alpha = 0.5,bins=30)
plt.hist(R,alpha = 0.5,bins=30)


rng = np.random.RandomState(10)  
a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.show()



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

      total_log[0,0] =  total_log[0,0] + v* np.log(v)
      total_log2[0,0] =  total_log2[0,0] + (v* np.log(v))**2
      nn[0,0] = nn[0,0] + 1

