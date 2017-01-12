#target 
#number of arms is n
using StatsBase
n = 100
pi(x) = (round(x*n) + 10*sin(x*n*5)) * (0<=x<=1) 
xx=(1/n):(1/n/100):1
yy=[pi(xx[i]) for i=1:length(xx)]
Z = n/2 + 2*(1-cos(5*n))/n

#initial proposal
q = ones(n)/n
nn = zeros(n)
Y0= 0.1
Y = Y0*ones(n)
R = Float64[] #regret
sigma = 0.001*ones(n)


NN = 60
num_iter = 16*NN
boost = true


using PyPlot

fig, axs = plt[:subplots](4,4, figsize=(15, 6)); suptitle("Every $(NN) iterations, Y0=$(Y0), boost=0.001*log(i)/nn[k]");
#fig1, axs1 = plt[:subplots](4,4, figsize=(15, 6)); suptitle("Every $(NN) iterations, histogram of Y");

R = Float64[] #regret
for i=1:num_iter
	k = sample(1:n,WeightVec(q))
	x = (rand(1)[1] + k -1)/n
	nn[k] += 1
	boost ? sigma[k] = 0.001*i/nn[k] : sigma = zeros(n)
	w = pi(x)/n
	Y[k] = (Y[k]*(nn[k]-1) + w^2)/nn[k] 
	q = sqrt(Y+sigma)./sum(sqrt(Y+sigma))
	push!(R, (w/q[k]-Z)^2)
	
	if (rem(i,NN) == 1) 
		ii=convert(Int64,(i-1)/NN+1)
		#axs1[ii][:hist](Y)  
		#axs1[ii][:set_xlim]([0,1])
		if (i < num_iter-NN)
			axs[ii][:plot]((1/n):(1/n):1,n*q,label="proposal")
			axs[ii][:plot](xx,yy/Z,label="target")
			axs[ii][:set_xlabel]("x")
			axs[ii][:legend](loc="top")
		end
	end		
end


axs[16][:plot](1:num_iter,R,label="loss")
axs[16][:set_xlabel]("iteration")
axs[16][:set_yscale]("log")
title("L2 loss")
fig[:canvas][:draw]() # Update the figure




