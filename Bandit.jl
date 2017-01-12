#target 
#number of arms is n
using StatsBase
n = 100
pi(x) = round(x*n) * (0<=x<=1) 
Z = (1+n)/2  #true normalising constant
xx=(1/n):(1/n):1
yy=[pi(xx[i]) for i=1:n]



#initial proposal
q = ones(n)/n
nn = zeros(n)
Y0= 1
Y = Y0*ones(n)
R = Float64[] #regret
sigma = 0.01*ones(n)


NN = 60
num_iter = 16*NN
boost = true


using PyPlot

fig, axs = plt[:subplots](4,4, figsize=(15, 6)); suptitle("Every $(NN) iterations, Y0=$(Y0), boost=0.001*log(i)/nn[k]");


R = Float64[] #regret
for i=1:num_iter
	k = sample(1:n,WeightVec(q))
	nn[k] += 1
	boost ? sigma[k] = 0.0001*log(i)/nn[k]  : sigma = zeros(n)
	w = pi(k/n)/n
	Y[k] = (Y[k]*(nn[k]-1) + w^2)/nn[k] 
	q = sqrt(Y+sigma)./sum(sqrt(Y+sigma))
	push!(R, (w/q[k]-5050)^2)
	
	if (rem(i,NN) == 1) && (i < num_iter-NN)
		ii=convert(Int64,(i-1)/NN+1)
		#axs[ii][:hist](sigma,50,color="blue")
		axs[ii][:plot](xx,q*n,label="proposal")
		axs[ii][:plot](xx,yy/Z,label="target")
		axs[ii][:set_xlabel]("x")
		axs[ii][:legend](loc="top")
	end
end
axs[16][:plot](1:num_iter,R,label="loss")
axs[16][:set_xlabel]("iteration")
axs[16][:set_yscale]("log")
title("L2 loss")
fig[:canvas][:draw]() # Update the figure




