#target 
#number of arms is n
using StatsBase
using Distributions
using PyPlot

n = 100
shape=(1:n)/10
scale = 10*(1:n)

pi(x) = shape[convert(Int64,round(x*n))] * scale[convert(Int64,round(x*n))] * (0<=x<=1) 
xx=(1/n):(1/n):1
yy=[pi(xx[i]) for i=1:length(xx)]
Z = sum(yy/n)

#initial proposal
q = ones(n)/n
nn = zeros(n)
Y0= 0.1
Y = Y0*ones(n)
R = Float64[] #regret
sigma = 0.001*ones(n)


NN = 10000
num_iter = 16*NN
boost = false


fig, axs = plt[:subplots](4,4, figsize=(15, 6)); suptitle("Every $(NN) iterations, Y0=$(Y0), boost=0.001*log(i)/nn[k]");
fig1, axs1 = plt[:subplots](4,4, figsize=(15, 6)); suptitle("Every $(NN) iterations, Y");

arm=Float64[]
weights=Float64[]
R = Float64[] #regret
for i=1:num_iter
	k = sample(1:n,WeightVec(q))
	nn[k] += 1
	boost ? sigma[k] = 10*log(i)/nn[k] : sigma = zeros(n)
	w = rand(Gamma(shape[k],scale[k]),1)[1]/n
	Y[k] = (Y[k]*(nn[k]-1) + w^2)/nn[k] 
	q = sqrt(Y+sigma)./sum(sqrt(Y+sigma))
	push!(R, (w/q[k]-Z)^2)
	push!(arm,k)
	push!(weights,w)
	
	if (rem(i,NN) == 1) 
		ii=convert(Int64,(i-1)/NN+1)
		axs1[ii][:plot](q)  

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




