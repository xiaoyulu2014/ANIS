#target 
#number of arms is n
@everywhere begin
	using StatsBase
	using Distributions
	using PyPlot
end
@everywhere begin
	n = 100
	shape=(1:n)/10
	scale = 10*(1:n)

	pii(x) = x * (0<=x<=1) 
	xx=(1/n):(1/n):1
	yy=[pii(xx[i]) for i=1:length(xx)]
	Z = 5050
	q = ones(n)*(1/n)

	#initial proposal
	nn = zeros(n)
	Y0= 0.1
	Y = Y0*ones(n)
	#sigma = 0.01*ones(n)
	sigma = zeros(n)
end


@everywhere NN = 100000
@everywhere num_iter = 16*NN
@everywhere boost = true



@everywhere function boost_func(i,k) 
	return sqrt(0.1*log(i+1)/nn[k])
end

@everywhere function bandit(num_iter::Int64, boost::Bool,boost_func,q,nn,Y,sigma)
	R = Float64[] 
	for i=1:num_iter
		k = sample(1:n,WeightVec(q))
		nn[k] += 1
		boost ? sigma[k] = boost_func(i,k) : sigma = zeros(n)
		w = rand(Bernoulli(0.01))*k[1]
		Y[k] = (Y[k]*(nn[k]-1) + w)/nn[k] 
		q = (Y+sigma)./sum(Y+sigma)	
		zz = collect(1:n)/Z
	    w == 0? push!(R,0) : push!(R,sum(zz.*log(zz./q)))
	   # w == 0? push!(R,0) : push!(R,(w/(Z*q[k])*log(w/(q[k]*Z)) - w/(q[k]*Z)) +1)

	end
	return(q, R[2:end])
end

@everywhere function Bandit(x)
	return bandit(num_iter,true,boost_func,q,nn,Y,sigma)
end

res = pmap(Bandit, 1:10)



R_mat = Array(Float64, num_iter-1,10)
for i=1:10
	R_mat[:,i] = res[i][2]
end

R4 = mean(R_mat,2)

#save("sample.jld","sample",sample)


q_mat = Array(Float64, 100,10)
for i=1:10
	q_mat[:,i] = res[i][1]
end
qq = mean(q_mat,2)


for i=1:(n-1)
	plot([(i-1)/n,i/n],[yy[i]/0.5,yy[i]/0.5],color="blue")
	plot([(i-1)/n,i/n],[n*qq[i],n*qq[i]],color="red")
end 
i=n
plot([(i-1)/n,i/n],[yy[i]/0.5,yy[i]/0.5],color="blue",label="target")
plot([(i-1)/n,i/n],[n*qq[i],n*qq[i]],color="red",label="proposal")
xlabel("number of iteration",fontsize=20)
ylabel("regret",fontsize=20)
legend(loc="upper left",fontsize=20)
title("target and proposal densities",fontsize=20)
#'##################################################################################################

load("sample.jld")
string=sample["string"]
for j=1:4
	res = sample["$j"]
	R_mat = Array(Float64, num_iter-1,10)
	for i=1:10
		R_mat[:,i] = res[i][2]
	end

	R = mean(R_mat,2)
	plot(cumsum(R),label=string[j])
end
legend(loc="upper left",fontsize=20)
title("accummulated KL regret",fontsize=20)
xlabel("number of iteration",fontsize=20)
ylabel("regret",fontsize=20)








