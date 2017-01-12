#target 
#number of arms is n
@everywhere begin
	using StatsBase
	import Distributions.Bernoulli
end
@everywhere function Bandit_tune(xxx)

	@everywhere begin
		using StatsBase
		using Distributions
		using PyPlot
	end
	@everywhere begin
		yyy = 0.5
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
		sigma = 0.01*ones(n)
	end


	@everywhere NN = 500000
	@everywhere num_iter = 16*NN
	@everywhere boost = true



	@everywhere function boost_func(i,k,xxx,yyy) 
		return (log(i+1)^xxx)/(nn[k])^yyy
	end

	@everywhere function bandit(num_iter::Int64, boost::Bool,boost_func,q,nn,Y,sigma,xxx,yyy)
		R = Float64[] 
		for i=1:num_iter
			k = sample(1:n,WeightVec(q))
			nn[k] += 1
			boost ? sigma[k] = boost_func(i,k,xxx,yyy) : sigma = zeros(n)
			w = rand(Bernoulli(0.01))*k[1]
			Y[k] = (Y[k]*(nn[k]-1) + w)/nn[k] 
			q = (Y+sigma)./sum(Y+sigma)	
			zz = collect(1:n)/Z
			w == 0? push!(R,0) : push!(R,sum(zz.*log(zz./q)))
		   # w == 0? push!(R,0) : push!(R,(w/(Z*q[k])*log(w/(q[k]*Z)) - w/(q[k]*Z)) +1)

		end
		return(q, R[2:end])
	end

	@everywhere function Bandit(xxx)
		return bandit(num_iter,true,boost_func,q,nn,Y,sigma,xxx,yyy)
	end
	
	res = pmap(Bandit, ones(1:10)*xxx)
	
	R_mat = Array(Float64, num_iter-1,6)
	for i=1:6
		R_mat[:,i] = res[i][2]
	end
	
	return(mean(R_mat,2))
end

xxx=linspace(0.01,1,6)
out_x = pmap(Bandit_tune, xxx)
	
#save("Documents/ANIS/regret_x.jld","out_x",out_x)

using PyPlot
for i=1:6
	xxxx = floor(xxx[i],4)
	plot(cumsum(out_x[i]),label="log(i^$xxxx)/sqrt(n)")
end
legend(loc="upper left")


