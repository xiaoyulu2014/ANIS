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

	#initial proposal
	nn = zeros(n)
	Y0= 0.1
	Y = Y0*ones(n)
	sigma = 0.001*ones(n)
end


@everywhere NN = 100000
@everywhere num_iter = 16*NN
@everywhere boost = true



@everywhere function boost_func(i,k) 
	return sqrt(i)/nn[k]
end

@everywhere function bandit(num_iter::Int64, boost::Bool,boost_func,q,nn,Y,sigma)
	arm,weights,R = [Float64[] for i=1:3]
	for i=1:num_iter
		k = sample(1:n,WeightVec(q))
		nn[k] += 1
		boost ? sigma[k] = boost_func(i,k) : sigma = zeros(n)
		w = rand(Bernoulli(0.01))*100*k[1]/n
		Y[k] = (Y[k]*(nn[k]-1) + w^2)/nn[k] 
		q = sqrt(Y+sigma)./sum(sqrt(Y+sigma))
		
		push!(R,(w/(q[k]*Z)-1)^2- (w/pii(k[1]/n)/n-1)^2 )
		#push!(R,(pii(k[1]/n)/Z-n*q[k])^2)
		push!(arm,k)
		push!(weights,w)
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


R1 = mean(R_mat,2)

plot(cumsum(R1),label="sqrt(t)/n")
plot(cumsum(R2),label="sqrt(t/n)")
plot(cumsum(R3),label="sqrt(log(t)/n)")
plot(cumsum(R4),label="log(t)/sqrt(n)")
plot(cumsum(R5),label="log(t)/log(n+1)")
plot(cumsum(R6),label="log(t)/n")
legend(loc="upper left")
title("accummulated regrest, average over 10 runs")
xlabel("number of iteration")
ylabel("regret")



regret=["R1" => R1, "R2" => R2, "R3" => R3,"R4" => R4, "R5" => R5, "R6" => R6]
save("Documents/ANIS/regrest.jld","regret",regret)



#=
n = 100
	shape=(1:n)/10
	scale = 10*(1:n)

	pii(x) = x * (0<=x<=1) 
	xx=(1/n):(1/n):1
	yy=[pii(xx[i]) for i=1:length(xx)]
	Z = 5050

	#initial proposal
	q = ones(n)/n
	nn = zeros(n)
	Y0= 0.1
	Y = Y0*ones(n)
	sigma = 0.001*ones(n)

fig, axs = plt[:subplots](4,4, figsize=(15, 6)); suptitle("Every $(NN) iterations, Y0=$(Y0), boost= sqrt(i)/nn[k]");

arm=Float64[]
weights=Float64[]
R = Float64[] #regret
for i=1:num_iter
	k = sample(1:n,WeightVec(q))
	nn[k] += 1
	boost ? sigma[k] = log(i)/sqrt(nn[k]) : sigma = zeros(n)
	w = rand(Bernoulli(0.01))*100*k[1]/n
	Y[k] = (Y[k]*(nn[k]-1) + w^2)/nn[k] 
	q = sqrt(Y+sigma)./sum(sqrt(Y+sigma))
	push!(R,(w/(q[k]*Z)-1)^2- (w/pii(k[1]/n)/n-1)^2 )
	#push!(R,(pii(k[1]/n)/Z-n*q[k])^2)
	push!(arm,k)
	push!(weights,w)
	
	if (rem(i,NN) == 1) 
		ii=convert(Int64,(i-1)/NN+1)

		if (i < num_iter-NN)
			axs[ii][:plot]((1/n):(1/n):1,n*q,label="proposal")
			axs[ii][:plot](xx,yy/0.5,label="target")
			axs[ii][:set_xlabel]("x")
			axs[ii][:legend](loc="top")
		end
	end		
end


axs[16][:plot](1:num_iter,cumsum(R),label="loss")
axs[16][:set_xlabel]("iteration")
axs[16][:set_yscale]("log")
title("L2 loss")
fig[:canvas][:draw]() 

=#
