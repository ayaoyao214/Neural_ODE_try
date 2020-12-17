#using Pkg
#Pkg.activate("C:/Users/yaoya/NODE")
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, CUDA, DiffEqSensitivity
using JLD
using MAT
CUDA.allowscalar(false) # Makes sure no slow operations are occuring

# This code works fine if scalar indexing is allowed but just slow
CUDA.allowscalar(true)

#generating exogenus signal and output signal
tspan = (0.1f0, Float32(10.0))
tsteps = range(tspan[1], tspan[2], length = 100)
t_vec = collect(tsteps)
ex = vec(ones(Float32,length(tsteps), 1))
f(x) = (atan(8.0 * x - 4.0) + atan(4.0)) / (2.0 * atan(4.0))

function hammerstein_system(u)
    y= zeros(size(u))
    for k in 2:length(u)
        y[k] = 0.2 * f(u[k-1]) + 0.8 * y[k-1]
    end
    return y
end

ex = vec([ones(Float32,50,1) 2*ones(Float32,50,1)]) #exogenus signal
ex = ex'
ode_data = gpu(Float32.(hammerstein_system(ex))) #signal we want to predict

#Define the ode layer
nn_dudt = FastChain(FastDense(2, 8, tanh),FastDense(8, 1))
u0 = Float32[0.0]|> gpu
p = initial_params(nn_dudt)|> gpu
#dudt2_(u, p, t) = dudt2(u,p)
ex = gpu(ex)
function dudt2(u,p,t,ex)
  nn_dudt(vcat(u,ex[Int(round(t*10))]), p)
end

_dudt2(u,p,t) = dudt2(u,p,t,ex)
prob_gpu = ODEProblem(_dudt2, u0, tspan, nothing)

# Runs on a GPU
function predict_neuralode(p)
  _prob_gpu = remake(prob_gpu,p=p)
  gpu(solve(_prob_gpu, Tsit5(), saveat = tsteps, abstol = 1e-8, reltol = 1e-6))
end

function loss_neuralode(p)
    pred =predict_neuralode(p)
    N = length(pred)
    l = sum(abs2, ode_data[1:N]' .- pred)/N
    return l, pred
end
res0 = DiffEqFlux.sciml_train(loss_neuralode,p ,ADAM(0.01), maxiters=10)

res1 = DiffEqFlux.sciml_train(loss_neuralode,res0.minimizer,ADAM(0.01), maxiters=20)


sol = predict_neuralode(res0.minimizer)
sol = Array(sol)
plot(sol')
plot!(ode_data')
