#using Pkg
#Pkg.activate("C:/Users/yaoya/NODE")
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, CUDA, DiffEqSensitivity
using JLD
using MAT
# Makes sure no slow operations are occuring
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
ex = ex
ode_data = Float32.(hammerstein_system(ex)) #signal we want to predict
plot(ode_data)
#Define the ode layer
nn_dudt = FastChain(
                  FastDense(2, 8, tanh),
                  FastDense(8, 1))
u0 = Float32[0.0]
p = initial_params(nn_dudt)
#dudt2_(u, p, t) = dudt2(u,p)
"""
function dudt2(u,p,t,ex)
  nn_dudt(vcat(u,ex[Int(round(t*10))]), p)
end

_dudt2(u,p,t) = dudt2(u,p,t,ex)
prob_gpu = ODEProblem(_dudt2, u0, tspan, nothing)
"""
function dudt3(u,p,t)
  nn_dudt(vcat(u,ex[Int(round(t*10))]), p)
end

prob_gpu = ODEProblem(dudt3, u0, tspan, nothing)

# Runs on a GPU
function predict_neuralode(p)
  _prob_gpu = remake(prob_gpu,p=p)
  Array(solve(_prob_gpu, Tsit5(), saveat = tsteps, abstol = 1e-8, reltol = 1e-6))
end

function loss_neuralode(p)
    pred =predict_neuralode(p)
    N = length(pred)
    return sum(abs2, ode_data[1:N]' .- pred)/N

end

"""
function loss_neuralode(p)
    sol = predict_neuralode(p)
    N = length(sol)
    l = 0.0
    for i = 1:N
        l += (y[i]-sol[i]).^2
    end
    return l
end
"""
res0 = DiffEqFlux.sciml_train(loss_neuralode,p ,ADAM(0.01), maxiters=100)

res1 = DiffEqFlux.sciml_train(loss_neuralode,res0.minimizer,ADAM(0.01), maxiters=300)


sol = predict_neuralode(res1.minimizer)

plot(sol')
plot!(ode_data)
