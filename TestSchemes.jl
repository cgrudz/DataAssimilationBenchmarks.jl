########################################################################################################################
module TestSchemes 
########################################################################################################################
# imports and exports
using Random, Distributions
using Debugger
using Distributed
using LinearAlgebra
using JLD
using DeSolvers
using L96 
using DifferentialEquations
export l96_cg_test, l96_de_test, l96s_de_test

########################################################################################################################
# General statements
# define the model
dx_dt = L96.dx_dt


# parameters for the runge-kutta scheme
h = 0.01

# define model run parameters
nanl = 50000
F = 8.0

########################################################################################################################
# generate timeseries based on in-house rk4 

function l96_cg_test(seed::Int64, state_dim::Int64, tanl::Float64, diffusion::Float64)
    

    # set storage for the ensemble timeseries
    obs = Array{Float64}(undef, state_dim, nanl)

    # define the integration parameters in the kwargs dict
    kwargs = Dict{String, Any}(
              "h" => h,
              "diffusion" => diffusion,
              "dx_params" => [F],
              "dx_dt" => dx_dt,
             )
    
    # seed the random generator
    μ = zeros(state_dim)
    σ = 1.0
    Random.seed!(seed)
    x_t = rand(MvNormal(μ,σ))

    # save the model state at timesteps of tanl
    f_steps = convert(Int64, tanl/h)
    forward_step = rk4_step!

    for j in 1:nanl
        for k in 1:f_steps
            x_t = forward_step(x_t, kwargs, 0.0)
        end
        obs[:, j] = x_t
    end
end

########################################################################################################################
# generate timeseries of l96 based on the DifferentialEquations.jl module

function l96_de_test(seed::Int64, state_dim::Int64, tanl::Float64, diffusion::Float64)
    # parameters for the solver
    tsteps = (0.0, tanl)
    dt_max = 0.01
    p = [F]

    # seed the random generator
    μ = zeros(state_dim)
    σ = 1.0
    Random.seed!(seed)
    x_t = rand(MvNormal(μ,σ))

    # set storage for the ensemble timeseries
    obs = Array{Float64}(undef, state_dim, nanl)

    # specify the solver
    @bp
    alg = Tsit5()

    for j in 1:nanl
        prob = ODEProblem(dx_dt, x_t, tsteps, p)
        sol = solve(prob, alg, dtmax=dt_max)
        x_t = sol.u[end]
        obs[:, j] = x_t
    end
end

########################################################################################################################
# generate timeseries of l96s based on the DifferentialEquations.jl module

function l96s_de_test(seed::Int64, state_dim::Int64, tanl::Float64, diffusion::Float64)
    # parameters for the solver
    tsteps = (0.0, tanl)
    dt_max = 0.01
    p = [F]

    # seed the random generator
    μ = zeros(state_dim)
    σ = 1.0
    Random.seed!(seed)
    x_t = rand(MvNormal(μ,σ))

    # set storage for the ensemble timeseries
    obs = Array{Float64}(undef, state_dim, nanl)

    # specify the solver
    alg = SOSRA()
    function g(W, p::Vector{Float64}, t::Float64)
        return ones(state_dim) 
    end

    for j in 1:nanl
        @bp
        prob = SDEProblem(dx_dt, g, x_t, tsteps, p)
        sol = solve(prob, alg, dtmax=dt_max)
        x_t = sol.u[end]
        obs[:, j] = x_t
    end
end

########################################################################################################################
# end module

end
