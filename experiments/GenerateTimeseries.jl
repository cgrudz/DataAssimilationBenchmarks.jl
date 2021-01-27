########################################################################################################################
module GenerateTimeseries 
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
export l96_timeseries

########################################################################################################################
# generate timeseries based on the model, solver and parameters

function l96_timeseries(seed::Int64, state_dim::Int64, tanl::Float64, diffusion::Float64)
    
    # define the model
    dx_dt = L96.dx_dt

    # define the integration scheme
    if diffusion == 0.0
        # generate the observations with the runge-kutta scheme
        forward_step = DeSolvers.rk4_step!

        # parameters for the runge-kutta scheme
        h = 0.01

    else
        # generate the observations with the strong taylor scheme
        forward_step = L96.l96s_tay2_step!
        
        # parameters for the order 2.0 strong taylor scheme
        h = 0.005
        p = 1
        α = L96.α(p)
        ρ = L96.ρ(p)
    end

    # define model run parameters
    nanl = 50000
    spin = 5000
    F = 8.0

    # generate some initial ensemble including parameter values
    f_steps = convert(Int64, tanl/h)

    # set storage for the ensemble timeseries
    obs = Array{Float64}(undef, state_dim, nanl)

    # define the integration parameters in the kwargs dict
    kwargs = Dict{String, Any}(
              "h" => h,
              "diffusion" => diffusion,
              "dx_params" => [F],
              "dx_dt" => dx_dt,
             )
    if diffusion != 0.0
        kwargs["p"] = p
        kwargs["α"] = α
        kwargs["ρ"] = ρ
    end

    # seed the random generator
    μ = zeros(state_dim)
    σ = 1.0
    Random.seed!(seed)
    x_t = rand(MvNormal(μ,σ))

    # spin the model onto the attractor
    for j in 1:spin
        for k in 1:f_steps
            x_t = forward_step(x_t, kwargs, 0.0)
        end
    end

    # save the model state at timesteps of tanl
    for j in 1:nanl
        for k in 1:f_steps
            x_t = forward_step(x_t, kwargs)
        end
        obs[:, j] = x_t
    end
    
    data = Dict{String, Any}(
                "h" => h,
                "diffusion" => diffusion,
                "F" => F,
                "tanl" => tanl,
                "nanl" => nanl,
                "spin" => spin,
                "state_dim" => state_dim,
                "obs" => obs
               )

    name = "l96_timeseries_seed_" * lpad(seed, 4, "0") * "_dim_" * lpad(state_dim, 2, "0") * "_diff_" * rpad(diffusion, 4, "0") * 
           "_tanl_" * rpad(tanl, 4, "0") * "_nanl_" * lpad(nanl, 5, "0") * "_spin_" * lpad(spin, 4, "0") * "_h_" * rpad(h, 5, "0") * ".jld"
    path = "../data/timeseries/"
    save(path * name, data)

end

########################################################################################################################
# end module

end
