########################################################################################################################
module GenerateTimeSeries 
########################################################################################################################
# imports and exports
using Debugger, JLD, Distributed
using Random, Distributions, LinearAlgebra
using DeSolvers, L96 
export l96_time_series

########################################################################################################################
# generate timeseries based on the model, solver and parameters

function l96_time_series(seed::Int64, state_dim::Int64, tanl::Float64, diffusion::Float64)
    
    # define the model
    dx_dt = L96.dx_dt

    # define the integration scheme
    if diffusion == 0.0
        # generate the observations with the runge-kutta scheme
        forward_step! = DeSolvers.rk4_step!

        # parameters for the runge-kutta scheme
        h = 0.01

    else
        # generate the observations with the strong taylor scheme
        forward_step! = L96.l96s_tay2_step!
        
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
            forward_step!(x_t, 0.0, kwargs)
        end
    end

    # save the model state at timesteps of tanl
    for j in 1:nanl
        for k in 1:f_steps
            forward_step!(x_t, 0.0, kwargs)
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

    name = "l96_time_series_seed_" * lpad(seed, 4, "0") * 
           "_dim_" * lpad(state_dim, 2, "0") * 
           "_diff_" * rpad(diffusion, 4, "0") * 
           "_tanl_" * rpad(tanl, 4, "0") * 
           "_nanl_" * lpad(nanl, 5, "0") * 
           "_spin_" * lpad(spin, 4, "0") * 
           "_h_" * rpad(h, 5, "0") * 
           ".jld"
    path = "../data/time_series/"
    save(path * name, data)

end

########################################################################################################################

end
