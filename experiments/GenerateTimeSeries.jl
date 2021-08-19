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

function l96_time_series(args::Tuple{Int64,Int64,Float64,Int64,Int64,Float64,Float64})

    # time the experiment
    t1 = time()

    # unpack the experiment parameters determining the time series
    seed, state_dim, tanl, nanl, spin, diffusion, F = args
    
    # define the model
    dx_dt = L96.dx_dt

    # define the integration scheme
    if diffusion == 0.0
        # generate the observations with the Runge-Kutta scheme
        step_model! = DeSolvers.rk4_step!

        # parameters for the Runge-Kutta scheme
        h = 0.01

    else
        # generate the observations with the strong Taylor scheme
        step_model! = L96.l96s_tay2_step!
        
        # parameters for the order 2.0 strong Taylor scheme
        h = 0.005
        p = 1
        α = L96.α(p)
        ρ = L96.ρ(p)
    end

    # set the number of discrete integrations steps between each observation time
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
    x = rand(MvNormal(μ,σ))

    # spin the model onto the attractor
    for j in 1:spin
        for k in 1:f_steps
            step_model!(x, 0.0, kwargs)
        end
    end

    # save the model state at timesteps of tanl
    for j in 1:nanl
        for k in 1:f_steps
            step_model!(x, 0.0, kwargs)
        end
        obs[:, j] = x
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
           "_F_" * lpad(F, 4, "0") * 
           "_tanl_" * rpad(tanl, 4, "0") * 
           "_nanl_" * lpad(nanl, 5, "0") * 
           "_spin_" * lpad(spin, 4, "0") * 
           "_h_" * rpad(h, 5, "0") * 
           ".jld"
    path = "../data/time_series/"
    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end

########################################################################################################################

end
