##############################################################################################
module GenerateTimeSeries 
##############################################################################################
# imports and exports
using Random, Distributions
using LinearAlgebra
using JLD2, HDF5
using ..DataAssimilationBenchmarks, ..DeSolvers, ..L96, ..IEEE39bus
export L96_time_series, IEEE39bus_time_series
##############################################################################################
"""
    L96_time_series((seed::Int64, h::Float64, state_dim::Int64, tanl::Float64, nanl::Int64,
                     spin::Int64, diffusion::Float64, F::Float64)::NamedTuple)

Simulate a "free run" time series of the [Lorenz-96 model](@ref) model 
for generating an observation process and truth twin for data assimilation twin experiments.
Output from the experiment is saved in a dictionary of the form,

    Dict{String, Any}(
                      "seed" => seed,
                      "h" => h,
                      "diffusion" => diffusion, 
                      "dx_params" => dx_params, 
                      "tanl" => tanl,
                      "nanl" => nanl,
                      "spin" => spin,
                      "state_dim" => state_dim,
                      "obs" => obs,
                      "model" => "L96"
                     )

Where the file name is written dynamically according to the selected parameters as follows:

    "L96_time_series_seed_" * lpad(seed, 4, "0") * 
    "_dim_" * lpad(state_dim, 2, "0") * 
    "_diff_" * rpad(diffusion, 5, "0") * 
    "_F_" * lpad(F, 4, "0") * 
    "_tanl_" * rpad(tanl, 4, "0") * 
    "_nanl_" * lpad(nanl, 5, "0") * 
    "_spin_" * lpad(spin, 4, "0") * 
    "_h_" * rpad(h, 5, "0") * 
    ".jld2"
"""
function L96_time_series((seed, h, state_dim, tanl, nanl, spin, diffusion, F)::NamedTuple{ 
                        (:seed,:h,:state_dim,:tanl,:nanl,:spin,:diffusion,:F),
                        <:Tuple{Int64,Float64,Int64,Float64,Int64,Int64,
                                Float64,Float64}})
    # time the experiment
    t1 = time()
    
    # define the model
    dx_dt = L96.dx_dt
    dx_params = Dict{String, Array{Float64}}("F" => [8.0])

    # define the integration scheme
    if diffusion == 0.0
        # generate the observations with the Runge-Kutta scheme
        step_model! = DeSolvers.rk4_step!

    else
        # generate the observations with the strong Taylor scheme
        step_model! = L96.l96s_tay2_step!
        
        # parameters for the order 2.0 strong Taylor scheme
        p = 1
        α, ρ = comput_α_ρ(p)
    end

    # set the number of discrete integrations steps between each observation time
    f_steps = convert(Int64, tanl/h)

    # set storage for the ensemble timeseries
    obs = Array{Float64}(undef, state_dim, nanl)

    # define the integration parameters in the kwargs dict
    kwargs = Dict{String, Any}(
                               "h" => h,
                               "diffusion" => diffusion,
                               "dx_params" => dx_params,
                               "dx_dt" => dx_dt,
                              )
    if diffusion != 0.0
        kwargs["p"] = p
        kwargs["α"] = α
        kwargs["ρ"] = ρ
    end

    # seed the random generator
    Random.seed!(seed)
    x = rand(Normal(), state_dim)

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
                             "seed" => seed,
                             "h" => h,
                             "diffusion" => diffusion,
                             "dx_params" => dx_params, 
                             "tanl" => tanl,
                             "nanl" => nanl,
                             "spin" => spin,
                             "state_dim" => state_dim,
                             "obs" => obs,
                             "model" => "L96"
                            )

    name = "L96_time_series_seed_" * lpad(seed, 4, "0") * 
           "_dim_" * lpad(state_dim, 2, "0") * 
           "_diff_" * rpad(diffusion, 5, "0") * 
           "_F_" * lpad(F, 4, "0") * 
           "_tanl_" * rpad(tanl, 4, "0") * 
           "_nanl_" * lpad(nanl, 5, "0") * 
           "_spin_" * lpad(spin, 4, "0") * 
           "_h_" * rpad(h, 5, "0") * 
           ".jld2"
    
    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/" 
    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
end


##############################################################################################
"""
    IEEE39bus_time_series((seed::Int64, h:Float64, tanl::Float64, nanl::Int64, spin::Int64,
                           diffusion::Float64)::NamedTuple)

Simulate a "free run" time series of the [IEEE39 Bus Swing Equation Model](@ref) for
generating an observation process and truth twin for data assimilation twin experiments.
Output from the experiment is saved in a dictionary of the form,

    Dict{String, Any}(
                      "seed" => seed,
                      "h" => h,
                      "diffusion" => diffusion,
                      "diff_mat" => diff_mat,
                      "dx_params" => dx_params, 
                      "tanl" => tanl,
                      "nanl" => nanl,
                      "spin" => spin,
                      "obs" => obs,
                      "model" => "IEEE39bus"
                     )

Where the file name is written dynamically according to the selected parameters as follows:

    "IEEE39bus_time_series_seed_" * lpad(seed, 4, "0") * 
    "_diff_" * rpad(diffusion, 5, "0") * 
    "_tanl_" * rpad(tanl, 4, "0") * 
    "_nanl_" * lpad(nanl, 5, "0") * 
    "_spin_" * lpad(spin, 4, "0") * 
    "_h_" * rpad(h, 5, "0") * 
    ".jld2"
"""
function IEEE39bus_time_series((seed, h, tanl, nanl, spin, diffusion)::NamedTuple{
                              (:seed,:h,:tanl,:nanl,:spin,:diffusion),
                              <:Tuple{Int64,Float64,Float64,Int64,Int64,Float64}})
    # time the experiment
    t1 = time()

    # set random seed 
    Random.seed!(seed)
    
    # define the model
    dx_dt = IEEE39bus.dx_dt
    state_dim = 20

    # define the model parameters
    input_data = pkgdir(DataAssimilationBenchmarks) * 
                 "/src/models/IEEE39bus_inputs/NE_EffectiveNetworkParams.jld2"
    tmp = load(input_data)
    dx_params = Dict{String, Array{Float64}}(
                                             "A" => tmp["A"], 
                                             "D" => tmp["D"], 
                                             "H" => tmp["H"], 
                                             "K" => tmp["K"],
                                             "γ" => tmp["γ"], 
                                             "ω" => tmp["ω"]
                                            )

    # define the integration scheme
    step_model! = DeSolvers.rk4_step!
    h = 0.01

    # define the diffusion coefficient structure matrix
    # notice that the perturbations are only applied to the frequencies
    # based on the change of variables derivation
    # likewise, the diffusion parameter is applied separately as an amplitude
    # in the Runge-Kutta scheme
    diff_mat = zeros(20,20)
    diff_mat[LinearAlgebra.diagind(diff_mat)[11:end]] = tmp["ω"][1] ./ (2.0 * tmp["H"])

    # set the number of discrete integrations steps between each observation time
    f_steps = convert(Int64, tanl/h)

    # set storage for the ensemble timeseries
    obs = Array{Float64}(undef, state_dim, nanl)

    # define the integration parameters in the kwargs dict
    kwargs = Dict{String, Any}(
              "h" => h,
              "diffusion" => diffusion,
              "dx_params" => dx_params, 
              "dx_dt" => dx_dt,
              "diff_mat" => diff_mat
             )

    # load the steady state, generated by long simulation without noise
    x = tmp["synchronous_state"]

    # spin the model onto the attractor
    for j in 1:spin
        for k in 1:f_steps
            step_model!(x, 0.0, kwargs)
            # set phase angles mod 2pi
            x[1:10] .= rem2pi.(x[1:10], RoundNearest)
        end
    end

    # save the model state at timesteps of tanl
    for j in 1:nanl
        for k in 1:f_steps
            step_model!(x, 0.0, kwargs)
            # set phase angles mod 2pi
            x[1:10] .= rem2pi.(x[1:10], RoundNearest)
        end
        obs[:, j] = x
    end
    
    data = Dict{String, Any}(
                             "seed" => seed,
                             "h" => h,
                             "diffusion" => diffusion,
                             "diff_mat" => diff_mat,
                             "dx_params" => dx_params, 
                             "tanl" => tanl,
                             "nanl" => nanl,
                             "spin" => spin,
                             "obs" => obs,
                             "model" => "IEEE39bus"
                            )

    name = "IEEE39bus_time_series_seed_" * lpad(seed, 4, "0") * 
           "_diff_" * rpad(diffusion, 5, "0") * 
           "_tanl_" * rpad(tanl, 4, "0") * 
           "_nanl_" * lpad(nanl, 5, "0") * 
           "_spin_" * lpad(spin, 4, "0") * 
           "_h_" * rpad(h, 5, "0") * 
           ".jld2"
    
    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"
    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
end


##############################################################################################
# end module

end
