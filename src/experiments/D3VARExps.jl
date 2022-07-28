##############################################################################################
module D3VARExps
##############################################################################################
# imports and exports
using Random, Distributions, LinearAlgebra, StatsBase
using JLD2, HDF5, Plots
using ..DataAssimilationBenchmarks, ..ObsOperators, ..DeSolvers, ..XdVAR
##############################################################################################
# Main 3DVAR experiments
##############################################################################################

#=function D3_var_filter_analysis((time_series, method, seed, nanl, lag, shift, obs_un, obs_dim,
                        γ, N_ens, s_infl)::NamedTuple{
                        (:time_series,:method,:seed,:nanl,:lag,:shift,:obs_un,:obs_dim,
                        :γ,:N_ens,:s_infl),
                        <:Tuple{String,String,Int64,Int64,Int64,Int64,Float64,Int64,
                            Float64,Int64,Float64}})=#
function D3_var_filter_analysis()
    
    # time the experiment
    t1 = time()

    # Define experiment parameters
    # number of cycles in experiment
    nanl = 40
    # load the timeseries and associated parameters
    # ts = load(time_series)::Dict{String,Any}
    # diffusion = ts["diffusion"]::Float64
    diffusion = 0.0
    # dx_params = ts["dx_params"]::ParamDict(Float64)
    # tanl = ts["tanl"]::Float64
    tanl = 0.05
    # model = ts["model"]::String
    γ = [8.0]

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # set the integration step size for the ensemble at 0.01 - we are assuming SDE
    h = 0.01

    # define derivative parameter
    dx_params = Dict{String, Vector{Float64}}("F" => [8.0])

    # define the dynamical model derivative for this experiment from the name
    # supplied in the time series - we are assuming Lorenz-96 model
    dx_dt = L96.dx_dt

    # define integration method
    step_model! = rk4_step!

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # set seed
    seed = 234
    Random.seed!(seed)

    # define the initialization
    # observation noise
    v = rand(Normal(0, 1), 40)

    # define the initial observation range and truth reference solution
    x_b = zeros(40)
    x_t = x_b + v
    
    # define kwargs for the analysis method
    # and the underlying dynamical model
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "dx_params" => dx_params,
                              "h" => h,
                              "diffusion" => diffusion,
                              "gamma" => γ,
                             )

    # create storage for the forecast and analysis statistics
    fore_rmse = Vector{Float64}(undef, nanl)
    filt_rmse = Vector{Float64}(undef, nanl)
    
    for i in 1:nanl
        #print("Iteration: ")
        #display(i)
        for j in 1:f_steps
            # M(x^b)
            step_model!(x_b, 0.0, kwargs)
            # M(x^t)
            step_model!(x_t, 0.0, kwargs)
        end

    # multivariate - rand(MvNormal(zeros(40), I))
    w = rand(Normal(0, 1), 40)
    obs = x_t + w

    state_cov = I
    obs_cov = I

    # generate initial forecast cost
    # J_i = XdVAR.D3_var_cost(x_b, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    # print("Cost Function Output: \n")
    # display(J)
    # optimized cost function input and value
    x_opt = XdVAR.D3_var_NewtonOp(x_b, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    # J_opt = XdVAR.D3_var_cost(x_opt, obs, x_b, state_cov, H_obs, obs_cov, kwargs)
    # print("Optimized Cost Function Output: \n")
    # display(J_opt)

    # compare model forecast and truth twin via RMSE
    rmse_forecast = sqrt(msd(x_b, x_t))
    #print("RMSE between Model Forecast and Truth Twin: ")
    #display(rmse_forecast)
    fore_rmse[i] = rmse_forecast

    # compare optimal forecast and truth twin via RMSE
    rmse_filter = sqrt(msd(x_opt, x_t))
    #print("RMSE between Optimal Forecast and Truth Twin: ")
    #display(rmse_filter)
    filt_rmse[i] = rmse_filter

    # reinitializing x_b and x_t for next cycle
    x_b = x_opt
    end

    data = Dict{String,Any}(
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "gamma" => γ,
                            "tanl" => tanl,
                            "nanl" => nanl,
                            "h" =>  h,
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse
                           )

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"
    name = "L96_3DVAR_time_series_seed_" * lpad(seed, 4, "0") *
           "_diff_" * rpad(diffusion, 5, "0") *
           #"_F_" * lpad(, 4, "0") *
           "_tanl_" * rpad(tanl, 4, "0") *
           "_nanl_" * lpad(nanl, 5, "0") *
           "_h_" * rpad(h, 5, "0") *
           ".jld2"

    save(path * name, data)
    # output time
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
    # make plot
    t = 1:nanl
    plot(t, fore_rmse, marker=(:circle,5), label = "Forecast")
    plot!(t, filt_rmse, marker=(:circle,5), label = "Filter")
    xlabel!("Time [Cycles]")
    ylabel!("Root-Mean-Square Error [RMSE]")
end

##############################################################################################
# end module

end