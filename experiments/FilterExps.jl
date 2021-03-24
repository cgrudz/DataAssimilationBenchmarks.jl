#######################################################################################################################
module FilterExps
########################################################################################################################
########################################################################################################################
# imports and exports
using Debugger, JLD
using Random, Distributions, Statistics
using LinearAlgebra
using EnsembleKalmanSchemes, DeSolvers, L96
export filter_state, filter_param

########################################################################################################################
########################################################################################################################
# Main filtering experiments, debugged and validated for use with schemes in methods directory
########################################################################################################################

function filter_state(args::Tuple{String,String,Int64,Float64,Int64,Int64,Float64})
    # time the experiment
    t1 = time()

    # Define experiment parameters
    time_series, method, seed, obs_un, obs_dim, N_ens, infl = args

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    f = ts["F"]::Float64
    tanl = ts["tanl"]::Float64
    h = 0.01
    dx_dt = L96.dx_dt
    step_model = rk4_step!
    
    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # number of analyses
    nanl = 25000

    # set seed 
    Random.seed!(seed)
    
    # define the initial ensembles, squeezing the sys_dim times 1 array from the timeseries generation
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 2:nanl + 1]
    truth = copy(obs)
    
    # define kwargs for the filtering method
    kwargs = Dict{String,Any}(
              "dx_dt" => dx_dt,
              "f_steps" => f_steps,
              "step_model" => step_model, 
              "dx_params" => [f],
              "h" => h,
              "diffusion" => diffusion,
             )

    # define the observation operator, observation error covariance and observations with error 
    H = alternating_obs_operator(sys_dim, obs_dim, kwargs)
    obs_cov = obs_un^2.0 * I
    obs = H * obs + obs_un * rand(Normal(), obs_dim, nanl)
    
    # create storage for the forecast and analysis statistics
    fore_rmse = Vector{Float64}(undef, nanl)
    filt_rmse = Vector{Float64}(undef, nanl)
    
    fore_spread = Vector{Float64}(undef, nanl)
    filt_spread = Vector{Float64}(undef, nanl)

    # loop over the number of observation-forecast-analysis cycles
    for i in 1:nanl
        # for each ensemble member
        for j in 1:N_ens
            # loop over the integration steps between observations
            for k in 1:f_steps
                ens[:, j] = step_model(ens[:, j], kwargs, 0.0)
            end
        end

        # compute the forecast statistics
        fore_rmse[i], fore_spread[i] = analyze_ensemble(ens, truth[:, i])

        # after the forecast step, perform assimilation of the observation
        analysis = ensemble_filter(method, ens, H, obs[:, i], obs_cov, infl, kwargs)
        ens = analysis["ens"]

        # compute the analysis statistics
        filt_rmse[i], filt_spread[i] = analyze_ensemble(ens, truth[:, i])
    end

    data = Dict{String,Any}(
            "fore_rmse" => fore_rmse,
            "filt_rmse" => filt_rmse,
            "fore_spread" => fore_spread,
            "filt_spread" => filt_spread,
            "method" => method,
            "seed" => seed, 
            "diffusion" => diffusion,
            "sys_dim" => sys_dim,
            "obs_dim" => obs_dim, 
            "obs_un" => obs_un,
            "nanl" => nanl,
            "tanl" => tanl,
            "h" =>  h,
            "N_ens" => N_ens, 
            "state_infl" => round(infl, digits=2)
           ) 
        
    path = "./data/" * method * "/" 
    name = method * 
            "_l96_state_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_diffusion_" * rpad(diffusion, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
            "_obs_un_" * rpad(obs_un, 4, "0") *
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_h_" * rpad(h, 4, "0") *
            "_N_ens_" * lpad(N_ens, 3,"0") * 
            "_state_inflation_" * rpad(round(infl, digits=2), 4, "0") * 
            ".jld"

    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
end


########################################################################################################################


function filter_param(args::Tuple{String,String,Int64,Float64,Int64,Float64,Float64,Int64,Float64,Float64})
    # time the experiment
    t1 = time()

    # Define experiment parameters
    time_series, method, seed, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl = args

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    f = ts["F"]::Float64
    tanl = ts["tanl"]::Float64
    h = 0.01
    dx_dt = L96.dx_dt
    step_model = rk4_step!
    
    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # number of analyses
    nanl = 45

    # set seed 
    Random.seed!(seed)
    
    # define the initial ensembles, squeezing the sys_dim times 1 array from the timeseries generation
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    param_truth = [f]
    state_dim = length(init)
    sys_dim = state_dim + length(param_truth)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 2:nanl + 1]
    truth = copy(obs)
    
    # define kwargs, note the possible exclusion of dx_params if this is the only parameter for
    # dx_dt and this is the parameter to be estimated
    kwargs = Dict{String,Any}(
              "dx_dt" => dx_dt,
              "f_steps" => f_steps,
              "step_model" => step_model,
              "h" => h,
              "diffusion" => diffusion,
              "state_dim" => state_dim,
              "param_infl" => param_infl
             )
    
    # define the initial ensembles
    ens = rand(MvNormal(init, I), N_ens)
    
    if length(param_truth) > 1
        # note here the covariance is supplied such that the standard deviation is a percent of the parameter value
        param_ens = rand(MvNormal(param_truth, diagm(param_truth * param_err).^2.0), N_ens)
    else
        # note here the standard deviation is supplied directly
        param_ens = rand(Normal(param_truth[1], param_truth[1]*param_err), 1, N_ens)
    end
    
    # defined the extended state ensemble
    ens = [ens; param_ens]

    # define the observation operator for the dynamic state variables -- note, the param_truth is not part of the
    # truth state vector below, this is stored separately
    H = alternating_obs_operator(state_dim, obs_dim, kwargs) 
    obs =  H * obs + obs_un * rand(Normal(), obs_dim, nanl)
    
    # define the observation operator on the extended state, used for the ensemble
    H = alternating_obs_operator(sys_dim, obs_dim, kwargs) 

    # define the associated time invariant observation error covariance
    obs_cov = obs_un^2.0 * I

    # create storage for the forecast and analysis statistics
    fore_rmse = Vector{Float64}(undef, nanl)
    filt_rmse = Vector{Float64}(undef, nanl)
    para_rmse = Vector{Float64}(undef, nanl)
    
    fore_spread = Vector{Float64}(undef, nanl)
    filt_spread = Vector{Float64}(undef, nanl)
    para_spread = Vector{Float64}(undef, nanl)

    # loop over the number of observation-forecast-analysis cycles
    for i in 1:nanl
        # for each ensemble member
        for j in 1:N_ens
            for k in 1:f_steps
                # loop over the integration steps between observations
                ens[:, j] = step_model(ens[:, j], kwargs, 0.0)
            end
        end
    
        # compute the forecast statistics
        fore_rmse[i], fore_spread[i] = analyze_ensemble(ens[1:state_dim, :], truth[:, i])

        # after the forecast step, perform assimilation of the observation
        @bp
        analysis = ensemble_filter(method, ens, H, obs[:, i], obs_cov, state_infl, kwargs)
        ens = analysis["ens"]::Array{Float64,2}

        # extract the parameter ensemble for later usage
        param_ens = ens[state_dim+1:end, :]

        # compute the analysis statistics
        filt_rmse[i], filt_spread[i] = analyze_ensemble(ens[1:state_dim, :], truth[:, i])
        para_rmse[i], para_spread[i] = analyze_ensemble_parameters(param_ens, param_truth)

        # include random walk for the ensemble of parameters
        param_ens = param_ens + param_wlk * rand(Normal(), length(param_truth), N_ens)
        ens[state_dim+1:end, :] = param_ens
    end

    data = Dict{String,Any}(
            "fore_rmse" => fore_rmse,
            "filt_rmse" => filt_rmse,
            "param_rmse" => para_rmse,
            "fore_spread" => fore_spread,
            "filt_spread" => filt_spread,
            "param_spread" => para_spread,
            "method" => method,
            "seed" => seed, 
            "diffusion" => diffusion,
            "sys_dim" => sys_dim,
            "state_dim" => state_dim,
            "obs_dim" => obs_dim, 
            "obs_un" => obs_un,
            "param_err" => param_err,
            "param_wlk" => param_wlk,
            "nanl" => nanl,
            "tanl" => tanl,
            "h" => h,
            "N_ens" => N_ens, 
            "state_infl" => round(state_infl, digits=2),
            "param_infl" => round(param_infl, digits=2)
            )
    
    path = "./data/" * method * "/" 
    name =  method * 
            "_l96_param_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_diffusion_" * rpad(diffusion, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
            "_state_dim_" * lpad(state_dim, 2, "0") * 
            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
            "_obs_un_" * rpad(obs_un, 4, "0") * 
            "_param_err_" * rpad(param_err, 4, "0") * 
            "_param_wlk_" * rpad(param_wlk, 6, "0") * 
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_h_" * rpad(h, 4, "0") * 
            "_N_ens_" * lpad(N_ens, 3, "0") * 
            "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") *
            "_param_infl_" * rpad(round(param_infl, digits=2), 4, "0") * 
            ".jld"

    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
end


########################################################################################################################

end

