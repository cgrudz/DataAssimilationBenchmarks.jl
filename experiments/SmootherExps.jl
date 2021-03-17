#######################################################################################################################
module SmootherExps
########################################################################################################################
########################################################################################################################
# imports and exports
using Debugger
using Random, Distributions, Statistics
using JLD
using LinearAlgebra
using EnsembleKalmanSchemes, DeSolvers, L96
export classic_state, classic_param, hybrid_state, hybrid_param, iterative_state

########################################################################################################################
########################################################################################################################
# Main smoothing experiments, debugged and validated for use with schemes in methods directory
########################################################################################################################
# All experiments are funcitonalized so that they can be called from an array of parameter values which will typically
# be varied with naive parallelism.  Relevant arguments and experimental results are dumped as a side effect to a
# dictionary in a JLD.  Returns runtime in minutes.
########################################################################################################################

function classic_state(args::Tuple{String,String,Int64,Int64,Int64,Float64,Int64,Int64,Float64})
    # time the experiment
    t1 = time()

    # Define experiment parameters
    time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, state_infl = args

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
    nanl = 25

    # set seed 
    Random.seed!(seed)
    
    # define the initial ensembles, squeezing the sys_dim times 1 array from the timeseries generation
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 2 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                "dx_dt" => dx_dt,
                "f_steps" => f_steps,
                "step_model" => step_model, 
                "dx_params" => [f],
                "h" => h,
                "diffusion" => diffusion,
                "shift" => shift,
                "mda" => false
                             )

    # define the observation operator, observation error covariance and observations with error 
    H = alternating_obs_operator(sys_dim, obs_dim, kwargs)
    obs_cov = obs_un^2.0 * I
    obs = H * obs + obs_un * rand(Normal(), size(obs))
    
    # create storage for the forecast and analysis statistics, indexed in relative time
    # the first index corresponds to time 1, last index corresponds to index nanl + 2 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1) 
    filt_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    
    fore_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)

    # make a place-holder first posterior of zeros length lag, this will become the "re-analyzed" posterior
    # for negative and first time indices
    posterior = Array{Float64}(undef, sys_dim, N_ens, lag)

    # we will run through nanl total analyses, i ranges in the absolute analysis-time index, 
    # we perform assimilation of the observation window from time 2 to time nanl + 1 + lag at increments of shift 
    # starting at time 2 because of no observations at time 1 
    # only the interval 2 : nanl + 1 is stored later for all statistics
    for i in 2: shift : nanl + 1 + lag
        kwargs["posterior"] = posterior
        # observations indexed in absolute time
        analysis = ls_smoother_classic(method, ens, H, obs[:, i: i + shift - 1], obs_cov, state_infl, kwargs)
        ens = analysis["ens"]
        fore = analysis["fore"]
        filt = analysis["filt"]
        post = analysis["post"]
        
        for j in 1:shift
            # compute the forecast, filter and analysis statistics -- indices for the forecast, filter, analysis 
            # statistics storage index starts at absolute time 1, truth index starts at absolute time 1
            fore_rmse[i + j - 1], fore_spread[i + j - 1] = analyze_ensemble(fore[:, :, j], 
                                                                                        truth[:, i + j - 1])
            filt_rmse[i + j - 1], filt_spread[i + j - 1] = analyze_ensemble(filt[:, :, j], 
                                                                                        truth[:, i + j - 1])

            # we analyze the posterior states that will be discarded in the non-overlapping DAWs
            if shift == lag
                # for the shift=lag, all states are analyzed and discared, no dummy past states are used
                # truth follows times minus 1 from the filter and forecast stastistics
                anal_rmse[i + j - 2], anal_spread[i + j - 2] = analyze_ensemble(post[:, :, j],
                                                                                truth[:, i + j - 2])

            elseif i > lag 
                # for lag > shift, we wait for the dummy lag-1-total posterior states to be cycled out
                # the first posterior starts with the first prior at time 1, later discarded to align stats
                anal_rmse[i - lag + j - 1], anal_spread[i - lag + j - 1] = analyze_ensemble(post[:, :, j], 
                                                                                    truth[:, i - lag + j - 1])
            end
        end
        
        # reset the posterior
        if lag == shift
            # the assimilation windows are disjoint and therefore we reset completely
            posterior = Array{Float64}(undef, sys_dim, N_ens, lag)
        else
            # the assimilation windows overlap and therefore we update the posterior by removing the first-shift 
            # values from the DAW and including the filter states in the last-shift values of the DAW
            posterior = cat(post[:, :, 1 + shift: end],  filt, dims=3)
        end
    end

    # cut the statistics so that they align on the same absolute time points 
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    anal_rmse = anal_rmse[2: nanl + 1]
    anal_spread = anal_spread[2: nanl + 1]

    data = Dict{String,Any}(
            "fore_rmse"=> fore_rmse,
            "filt_rmse"=> filt_rmse,
            "anal_rmse"=> anal_rmse,
            "fore_spread"=> fore_spread,
            "filt_spread"=> filt_spread,
            "anal_spread"=> anal_spread,
            "method"=> method,
            "seed" => seed, 
            "diffusion"=> diffusion,
            "sys_dim"=> sys_dim,
            "obs_dim"=> obs_dim, 
            "obs_un"=> obs_un,
            "nanl"=> nanl,
            "tanl"=> tanl,
            "lag"=> lag,
            "shift"=> shift,
            "h"=> h,
            "N_ens"=> N_ens, 
            "state_infl"=> round(state_infl, digits=2)
           )
    
    path = "./data/" * method * "_classic/" 
    name = method * 
            "_classic_smoother_l96_state_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
            "_obs_un_" * rpad(obs_un, 4, "0") *
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_h_" * rpad(h, 4, "0") *
            "_lag_" * lpad(lag, 3, "0") * 
            "_shift_" * lpad(shift, 3, "0") *
            "_N_ens_" * lpad(N_ens, 3,"0") * 
            "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") * 
            ".jld"


    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


#########################################################################################################################

function classic_param(args::Tuple{String,String,Int64,Int64,Int64,Float64,Int64,Float64,Float64,Int64,Float64,Float64})
    # time the experiment
    t1 = time()

    # Define experiment parameters
    time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl = args

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
    
    # define the initialization 
    obs = ts["obs"]::Array{Float64,2}
    init = obs[:, 1]

    # define the initial state ensemble
    ens = rand(MvNormal(init, I), N_ens)
    param_truth = [f]
    state_dim = length(init)
    sys_dim = state_dim + length(param_truth)

    # extend this by the parameter ensemble
    if length(param_truth) > 1
        # note here the covariance is supplied such that the standard deviation is a percent of the parameter value
        param_ens = rand(MvNormal(param_truth, diagm(param_truth * param_err).^2.0), N_ens)
    else
        # note here the standard deviation is supplied directly
        param_ens = rand(Normal(param_truth[1], param_truth[1]*param_err), 1, N_ens)
    end

    # define the extended state ensemble
    ens = [ens; param_ens]

    # define kwargs
    kwargs = Dict{String,Any}(
                "dx_dt" => dx_dt,
                "f_steps" => f_steps,
                "step_model" => step_model, 
                "h" => h,
                "diffusion" => diffusion,
                "state_dim" => state_dim,
                "param_wlk" => param_wlk,
                "param_infl" => param_infl,
                "shift" => shift,
                "mda" => false
                             )

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 2 * lag + 1]
    truth = copy(obs)
    H = alternating_obs_operator(state_dim, obs_dim, kwargs) 
    obs =  H * obs + obs_un * rand(Normal(), size(obs))
    obs_cov = obs_un^2.0 * I

    # define the observation operator on the extended state, used for the ensemble
    H = alternating_obs_operator(sys_dim, obs_dim, kwargs) 

    # create storage for the forecast and analysis statistics, indexed in relative time
    # the first index corresponds to time 1, last index corresponds to index nanl + 2 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1) 
    filt_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    para_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    
    fore_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    para_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)

    # make a place-holder first posterior of zeros length lag, this will become the "re-analyzed" posterior
    # for negative and first time indices
    posterior = Array{Float64}(undef, sys_dim, N_ens, lag)

    # we will run through nanl total analyses, i ranges in the absolute analysis-time index, 
    # we perform assimilation of the observation window from time 2 to time nanl + 1 + lag at increments of shift 
    # starting at time 2 because of no observations at time 1 
    # only the interval 2 : nanl + 1 is stored later for all statistics
    for i in 2: shift : nanl + 1 + lag
        kwargs["posterior"] = posterior
        # observations indexed in absolute time
        analysis = ls_smoother_classic(method, ens, H, obs[:, i: i + shift - 1], obs_cov, state_infl, kwargs)
        ens = analysis["ens"]
        fore = analysis["fore"]
        filt = analysis["filt"]
        post = analysis["post"]
        
        for j in 1:shift
            # compute the forecast, filter and analysis statistics -- indices for the forecast, filter, analysis 
            # statistics storage index starts at absolute time 1, truth index starts at absolute time 1
            fore_rmse[i + j - 1], fore_spread[i + j - 1] = analyze_ensemble(fore[1:state_dim, :, j], 
                                                                                        truth[:, i + j - 1])
            filt_rmse[i + j - 1], filt_spread[i + j - 1] = analyze_ensemble(filt[1:state_dim, :, j], 
                                                                                        truth[:, i + j - 1])

            # we analyze the posterior states that will be discarded in the non-overlapping DAWs
            if shift == lag
                # for the shift=lag, all states are analyzed and discared, no dummy past states are used
                # truth follows times minus 1 from the filter and forecast stastistics
                anal_rmse[i + j - 2], anal_spread[i + j - 2] = analyze_ensemble(post[1:state_dim, :, j],
                                                                                truth[:, i + j - 2])

                para_rmse[i + j - 2], 
                para_spread[i + j - 2] = analyze_ensemble_parameters(post[state_dim + 1: end, :, j], 
                                                                                param_truth)
            elseif i > lag 
                # for lag > shift, we wait for the dummy lag-1-total posterior states to be cycled out
                # the first posterior starts with the first prior at time 1, later discarded to align stats
                anal_rmse[i - lag + j - 1], anal_spread[i - lag + j - 1] = analyze_ensemble(post[1:state_dim, :, j], 
                                                                                    truth[:, i - lag + j - 1])

                para_rmse[i - lag + j - 1], 
                para_spread[i - lag + j - 1] = analyze_ensemble_parameters(post[state_dim + 1: end, :, j], 
                                                                                param_truth)
            end
        end
        
        # reset the posterior
        if lag == shift
            # the assimilation windows are disjoint and therefore we reset completely
            posterior = Array{Float64}(undef, sys_dim, N_ens, lag)
        else
            # the assimilation windows overlap and therefore we update the posterior by removing the first-shift 
            # values from the DAW and including the filter states in the last-shift values of the DAW
            posterior = cat(post[:, :, 1 + shift: end],  filt, dims=3)
        end
    end

    # cut the statistics so that they align on the same absolute time points 
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    anal_rmse = anal_rmse[2: nanl + 1]
    anal_spread = anal_spread[2: nanl + 1]
    para_rmse = para_rmse[2: nanl + 1]
    para_spread = para_spread[2: nanl + 1]

    data = Dict{String,Any}(
            "fore_rmse"=> fore_rmse,
            "filt_rmse"=> filt_rmse,
            "anal_rmse"=> anal_rmse,
            "param_rmse"=> para_rmse,
            "fore_spread"=> fore_spread,
            "filt_spread"=> filt_spread,
            "anal_spread"=> anal_spread,
            "param_spread"=> para_spread,
            "method"=> method,
            "seed" => seed, 
            "diffusion"=> diffusion,
            "sys_dim"=> sys_dim,
            "state_dim" => state_dim,
            "obs_dim"=> obs_dim, 
            "obs_un"=> obs_un,
            "param_err" => param_err,
            "param_wlk" => param_wlk,
            "nanl"=> nanl,
            "tanl"=> tanl,
            "lag"=> lag,
            "shift"=> shift,
            "h"=> h,
            "N_ens"=> N_ens, 
            "state_infl"=> round(state_infl, digits=2),
            "param_infl" => round(param_infl, digits=2)
           )
    
    path = "./data/" * method * "_classic/" 
    name = method * 
            "_classic_smoother_l96_param_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
            "_obs_un_" * rpad(obs_un, 4, "0") *
            "_param_err_" * rpad(param_err, 4, "0") * 
            "_param_wlk_" * rpad(param_wlk, 6, "0") * 
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_h_" * rpad(h, 4, "0") *
            "_lag_" * lpad(lag, 3, "0") * 
            "_shift_" * lpad(shift, 3, "0") *
            "_N_ens_" * lpad(N_ens, 3,"0") * 
            "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") * 
            "_param_infl_" * rpad(round(param_infl, digits=2), 4, "0") * 
            ".jld"

    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


#########################################################################################################################

function hybrid_state(args::Tuple{String,String,Int64,Int64,Int64,Bool,Float64,Int64,Int64,Float64})
    
    # time the experiment
    t1 = time()

    # Define experiment parameters
    time_series, method, seed, lag, shift, mda, obs_un, obs_dim, N_ens, state_infl = args

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
    nanl = 25

    # set seed 
    Random.seed!(seed)
    
    # define the initial ensembles
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 2 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                "dx_dt" => dx_dt,
                "f_steps" => f_steps,
                "step_model" => step_model, 
                "dx_params" => [f],
                "h" => h,
                "diffusion" => diffusion,
                "shift" => shift,
                "mda" => mda 
                             )

    # define the observation operator, observation error covariance and observations with error 
    H = alternating_obs_operator(sys_dim, obs_dim, kwargs)
    obs_cov = obs_un^2.0 * I
    obs = H * obs + obs_un * rand(Normal(), size(obs))
    
    # create storage for the forecast and analysis statistics, indexed in relative time
    # the first index corresponds to time 1, last index corresponds to index nanl + 2 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1) 
    filt_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    
    fore_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)

    # perform an initial spin for the smoothed re-analyzed first prior estimate while handling 
    # new observations with a filtering step to prevent divergence of the forecast for long lags
    spin = true
    kwargs["spin"] = spin
    posterior = Array{Float64}(undef, sys_dim, N_ens, shift)
    kwargs["posterior"] = posterior
    
    # we will run through nanl + 2 * lag total analyses but discard the last-lag forecast values and
    # first-lag posterior values so that the statistics align on the same time points after the spin
    for i in 2: shift : nanl + lag + 1
        # perform assimilation of the DAW
        # we use the observation window from current time +1 to current time +lag
        if mda
            # NOTE: mda spin weights are only designed for shift=1
            if spin
                # for the first rebalancing step, all observations are new and get fully assimilated
                # observation weights are given with respect to a special window in terms of the
                # number of times the observation will be assimilated
                kwargs["obs_weights"] = [i-1:lag-1; ones(i-1) * lag] 
                kwargs["reb_weights"] = ones(lag) 

            elseif i <= lag
                # if still processing observations from the spin cycle, deal with special weights
                # given by the number of times the observation is assimilated
                obs_weights = [i-1:lag-1; ones(i-1) * lag]
                kwargs["obs_weights"] = obs_weights
                one_minus_α_k = (Vector{Float64}(1:lag)) ./ obs_weights
                kwargs["reb_weights"] = 1 ./ one_minus_α_k 

            else
                # otherwise equal weights
                kwargs["obs_weights"] = ones(lag) * lag
                kwargs["reb_weights"] = 1 ./ (Vector{Float64}(1:lag) ./ lag)
            end
        end

        # peform the analysis
        analysis = ls_smoother_hybrid(method, ens, H, obs[:, i: i + lag - 1], obs_cov, state_infl, kwargs)
        ens = analysis["ens"]
        fore = analysis["fore"]
        filt = analysis["filt"]
        post = analysis["post"]

        if spin
            for j in 1:lag 
                # compute forecast and filter statistics on the first lag states during spin period
                fore_rmse[i - 1 + j], fore_spread[i - 1 + j] = analyze_ensemble(fore[:, :, j], 
                                                                                    truth[:, i - 1 + j])
                
                filt_rmse[i - 1 + j], filt_spread[i - 1 + j] = analyze_ensemble(filt[:, :, j], 
                                                                                    truth[:, i - 1 + j])
            end

            for j in 1:shift
                # compute only the reanalyzed prior and the shift-forward forecasted reanalysis
                anal_rmse[i - 2 + j], anal_spread[i - 2 + j] = analyze_ensemble(post[:, :, j], truth[:, i - 2 + j])
            end

            # turn off the initial spin period, continue hereafter on the normal assimilation cycle
            spin = false
            kwargs["spin"] = spin

        else
            for j in 1:shift
                # compute the forecast, filter and analysis statistics
                # indices for the forecast, filter, analysis and truth arrays are in absolute time,
                # forecast / filter stats computed beyond the first lag period for the spin
                fore_rmse[i + lag - 1 - shift + j], 
                fore_spread[i + lag - 1 - shift + j] = analyze_ensemble(fore[:, :, j], 
                                                                                    truth[:, i + lag - 1 - shift + j])
                
                filt_rmse[i + lag - 1 - shift + j], 
                filt_spread[i + lag - 1 - shift + j] = analyze_ensemble(filt[:, :, j], 
                                                                                    truth[:, i + lag - 1 - shift + j])
                
                # analysis statistics computed beyond the first shift
                anal_rmse[i - 2 + j], anal_spread[i - 2 + j] = analyze_ensemble(post[:, :, j], truth[:, i - 2 + j])
            end
        end
    end

    # cut the statistics so that they align on the same absolute time points 
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    anal_rmse = anal_rmse[2: nanl + 1]
    anal_spread = anal_spread[2: nanl + 1]

    data = Dict{String,Any}(
            "fore_rmse"=> fore_rmse,
            "filt_rmse"=> filt_rmse,
            "anal_rmse"=> anal_rmse,
            "fore_spread"=> fore_spread,
            "filt_spread"=> filt_spread,
            "anal_spread"=> anal_spread,
            "method"=> method,
            "seed" => seed, 
            "diffusion"=> diffusion,
            "sys_dim"=> sys_dim,
            "obs_dim"=> obs_dim, 
            "obs_un"=> obs_un,
            "nanl"=> nanl,
            "tanl"=> tanl,
            "lag"=> lag,
            "shift"=> shift,
            "mda" => mda,
            "h"=> h,
            "N_ens"=> N_ens, 
            "state_infl"=> round(state_infl, digits=2)
           )
    
    path = "./data/" * method * "_hybrid/" 
    name = method * 
            "_hybrid_smoother_l96_state_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
            "_obs_un_" * rpad(obs_un, 4, "0") *
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_h_" * rpad(h, 4, "0") *
            "_lag_" * lpad(lag, 3, "0") * 
            "_shift_" * lpad(shift, 3, "0") * 
            "_mda_" * string(mda) *
            "_N_ens_" * lpad(N_ens, 3,"0") * 
            "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") * 
            ".jld"


    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


#########################################################################################################################

function hybrid_adaptive_state(args::Tuple{String,String,Int64,Int64,Int64,Bool,Float64,Int64,Int64,Float64}, tail::Int64=3)
    
    # time the experiment
    t1 = time()

    # Define experiment parameters
    time_series, method, seed, lag, shift, mda, obs_un, obs_dim, N_ens, state_infl = args

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
    nanl = 25

    # set seed 
    Random.seed!(seed)
    
    # define the initial ensembles
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 2 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                "dx_dt" => dx_dt,
                "f_steps" => f_steps,
                "step_model" => step_model, 
                "dx_params" => [f],
                "h" => h,
                "diffusion" => diffusion,
                "shift" => shift,
                "mda" => mda 
                             )

    if method == "etks_adaptive"
        tail_spin = true
        kwargs["tail_spin"] = tail_spin
        kwargs["analysis"] = Array{Float64}(undef, sys_dim, N_ens, lag)
        kwargs["analysis_innovations"] = Array{Float64}(undef, sys_dim, lag)
    end

    # define the observation operator, observation error covariance and observations with error 
    H = alternating_obs_operator(sys_dim, obs_dim, kwargs)
    obs_cov = obs_un^2.0 * I
    obs = H * obs + obs_un * rand(Normal(), size(obs))
    
    # create storage for the forecast and analysis statistics, indexed in relative time
    # the first index corresponds to time 1, last index corresponds to index nanl + 2 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1) 
    filt_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    
    fore_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)

    # perform an initial spin for the smoothed re-analyzed first prior estimate while handling 
    # new observations with a filtering step to prevent divergence of the forecast for long lags
    spin = true
    kwargs["spin"] = spin
    posterior = Array{Float64}(undef, sys_dim, N_ens, shift)
    kwargs["posterior"] = posterior
    
    # we will run through nanl + 2 * lag total analyses but discard the last-lag forecast values and
    # first-lag posterior values so that the statistics align on the same time points after the spin
    for i in 2: shift : nanl + lag + 1
        # perform assimilation of the DAW
        # we use the observation window from current time +1 to current time +lag
        if mda
            # NOTE: mda spin weights are only designed for shift=1
            if spin
                # for the first rebalancing step, all observations are new and get fully assimilated
                # observation weights are given with respect to a special window in terms of the
                # number of times the observation will be assimilated
                kwargs["obs_weights"] = [i-1:lag-1; ones(i-1) * lag] 
                kwargs["reb_weights"] = ones(lag) 

            elseif i <= lag
                # if still processing observations from the spin cycle, deal with special weights
                # given by the number of times the observation is assimilated
                obs_weights = [i-1:lag-1; ones(i-1) * lag]
                kwargs["obs_weights"] = obs_weights
                one_minus_α_k = (Vector{Float64}(1:lag)) ./ obs_weights
                kwargs["reb_weights"] = 1 ./ one_minus_α_k 

            else
                # otherwise equal weights
                kwargs["obs_weights"] = ones(lag) * lag
                kwargs["reb_weights"] = 1 ./ (Vector{Float64}(1:lag) ./ lag)
            end
        end

        # peform the analysis
        analysis = ls_smoother_hybrid(method, ens, H, obs[:, i: i + lag - 1], obs_cov, state_infl, kwargs)
        ens = analysis["ens"]
        fore = analysis["fore"]
        filt = analysis["filt"]
        post = analysis["post"]

        if method == "etks_adaptive"
            @bp
            # cycle the analysis states for the new DAW
            kwargs["analysis"] = analysis["anal"]
            if tail_spin
                # check if we have reached a long enough tail of innovation statistics
                analysis_innovations = analysis["inno"]  
                if size(analysis_innovations, 2) / lag >= tail
                    # if so, stop the tail spin
                    tail_spin = false
                    kwargs["tail_spin"] = tail_spin
                end
            end 
            # cycle  the analysis states for the new DAW
            kwargs["analysis_innovations"] = analysis["inno"]
        end

        if spin
            for j in 1:lag 
                # compute forecast and filter statistics on the first lag states during spin period
                fore_rmse[i - 1 + j], fore_spread[i - 1 + j] = analyze_ensemble(fore[:, :, j], 
                                                                                    truth[:, i - 1 + j])
                
                filt_rmse[i - 1 + j], filt_spread[i - 1 + j] = analyze_ensemble(filt[:, :, j], 
                                                                                    truth[:, i - 1 + j])
            end

            for j in 1:shift
                # compute only the reanalyzed prior and the shift-forward forecasted reanalysis
                anal_rmse[i - 2 + j], anal_spread[i - 2 + j] = analyze_ensemble(post[:, :, j], truth[:, i - 2 + j])
            end

            # turn off the initial spin period, continue hereafter on the normal assimilation cycle
            spin = false
            kwargs["spin"] = spin

        else
            for j in 1:shift
                # compute the forecast, filter and analysis statistics
                # indices for the forecast, filter, analysis and truth arrays are in absolute time,
                # forecast / filter stats computed beyond the first lag period for the spin
                fore_rmse[i + lag - 1 - shift + j], 
                fore_spread[i + lag - 1 - shift + j] = analyze_ensemble(fore[:, :, j], 
                                                                                    truth[:, i + lag - 1 - shift + j])
                
                filt_rmse[i + lag - 1 - shift + j], 
                filt_spread[i + lag - 1 - shift + j] = analyze_ensemble(filt[:, :, j], 
                                                                                    truth[:, i + lag - 1 - shift + j])
                
                # analysis statistics computed beyond the first shift
                anal_rmse[i - 2 + j], anal_spread[i - 2 + j] = analyze_ensemble(post[:, :, j], truth[:, i - 2 + j])
            end
        end
    end

    # cut the statistics so that they align on the same absolute time points 
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    anal_rmse = anal_rmse[2: nanl + 1]
    anal_spread = anal_spread[2: nanl + 1]

    data = Dict{String,Any}(
            "fore_rmse"=> fore_rmse,
            "filt_rmse"=> filt_rmse,
            "anal_rmse"=> anal_rmse,
            "fore_spread"=> fore_spread,
            "filt_spread"=> filt_spread,
            "anal_spread"=> anal_spread,
            "method"=> method,
            "seed" => seed, 
            "diffusion"=> diffusion,
            "sys_dim"=> sys_dim,
            "obs_dim"=> obs_dim, 
            "obs_un"=> obs_un,
            "nanl"=> nanl,
            "tanl"=> tanl,
            "lag"=> lag,
            "shift"=> shift,
            "mda" => mda,
            "h"=> h,
            "N_ens"=> N_ens, 
            "state_infl"=> round(state_infl, digits=2)
           )
    
    if method == "etks_adaptive"
        data["tail"] = tail
    end

    path = "./data/" * method * "_hybrid/" 
    name = method * 
            "_hybrid_smoother_l96_state_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
            "_obs_un_" * rpad(obs_un, 4, "0") *
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_h_" * rpad(h, 4, "0") *
            "_lag_" * lpad(lag, 3, "0") * 
            "_shift_" * lpad(shift, 3, "0") * 
            "_mda_" * string(mda) *
            "_N_ens_" * lpad(N_ens, 3,"0") * 
            "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") * 
            ".jld"


    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


#########################################################################################################################

function hybrid_param(args::Tuple{String,String,Int64,Int64,Int64,Bool,Float64,Int64,
                                  Float64,Float64,Int64,Float64,Float64})
    
    # time the experiment
    t1 = time()

    # Define experiment parameters
    time_series, method, seed, lag, shift, mda, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl = args

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
    
    # define the initialization for the ensemble mean 
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 2 * lag + 1]
    truth = copy(obs)
    
    # define the initial state ensemble
    ens = rand(MvNormal(init, I), N_ens)
    param_truth = [f]
    state_dim = length(init)
    sys_dim = state_dim + length(param_truth)

    # extend this by the parameter ensemble
    if length(param_truth) > 1
        # note here the covariance is supplied such that the standard deviation is a percent of the parameter value
        param_ens = rand(MvNormal(param_truth, diagm(param_truth * param_err).^2.0), N_ens)
    else
        # note here the standard deviation is supplied directly
        param_ens = rand(Normal(param_truth[1], param_truth[1]*param_err), 1, N_ens)
    end

    # define the extended state ensemble
    ens = [ens; param_ens]

    # define kwargs
    kwargs = Dict{String,Any}(
                "dx_dt" => dx_dt,
                "f_steps" => f_steps,
                "step_model" => step_model, 
                "dx_params" => [f],
                "h" => h,
                "diffusion" => diffusion,
                "state_dim" => state_dim,
                "shift" => shift,
                "param_wlk" => param_wlk,
                "param_infl" => param_infl,
                "mda" => mda 
                             )

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    H = alternating_obs_operator(state_dim, obs_dim, kwargs) 
    obs =  H * obs + obs_un * rand(Normal(), size(obs))
    obs_cov = obs_un^2.0 * I

    # define the observation operator on the extended state, used for the ensemble
    H = alternating_obs_operator(sys_dim, obs_dim, kwargs) 
    
    # create storage for the forecast and analysis statistics, indexed in relative time
    # the first index corresponds to time 1, last index corresponds to index nanl + 2 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1) 
    filt_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    para_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    
    fore_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    para_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)

    # perform an initial spin for the smoothed re-analyzed first prior estimate while handling 
    # new observations with a filtering step to prevent divergence of the forecast for long lags
    spin = true
    kwargs["spin"] = spin
    posterior = zeros(sys_dim, N_ens, shift)
    kwargs["posterior"] = posterior
    
    # we will run through nanl + 2 * lag total analyses but discard the last-lag forecast values and
    # first-lag posterior values so that the statistics align on the same time points after the spin
    for i in 2: shift : nanl + lag + 1
        # perform assimilation of the DAW
        # we use the observation window from current time +1 to current time +lag
        if mda
            # NOTE: mda spin weights are only designed for shift=1
            if spin
                # for the first rebalancing step, all observations are new and get fully assimilated
                # observation weights are given with respect to a special window in terms of the
                # number of times the observation will be assimilated
                kwargs["obs_weights"] = [i-1:lag-1; ones(i-1) * lag] 
                kwargs["reb_weights"] = ones(lag) 

            elseif i <= lag
                # if still processing observations from the spin cycle, deal with special weights
                # given by the number of times the observation is assimilated
                obs_weights = [i-1:lag-1; ones(i-1) * lag]
                kwargs["obs_weights"] = obs_weights
                one_minus_α_k = (Vector{Float64}(1:lag)) ./ obs_weights
                kwargs["reb_weights"] = 1 ./ one_minus_α_k 

            else
                # otherwise equal weights
                kwargs["obs_weights"] = ones(lag) * lag
                kwargs["reb_weights"] = 1 ./ (Vector{Float64}(1:lag) ./ lag)
            end
        end

        analysis = ls_smoother_hybrid(method, ens, H, obs[:, i: i + lag - 1], obs_cov, state_infl, kwargs)
        ens = analysis["ens"]
        fore = analysis["fore"]
        filt = analysis["filt"]
        post = analysis["post"]

        if spin
            for j in 1:lag 
                # compute forecast and filter statistics on the first lag states during spin period
                fore_rmse[i - 1 + j], 
                fore_spread[i - 1 + j] = analyze_ensemble(fore[1:state_dim, :, j], 
                                                          truth[:, i - 1 + j])
                
                filt_rmse[i - 1 + j], 
                filt_spread[i - 1 + j] = analyze_ensemble(filt[1:state_dim, :, j], 
                                                          truth[:, i - 1 + j])
                
            end

            for j in 1:shift
                # compute only the reanalyzed prior and the shift-forward forecasted reanalysis
                anal_rmse[i - 2 + j], 
                anal_spread[i - 2 + j] = analyze_ensemble(post[1:state_dim, :, j], 
                                                          truth[:, i - 2 + j])
                
                para_rmse[i - 2 + j], 
                para_spread[i - 2 + j] = analyze_ensemble_parameters(post[state_dim+1:end, :, j], 
                                                                     param_truth)
                
            end

            # turn off the initial spin period, continue hereafter on the normal assimilation cycle
            spin = false
            kwargs["spin"] = spin

        else
            for j in 1:shift
                # compute the forecast, filter and analysis statistics
                # indices for the forecast, filter, analysis and truth arrays are in absolute time,
                # forecast / filter stats computed beyond the first lag period for the spin
                fore_rmse[i + lag - 1 - shift + j], 
                fore_spread[i + lag - 1 - shift + j] = analyze_ensemble(fore[1:state_dim, :, j], 
                                                                        truth[:, i + lag - 1 - shift + j])
                
                filt_rmse[i + lag - 1 - shift + j], 
                filt_spread[i + lag - 1 - shift + j] = analyze_ensemble(filt[1:state_dim, :, j],
                                                                        truth[:, i + lag - 1 - shift + j])
                
                # analysis statistics computed beyond the first shift
                anal_rmse[i - 2 + j], 
                anal_spread[i - 2 + j] = analyze_ensemble(post[1:state_dim, :, j], 
                                                          truth[:, i - 2 + j])
                
                para_rmse[i - 2 + j], 
                para_spread[i - 2 + j] = analyze_ensemble_parameters(post[state_dim+1:end, :, j], 
                                                                     param_truth)

            end

        end

    end

    # cut the statistics so that they align on the same absolute time points 
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    anal_rmse = anal_rmse[2: nanl + 1]
    anal_spread = anal_spread[2: nanl + 1]
    para_rmse = para_rmse[2: nanl + 1]
    para_spread = para_spread[2: nanl + 1]

    data = Dict{String,Any}(
            "fore_rmse"=> fore_rmse,
            "filt_rmse"=> filt_rmse,
            "anal_rmse"=> anal_rmse,
            "param_rmse"=> para_rmse,
            "fore_spread"=> fore_spread,
            "filt_spread"=> filt_spread,
            "anal_spread"=> anal_spread,
            "param_spread"=> para_spread,
            "method"=> method,
            "seed" => seed, 
            "diffusion"=> diffusion,
            "sys_dim"=> sys_dim,
            "obs_dim"=> obs_dim, 
            "obs_un"=> obs_un,
            "param_wlk" => param_wlk,
            "param_infl" => param_infl,
            "nanl"=> nanl,
            "tanl"=> tanl,
            "lag"=> lag,
            "shift"=> shift,
            "mda" => mda,
            "h"=> h,
            "N_ens"=> N_ens, 
            "state_infl"=> round(state_infl, digits=2),
            "param_infl"=> round(param_infl, digits=2)
           )
    

    path = "./data/" * method * "_hybrid/" 
    name = method * 
            "_hybrid_smoother_l96_param_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
            "_obs_un_" * rpad(obs_un, 4, "0") *
            "_param_err_" * rpad(param_err, 4, "0") * 
            "_param_wlk_" * rpad(param_wlk, 6, "0") * 
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_h_" * rpad(h, 4, "0") *
            "_lag_" * lpad(lag, 3, "0") * 
            "_shift_" * lpad(shift, 3, "0") * 
            "_mda_" * string(mda) *
            "_N_ens_" * lpad(N_ens, 3,"0") * 
            "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") * 
            "_param_infl_" * rpad(round(param_infl, digits=2), 4, "0") * 
            ".jld"


    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


#########################################################################################################################

function iterative_state(args::Tuple{String,String,Int64,Int64,Int64,Bool,Bool,Float64,Int64,Int64,Float64})
    
    # time the experiment
    t1 = time()

    # Define experiment parameters
    time_series, method, seed, lag, shift, adaptive, mda, obs_un, obs_dim, N_ens, state_infl = args

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
    nanl = 2500

    # set seed 
    Random.seed!(seed)
    
    # define the initial ensembles, squeezing the sys_dim times 1 array from the timeseries generation
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 2 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                "dx_dt" => dx_dt,
                "f_steps" => f_steps,
                "step_model" => step_model, 
                "dx_params" => [f],
                "h" => h,
                "diffusion" => diffusion,
                "shift" => shift,
                "adaptive" => adaptive,
                "mda" => mda 
                             )

    # define the observation operator, observation error covariance and observations with error 
    H = alternating_obs_operator(sys_dim, obs_dim, kwargs)
    obs_cov = obs_un^2.0 * I
    obs = H * obs + obs_un * rand(Normal(), size(obs))
    
    # create storage for the forecast and analysis statistics, indexed in relative time
    # the first index corresponds to time 1, last index corresponds to index nanl + 2 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1) 
    filt_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_rmse = Vector{Float64}(undef, nanl + 2 * lag + 1)
    
    fore_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)
    anal_spread = Vector{Float64}(undef, nanl + 2 * lag + 1)

    # create storage for the iteration sequence
    iteration_sequence = Vector{Float64}(undef, nanl + 2 * lag + 1)
    k = 1

    # perform an initial spin for the smoothed re-analyzed first prior estimate while handling 
    # new observations with a filtering step to prevent divergence of the forecast for long lags
    spin = true
    kwargs["spin"] = spin
    posterior = zeros(sys_dim, N_ens, shift)
    kwargs["posterior"] = posterior
    
    # we will run through nanl + 2 * lag total analyses but discard the last-lag forecast values and
    # first-lag posterior values so that the statistics align on the same time points after the spin
    for i in 2: shift : nanl + lag + 1
        # perform assimilation of the DAW
        # we use the observation window from current time +1 to current time +lag
        if mda
            # if still processing observations from the spin cycle, deal with special weights
            # given by the number of times the observation is assimilated
            # NOTE: mda spin weights are only designed for shift=1
            if i <= lag
                kwargs["obs_weights"] = [i-1:lag-1; ones(i-1) * lag]

            # otherwise equal weights
            else
                kwargs["obs_weights"] = ones(lag) * lag
            end
        end
        
        analysis = ls_smoother_iterative(method, ens, H, obs[:, i: i + lag - 1], obs_cov, state_infl, kwargs)
        ens = analysis["ens"]
        fore = analysis["fore"]
        filt = analysis["filt"]
        post = analysis["post"]
        iteration_sequence[k] = analysis["iterations"][1]
        k+=1

        if spin
            for j in 1:lag 
                # compute forecast and filter statistics on the first lag states during spin period
                fore_rmse[i - 1 + j], fore_spread[i - 1 + j] = analyze_ensemble(fore[:, :, j], 
                                                                                    truth[:, i - 1 + j])
                
                filt_rmse[i - 1 + j], filt_spread[i - 1 + j] = analyze_ensemble(filt[:, :, j], 
                                                                                    truth[:, i - 1 + j])
                
            end

            for j in 1:shift
                # compute only the reanalyzed prior and the shift-forward forecasted reanalysis
                anal_rmse[i - 2 + j], anal_spread[i - 2 + j] = analyze_ensemble(post[:, :, j], truth[:, i - 2 + j])
                
            end

            # turn off the initial spin period, continue hereafter on the normal assimilation cycle
            spin = false
            kwargs["spin"] = spin

        else
            for j in 1:shift
                # compute the forecast, filter and analysis statistics
                # indices for the forecast, filter, analysis and truth arrays are in absolute time,
                # forecast / filter stats computed beyond the first lag period for the spin
                fore_rmse[i + lag - 1 - shift + j], 
                fore_spread[i + lag - 1 - shift + j] = analyze_ensemble(fore[:, :, j], 
                                                                                    truth[:, i + lag - 1 - shift + j])
                
                filt_rmse[i + lag - 1 - shift + j], 
                filt_spread[i + lag - 1 - shift + j] = analyze_ensemble(filt[:, :, j], 
                                                                                    truth[:, i + lag - 1 - shift + j])
                
                # analysis statistics computed beyond the first shift
                anal_rmse[i - 2 + j], anal_spread[i - 2 + j] = analyze_ensemble(post[:, :, j], truth[:, i - 2 + j])

            end

        end

    end

    # cut the statistics so that they align on the same absolute time points 
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    anal_rmse = anal_rmse[2: nanl + 1]
    anal_spread = anal_spread[2: nanl + 1]
    iteration_sequence = iteration_sequence[2:nanl+1]

    data = Dict{String,Any}(
            "fore_rmse"=> fore_rmse,
            "filt_rmse"=> filt_rmse,
            "anal_rmse"=> anal_rmse,
            "fore_spread"=> fore_spread,
            "filt_spread"=> filt_spread,
            "anal_spread"=> anal_spread,
            "iteration_sequence" => iteration_sequence,
            "method"=> method,
            "seed" => seed, 
            "diffusion"=> diffusion,
            "sys_dim"=> sys_dim,
            "obs_dim"=> obs_dim, 
            "obs_un"=> obs_un,
            "nanl"=> nanl,
            "tanl"=> tanl,
            "lag"=> lag,
            "shift"=> shift,
            "adaptive" => adaptive,
            "mda" => mda,
            "h"=> h,
            "N_ens"=> N_ens, 
            "state_infl"=> round(state_infl, digits=2)
           )
    

    path = "./data/" * method * "/"
    name = method * 
            "_l96_state_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * 
            "_obs_dim_" * lpad(obs_dim, 2, "0") * 
            "_obs_un_" * rpad(obs_un, 4, "0") *
            "_nanl_" * lpad(nanl, 5, "0") * 
            "_tanl_" * rpad(tanl, 4, "0") * 
            "_h_" * rpad(h, 4, "0") *
            "_lag_" * lpad(lag, 3, "0") * 
            "_shift_" * lpad(shift, 3, "0") * 
            "_adaptive_" * string(adaptive) *
            "_mda_" * string(mda) *
            "_N_ens_" * lpad(N_ens, 3,"0") * 
            "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") * 
            ".jld"


    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


#########################################################################################################################

end
