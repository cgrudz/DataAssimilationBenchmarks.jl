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
export classic_state, classic_param

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
    nanl = 45

    # set seed 
    Random.seed!(seed)
    
    # define the initial ensembles, squeezing the sys_dim times 1 array from the timeseries generation
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + lag + shift + 1]
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
    name = method * "_classic_smoother_l96_state_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * "_obs_dim_" * lpad(obs_dim, 2, "0") * "_obs_un_" * rpad(obs_un, 4, "0") *
            "_nanl_" * lpad(nanl, 5, "0") * "_tanl_" * rpad(tanl, 4, "0") * "_h_" * rpad(h, 4, "0") *
            "_lag_" * lpad(lag, 3, "0") * "_shift_" * lpad(shift, 3, "0") *
            "_N_ens_" * lpad(N_ens, 3,"0") * "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") * ".jld"


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
    nanl = 45

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
    obs = obs[:, 1:nanl + lag + shift + 1]
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
    @bp
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
            @bp
            if shift == lag
                # for the shift=lag, all states are analyzed and discared, no dummy past states are used
                # truth follows times minus 1 from the filter and forecast stastistics
                anal_rmse[i + j - 2], anal_spread[i + j - 2] = analyze_ensemble(post[:, :, j],
                                                                                truth[:, i + j - 2])

                para_rmse[i + j - 2], 
                param_spread[i + j - 2] = analyze_ensemble_parameters(post[state_dim + 1: end, :, j], 
                                                                                param_truth)
            elseif i > lag 
                # for lag > shift, we wait for the dummy lag-1-total posterior states to be cycled out
                # the first posterior starts with the first prior at time 1, later discarded to align stats
                anal_rmse[i - lag + j - 1], anal_spread[i - lag + j - 1] = analyze_ensemble(post[:, :, j], 
                                                                                    truth[:, i - lag + j - 1])

                para_rmse[i - lag + j - 1], 
                para_spread[i - lag + j - 1] = analyze_ensemble_parameters(post[state_dim + 1: end, :, j], 
                                                                                param_truth)
            end
        end
        
        # reset the posterior
        @bp
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
    name = method * "_classic_smoother_l96_state_benchmark_seed_" * lpad(seed, 4, "0") * 
            "_sys_dim_" * lpad(sys_dim, 2, "0") * "_obs_dim_" * lpad(obs_dim, 2, "0") * "_obs_un_" * rpad(obs_un, 4, "0") *
            "_nanl_" * lpad(nanl, 5, "0") * "_tanl_" * rpad(tanl, 4, "0") * "_h_" * rpad(h, 4, "0") *
            "_lag_" * lpad(lag, 3, "0") * "_shift_" * lpad(shift, 3, "0") *
            "_N_ens_" * lpad(N_ens, 3,"0") * "_state_inflation_" * rpad(round(state_infl, digits=2), 4, "0") * ".jld"


    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


#########################################################################################################################

end
#########################################################################################################################
#
#
#def classic_param(args):
#    # Define experiment parameters
#    [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
#    # load the timeseries and associated parameters
#    tmp = picopen(time_series)
#    diffusion = tmp["diffusion"]
#    f = tmp["f"]
#    tanl = tmp["tanl"]
#    h = 0.01
#    
#    # number of discrete forecast steps
#    f_steps = int(tanl / h)
#
#    # unpack the observations and the initial true state of the dynamic variables at time 0
#    obs = np.squeeze(tmp["obs"])
#    init = obs[:, 0]
#
#    # define the state dynamic state dimension and the extended state parameters to be estimated
#    state_dim = len(init)
#    sys_dim = state_dim
#    param_truth = np.array([f])
#    sys_dim = state_dim + len(param_truth)
#
#    # define kwargs
#    kwargs = {
#              "dx_dt"=> dx_dt,
#              "f_steps"=> f_steps,
#              "step_model"=> step_model, 
#              "h"=> h,
#              "diffusion"=> diffusion,
#              "shift"=> shift,
#              "mda"=> False,
#              "state_dim"=> state_dim,
#              "param_infl"=> param_infl,
#              "param_wlk"=> param_wlk
#             }
#
#    # number of analyses
#    nanl = 4500
#
#    # set seed 
#    np.random.seed(seed)
#    
#    # define the initial ensembles
#    ens = np.random.multivariate_normal(init, np.eye(state_dim), size=N_ens).transpose()
#
#    if len(param_truth) > 1:
#        param_ens = np.random.multivariate_normal(np.squeeze(param_truth), np.diag(param_truth * param_err)**2, size=N_ens)
#    else:
#        param_ens = np.reshape(np.random.normal(np.squeeze(param_truth), scale=np.squeeze(param_truth)*param_err, size=N_ens), [1, N_ens])
#
#    # defined the extended state ensemble
#    ens = np.concatenate([ens, param_ens], axis=0)
#
#    # observations and truth arrays are indexed in absolute time and padded to cut all statistics to the same
#    # time interval of the interior nanl-time analyses
#    obs = obs[:, :nanl + 3 * lag + 1]
#    truth = copy.copy(obs)
#    
#    # define the observation sequence where we project the true state into the observation space and
#    # perturb by white-in-time-and-space noise with standard deviation obs_un
#    # note, the param_truth is not part of the  truth state vector below
#    H = alternating_obs_operator(state_dim, obs_dim) 
#    obs = H @ obs + obs_un * np.random.standard_normal(np.shape(obs))
#    obs_cov = obs_un**2 * np.eye(obs_dim)
#
#    # define the observation operator on the extended state, used for the ensemble
#    H = alternating_obs_operator(sys_dim, obs_dim, **kwargs)
#
#    # create storage for the forecast and analysis statistics, indexed in absolute time
#    fore_rmse = np.zeros(nanl + 3 * lag + 1)
#    filt_rmse = np.zeros(nanl + 3 * lag + 1)
#    anal_rmse = np.zeros(nanl + 3 * lag + 1)
#    param_rmse = np.zeros(nanl + 3 * lag + 1)
#    
#    fore_spread = np.zeros(nanl + 3 * lag + 1)
#    filt_spread = np.zeros(nanl + 3 * lag + 1)
#    anal_spread = np.zeros(nanl + 3 * lag + 1)
#    param_spread = np.zeros(nanl + 3 * lag + 1)
#
#    # make a place-holder first posterior of zeros length lag, this will become the first "re-analyzed" posterior
#    # to be discarded
#    posterior = np.zeros([sys_dim, N_ens, lag])
#    
#    # we will run through nanl + lag total analyses, discarding the first lag reanalysis and last lag filter
#    # and forecast values such that the statistics align on the same absolute time points
#    for i in range(1, nanl + 2 * lag + 1, shift):
#        # perform assimilation of the DAW, resassgining the posterior for the window
#        kwargs["posterior"] = posterior
#        analysis = lag_shift_smoother_classic(method, ens, H, obs[:, i: i + shift], obs_cov, state_infl, **kwargs)
#        ens = analysis["ens"]
#        fore = analysis["fore"]
#        filt = analysis["filt"]
#        post = analysis["post"]
#        
#        for j in range(shift):
#            # compute the forecast, filter and analysis statistics
#            # indices for the forecast, filter, analysis statistics and the truth are in absolute time, not relative
#            # starting from time 0 
#            fore_rmse[i + j], fore_spread[i + j] = analyze_ensemble(fore[:state_dim, :, j], truth[:, i + j])
#            filt_rmse[i + j], filt_spread[i + j] = analyze_ensemble(filt[:state_dim, :, j], truth[:, i + j])
#            
#            if shift == lag:
#                anal_rmse[i - 1 + j], anal_spread[i - 1 + j] = analyze_ensemble(post[:state_dim, :, j],
#                                                                                truth[:state_dim, i - 1 + j])
#
#                param_rmse[i - 1 + j], param_spread[i - 1 + j] = analyze_ensemble_parameters(post[state_dim:, :, j], 
#                                                                                param_truth)
#
#            elif i >= lag:
#                anal_rmse[i - lag + j], anal_spread[i - lag + j] = analyze_ensemble(post[:state_dim, :, j], 
#                                                                                    truth[:, i - lag + j])
#        
#                param_rmse[i - lag + j], param_spread[i - lag + j] = analyze_ensemble_parameters(post[state_dim:, :, j], 
#                                                                                param_truth)
#
#        # reset the posterior
#        if lag == shift:
#            # the assimilation windows are disjoint and therefore we reset completely
#            posterior = np.zeros([sys_dim, N_ens, lag])
#        else:
#            # the assimilation windows overlap and therefore we update the posterior by removing the first-shift 
#            # values from the DAW and including the filter states in the last-shift values of the DAW
#            posterior = np.concatenate([post[:, :, shift:],  filt], axis=2)
#
#
#    # cut the statistics so that they align on the same time points
#    fore_rmse = fore_rmse[lag + 1: lag + 1 + nanl]
#    fore_spread = fore_spread[lag + 1: lag + 1 + nanl]
#    filt_rmse = filt_rmse[lag + 1: lag + 1 + nanl]
#    filt_spread = filt_spread[lag + 1: lag + 1 + nanl]
#    anal_rmse = anal_rmse[lag + 1: lag + 1 + nanl]
#    anal_spread = anal_spread[lag + 1: lag + 1 + nanl]
#    param_rmse = param_rmse[lag + 1: lag + 1 + nanl]
#    param_spread = param_spread[lag + 1: lag + 1 + nanl]
#
#    data = {
#            "fore_rmse": fore_rmse,
#            "filt_rmse"=> filt_rmse,
#            "anal_rmse"=> anal_rmse,
#            "param_rmse"=> param_rmse,
#            "fore_spread"=> fore_spread,
#            "filt_spread"=> filt_spread,
#            "anal_spread"=> anal_spread,
#            "param_spread"=> param_spread,
#            "seed" => seed, 
#            "method"=> method,
#            "diffusion"=> diffusion,
#            "sys_dim"=> sys_dim,
#            "state_dim"=> state_dim,
#            "obs_dim"=> obs_dim, 
#            "obs_un"=> obs_un,
#            "param_err"=> param_err,
#            "param_wlk"=> param_wlk,
#            "nanl"=> nanl,
#            "tanl"=> tanl,
#            "lag"=> lag,
#            "shift"=> shift,
#            "h"=> h,
#            "N_ens"=> N_ens, 
#            "state_infl"=> round(state_infl, 2),
#            "param_infl"=> round(param_infl, 2)
#            }
#    
#    fname = "./data/" + method + "_classic/" + method + "_classic_smoother_l96_param_benchmark_seed_" +\
#            str(seed).zfill(2) + "_diffusion_" + str(float(diffusion)).ljust(4, "0") + "_sys_dim_" +\
#            str(sys_dim) + "_state_dim_" + str(state_dim)+ "_obs_dim_" + str(obs_dim) +\
#            "_obs_un_" + str(obs_un).ljust(4, "0") + "_param_err_" + str(param_err).ljust(4, "0") +\
#            "_param_wlk_" + str(param_wlk).ljust(6, "0") + "_nanl_" + str(nanl).zfill(3) +\
#            "_tanl_" + str(tanl).zfill(3) + "_h_" + str(h).ljust(4, "0") + "_lag_" + str(lag).zfill(3) +\
#            "_shift_" + str(shift).zfill(3) + "_N_ens_" + str(N_ens).zfill(3) +\
#            "_state_infl_" + str(round(state_infl, 2)).ljust(4, "0") +\
#            "_param_infl_" + str(round(param_infl, 2)).ljust(4, "0") + ".txt"
#
#    picwrite(data, fname)
#    return(args)
#
#########################################################################################################################
#
#def hybrid_state(args)=>
#    # Define experiment parameters
#    [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, state_infl] = args
#
#    # load the timeseries and associated parameters
#    tmp = picopen(time_series)
#    diffusion = tmp["diffusion"]
#    f = tmp["f"]
#    tanl = tmp["tanl"]
#    h = 0.01
#    
#    # number of discrete forecast steps
#    f_steps = int(tanl / h)
#
#    # define kwargs
#    kwargs = {
#              "dx_dt"=> dx_dt,
#              "f_steps"=> f_steps,
#              "step_model"=> step_model, 
#              "dx_params"=> [f],
#              "h"=> h,
#              "diffusion"=> diffusion,
#              "shift"=> shift,
#             }
#
#    # number of analyses
#    nanl = 4500 
#
#    # set seed 
#    np.random.seed(seed)
#    
#    # define the initial ensembles, squeezing the sys_dim times 1 array from the timeseries generation
#    obs = np.squeeze(tmp["obs"])
#    init = obs[:, 0]
#    sys_dim = len(init)
#    ens = np.random.multivariate_normal(init, np.eye(sys_dim), size=N_ens).transpose()
#
#    # obs and truth indices are defined in absolute time from time zero, we pad the obs sequence with 
#    # 2*lag-length states to align the posterior, filter and forecast statistics
#    obs = obs[:, : nanl + 3 * lag + 1]
#    truth = copy.copy(obs)
#    H = alternating_obs_operator(sys_dim, obs_dim)
#    obs = H @ obs + obs_un * np.random.standard_normal(np.shape(obs))
#    
#    # define the associated time-invariant observation error covariance
#    obs_cov = obs_un**2 * np.eye(obs_dim)
#
#    # create storage for the forecast and analysis statistics
#    fore_rmse = np.zeros(nanl + 3 * lag + 1)
#    filt_rmse = np.zeros(nanl + 3 * lag + 1)
#    anal_rmse = np.zeros(nanl + 3 * lag + 1)
#    
#    fore_spread = np.zeros(nanl + 3 * lag + 1)
#    filt_spread = np.zeros(nanl + 3 * lag + 1)
#    anal_spread = np.zeros(nanl + 3 * lag + 1)
#
#    # set multiple data assimilation (MDA) boolean and supply weights for spin if true
#    mda = True
#    kwargs["mda"] = mda
#
#    # spin weights are supplied to assimilate each observation fully over the shifting
#    # MDA weights are multiplied by the observation error covariance matrix
#    kwargs["obs_weights"] =  np.arange(1, lag + 1)
#
#    # perform an initial spin for the smoothed re-analyzed first prior estimate 
#    # using observations over absolute times 1 to lag, resulting in ens at time 0+shift
#    kwargs["spin"] = True
#    analysis = lag_shift_smoother_hybrid(method, ens, H, obs[:, 1: lag + 1], obs_cov, state_infl, **kwargs)
#    ens = analysis["ens"]
#
#    # reset the spin pameter for the regular assimilation cycle
#    kwargs["spin"] = False
#    
#    # we will run through nanl + 2 * lag total analyses but discard the last-lag forecast values and
#    # first-lag posterior values so that the statistics align on the same time points after the spin
#    for i in range(shift + 1, nanl + lag + 2, shift):
#        # perform assimilation of the DAW
#        # we use the observation window from current time +1 to current time +lag
#        if mda:
#            # if still processing observations from the spin cycle, deal with special weights
#            if i <= lag:
#                kwargs["obs_weights"] = np.concatenate([np.arange(i, lag + 1), np.ones(i-1) * lag], axis=0)
#
#            # otherwise equal weights
#            else:
#                kwargs["obs_weights"] = np.ones([lag]) * lag
#
#        analysis = lag_shift_smoother_hybrid(method, ens, H, obs[:, i: i + lag], obs_cov, state_infl, **kwargs)
#        ens = analysis["ens"]
#        fore = analysis["fore"]
#        filt = analysis["filt"]
#        post = analysis["post"]
#
#        for j in range(shift):
#            # compute the forecast, filter and analysis statistics
#            # forward index the true state by 1, because the sequence starts at time zero for which there is no
#            # observation
#            # indices for the forecast, filter, analysis and truth arrays are in absolute time, not relative
#            fore_rmse[i + lag - shift + j], fore_spread[i + lag - shift + j] = analyze_ensemble(fore[:, :, j], 
#                                                                                    truth[:, i + lag - shift + j])
#            filt_rmse[i + lag - shift + j], filt_spread[i + lag - shift + j] = analyze_ensemble(filt[:, :, j], 
#                                                                                    truth[:, i + lag - shift + j])
#            anal_rmse[i - 1 + j], anal_spread[i - 1 + j] = analyze_ensemble(post[:, :, j], truth[:, i - 1 + j])
#
#    # cut the statistics so that they align on the same time points
#    fore_rmse = fore_rmse[lag + 1: lag + 1 + nanl]
#    fore_spread = fore_spread[lag + 1: lag + 1 + nanl]
#    filt_rmse = filt_rmse[lag + 1: lag + 1 + nanl]
#    filt_spread = filt_spread[lag + 1: lag + 1 + nanl]
#    anal_rmse = anal_rmse[lag + 1: lag + 1 + nanl]
#    anal_spread = anal_spread[lag + 1: lag + 1 + nanl]
#
#    data = {
#            "fore_rmse"=> fore_rmse,
#            "filt_rmse"=> filt_rmse,
#            "anal_rmse"=> anal_rmse,
#            "fore_spread"=> fore_spread,
#            "filt_spread"=> filt_spread,
#            "anal_spread"=> anal_spread,
#            "method"=> method,
#            "seed" => seed, 
#            "diffusion"=> diffusion,
#            "sys_dim"=> sys_dim,
#            "obs_dim"=> obs_dim, 
#            "obs_un"=> obs_un,
#            "nanl"=> nanl,
#            "tanl"=> tanl,
#            "lag"=> lag,
#            "shift"=> shift,
#            "h"=> h,
#            "N_ens"=> N_ens, 
#            "state_infl"=> round(state_infl, 2)
#            }
#    
#    fname = "./data/" + method + "_hybrid/" + method + "_hybrid_smoother_l96_state_benchmark_mda_" + str(mda) + "_seed_" +\
#            str(seed).zfill(2) + "_diffusion_" + str(float(diffusion)).ljust(4, "0") + "_sys_dim_" + str(sys_dim) +\
#            "_obs_dim_" + str(obs_dim) + "_obs_un_" + str(obs_un).ljust(4, "0") + "_nanl_" +\
#            str(nanl).zfill(3) + "_tanl_" + str(tanl).zfill(3) + "_h_" + str(h).ljust(4, "0") + \
#            "_lag_" + str(lag).zfill(3) + "_shift_" + str(shift).zfill(3) +\
#            "_N_ens_" + str(N_ens).zfill(3) + "_state_inflation_" + str(round(state_infl, 2)).ljust(4, "0") + ".txt"
#
#    picwrite(data, fname)
#    return(args)
#
#########################################################################################################################
#
#
#def hybrid_param(args):
#    # Define experiment parameters
#
#    [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
#    # load the timeseries and associated parameters
#    tmp = picopen(time_series)
#    diffusion = tmp["diffusion"]
#    f = tmp["f"]
#    tanl = tmp["tanl"]
#    h = 0.01
#
#    # number of discrete forecast steps
#    f_steps = int(tanl / h)
#
#    # unpack the observations and the initial true state of the dynamic variables
#    obs = np.squeeze(tmp["obs"])
#    init = obs[:, 0]
#
#    # define the state dynamic state dimension and the extended state parameters to be estimated
#    state_dim = len(init)
#    sys_dim = state_dim
#    param_truth = np.array([f])
#    sys_dim = state_dim + len(param_truth)
#
#    # define kwargs
#    kwargs = {
#              "dx_dt"=> dx_dt,
#              "f_steps"=> f_steps,
#              "step_model"=> step_model, 
#              "h"=> h,
#              "diffusion"=> diffusion,
#              "shift"=> shift,
#              "mda"=> False,
#              "state_dim"=> state_dim,
#              "param_infl"=> param_infl,
#              "param_wlk"=> param_wlk
#             }
#
#    # number of analyses
#    nanl = 4500 
#
#    # set seed 
#    np.random.seed(seed)
#    
#    # define the initial ensembles
#    ens = np.random.multivariate_normal(init, np.eye(state_dim), size=N_ens).transpose()
#
#    if len(param_truth) > 1:
#        param_ens = np.random.multivariate_normal(np.squeeze(param_truth), np.diag(param_truth * param_err)**2, size=N_ens)
#    else:
#        param_ens = np.reshape(np.random.normal(np.squeeze(param_truth), scale=np.squeeze(param_truth)*param_err, size=N_ens), [1, N_ens])
#
#    # defined the extended state ensemble
#    ens = np.concatenate([ens, param_ens], axis=0)
#
#    # obs and truth indices are defined in absolute time from time zero, we pad the obs sequence with 
#    # 2*lag-length states to align the posterior, filter and forecast statistics
#    obs = obs[:, :nanl + 3 * lag + 1]
#    truth = copy.copy(obs)
#    
#    # define the observation operator for the dynamic state variables -- note, the param_truth is not part of the
#    # truth state vector below, this is stored separately
#    H = alternating_obs_operator(state_dim, obs_dim) 
#    obs = H @ obs + obs_un * np.random.standard_normal(np.shape(obs))
#    
#    # define the associated time-invariant observation error covariance
#    obs_cov = obs_un**2 * np.eye(obs_dim)
#
#    # define the observation operator on the extended state, used for the ensemble
#    H = alternating_obs_operator(sys_dim, obs_dim, **kwargs)
#
#    # create storage for the forecast and analysis statistics
#    fore_rmse = np.zeros(nanl + 3 * lag + 1)
#    filt_rmse = np.zeros(nanl + 3 * lag + 1)
#    anal_rmse = np.zeros(nanl + 3 * lag + 1)
#    param_rmse = np.zeros(nanl + 3 * lag + 1)
#    
#    fore_spread = np.zeros(nanl + 3 * lag + 1)
#    filt_spread = np.zeros(nanl + 3 * lag + 1)
#    anal_spread = np.zeros(nanl + 3 * lag + 1)
#    param_spread = np.zeros(nanl + 3 * lag + 1)
#
#    # perform an initial spin for the smoothed re-analyzed first prior estimate 
#    # using observations over absolute times 1 to lag, resulting in ens at time 0+shift
#    kwargs["spin"] = True
#    analysis = lag_shift_smoother_hybrid(method, ens, H, obs[:, 1: lag + 1], obs_cov, state_infl, **kwargs)
#    ens = analysis["ens"]
#
#    # reset the spin pameter for the regular assimilation cycle
#    kwargs["spin"] = False
#    
#    # we will run through nanl + 2 * lag total analyses but discard the last-lag forecast values and
#    # first-lag posterior values so that the statistics align on the same time points after the spin
#    for i in range(shift + 1, nanl +  lag + 2, shift):
#        # perform assimilation of the DAW
#        # we use the observation windo from time zero to time lag
#        analysis = lag_shift_smoother_hybrid(method, ens, H, obs[:, i: i + lag], obs_cov, state_infl, **kwargs)
#        ens = analysis["ens"]
#        fore = analysis["fore"]
#        filt = analysis["filt"]
#        post = analysis["post"]
#        
#        for j in range(shift):
#            # compute the forecast, filter and analysis statistics
#            # forward index the true state by 1, because the sequence starts at time zero for which there is no
#            # observation
#            # indices for the forecast, filter, analysis and truth arrays are in absolute time, not relative
#            fore_rmse[i + lag - shift + j], fore_spread[i + lag - shift + j] = analyze_ensemble(fore[:state_dim, :, j], 
#                                                                                    truth[:, i + lag - shift + j])
#            
#            filt_rmse[i + lag - shift + j], filt_spread[i + lag - shift + j] = analyze_ensemble(filt[:state_dim, :, j], 
#                                                                                    truth[:, i + lag - shift + j])
#            
#            anal_rmse[i - 1 + j], anal_spread[i - 1 + j] = analyze_ensemble(post[:state_dim, :, j], 
#                                                                                truth[:, i - 1 + j])
#
#            param_rmse[i - 1 + j], param_spread[i - 1 + j] = analyze_ensemble_parameters(post[state_dim:, :, j], 
#                                                                                param_truth)
#
#            
#    # cut the statistics so that they align on the same time points
#    fore_rmse = fore_rmse[lag + 1: lag + 1 + nanl]
#    fore_spread = fore_spread[lag + 1: lag + 1 + nanl]
#    filt_rmse = filt_rmse[lag + 1: lag + 1 + nanl]
#    filt_spread = filt_spread[lag + 1: lag + 1 + nanl]
#    anal_rmse = anal_rmse[lag + 1: lag + 1 + nanl]
#    anal_spread = anal_spread[lag + 1: lag + 1 + nanl]
#    param_rmse = param_rmse[lag + 1: lag + 1 + nanl]
#    param_spread = param_spread[lag + 1: lag + 1 + nanl]
#
#    data = {
#            "fore_rmse"=> fore_rmse,
#            "filt_rmse"=> filt_rmse,
#            "anal_rmse"=> anal_rmse,
#            "param_rmse"=> param_rmse,
#            "fore_spread"=> fore_spread,
#            "filt_spread"=> filt_spread,
#            "anal_spread"=> anal_spread,
#            "param_spread"=> param_spread,
#            "seed" => seed, 
#            "method"=> method,
#            "diffusion"=> diffusion,
#            "sys_dim"=> sys_dim,
#            "state_dim"=> state_dim,
#            "obs_dim"=> obs_dim, 
#            "obs_un"=> obs_un,
#            "param_err"=> param_err,
#            "param_wlk"=> param_wlk,
#            "nanl"=> nanl,
#            "tanl"=> tanl,
#            "lag"=> lag,
#            "shift"=> shift,
#            "h"=> h,
#            "N_ens"=> N_ens, 
#            "state_infl"=> round(state_infl, 2),
#            "param_infl"=> round(param_infl, 2)
#            }
#    
#    fname = "./data/" + method + "_hybrid/" + method + "_hybrid_smoother_l96_param_benchmark_seed_" +\
#            str(seed).zfill(2) + "_diffusion_" + str(float(diffusion)).ljust(4, "0") + "_sys_dim_" + str(sys_dim) +\
#            "_state_dim_" + str(state_dim)+ "_obs_dim_" + str(obs_dim) + "_obs_un_" + str(obs_un).ljust(4, "0") + \
#            "_param_err_" + str(param_err).ljust(4, "0") + "_param_wlk_" + str(param_wlk).ljust(6, "0") +\
#            "_nanl_" + str(nanl).zfill(3) + "_tanl_" + str(tanl).zfill(3) + "_h_" + str(h).ljust(4, "0") + \
#            "_lag_" + str(lag).zfill(3) + "_shift_" + str(shift).zfill(3) +\
#            "_N_ens_" + str(N_ens).zfill(3) + "_state_infl_" + str(round(state_infl, 2)).ljust(4, "0") +\
#            "_param_infl_" + str(round(param_infl, 2)).ljust(4, "0") + ".txt"
#
#    picwrite(data, fname)
#    return(args)
#
#########################################################################################################################
