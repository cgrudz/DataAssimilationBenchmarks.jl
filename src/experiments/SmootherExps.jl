##############################################################################################
module SmootherExps
##############################################################################################
# imports and exports
using Random, Distributions
using LinearAlgebra
using JLD2, HDF5
using ..DataAssimilationBenchmarks, ..ObsOperators, ..EnsembleKalmanSchemes, ..DeSolvers,
      ..L96, ..IEEE39bus
export classic_ensemble_state, classic_ensemble_param, single_iteration_ensemble_state,
       single_iteration_ensemble_param,
       iterative_ensemble_state, iterative_ensemble_param
##############################################################################################
# Main smoothing experiments, debugged and validated for use with schemes in methods directory
##############################################################################################
"""
    classic_ensemble_state((time_series::String, method::String, seed::Int64, nanl::Int64,
                            lag::Int64, shift::Int64, obs_un::Float64, obs_dim::Int64,
                            γ::Float64, N_ens::Int64, s_infl::Float64)::NamedTuple)

Classic ensemble Kalman smoother state estimation twin experiment.

NOTE: the classic scheme does not use multiple data assimilation and we hard code
`mda=false` in the function for consistency with the API of other methods.

Output from the experiment is saved in a dictionary of the form,

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "method" => method,
                            "seed"  => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "h" => h,
                            "N_ens" => N_ens,
                            "mda"  => mda,
                            "s_infl" => round(s_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

Experiment output is written to a directory defined by

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "-classic/"

where the file name is written dynamically according to the selected parameters as follows:

    method * "-classic_" * model *
             "_state_seed_" * lpad(seed, 4, "0") *
             "_diff_" * rpad(diffusion, 5, "0") *
             "_sysD_" * lpad(sys_dim, 2, "0") *
             "_obsD_" * lpad(obs_dim, 2, "0") *
             "_obsU_" * rpad(obs_un, 4, "0") *
             "_gamma_" * lpad(γ, 5, "0") *
             "_nanl_" * lpad(nanl, 5, "0") *
             "_tanl_" * rpad(tanl, 4, "0") *
             "_h_" * rpad(h, 4, "0") *
             "_lag_" * lpad(lag, 3, "0") *
             "_shift_" * lpad(shift, 3, "0") *
             "_mda_" * string(mda) *
             "_nens_" * lpad(N_ens, 3,"0") *
             "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
             ".jld2"
"""
function classic_ensemble_state((time_series, method, seed, nanl, lag, shift, obs_un, obs_dim,
                                 γ, N_ens, s_infl)::NamedTuple{
                               (:time_series,:method,:seed,:nanl,:lag,:shift,:obs_un,:obs_dim,
                                :γ,:N_ens,:s_infl),
                               <:Tuple{String,String,Int64,Int64,Int64,Int64,Float64,Int64,
                                       Float64,Int64,Float64}})
    # time the experiment
    t1 = time()

    # Define experiment parameters

    # define static mda parameter, not used for classic smoother
    mda = false

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    dx_params = ts["dx_params"]::ParamDict(Float64)
    tanl = ts["tanl"]::Float64
    model = ts["model"]::String

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # set the integration step size for the ensemble at 0.01 if an SDE, if deterministic
    # simply use the same step size as the observation model
    if diffusion > 0.0
        h = 0.01
    else
        h = ts["h"]
    end

    # define the dynamical model derivative for this experiment from the name
    # supplied in the time series
    if model == "L96"
        dx_dt = L96.dx_dt
    elseif model == "IEEE39bus"
        dx_dt = IEEE39bus.dx_dt
    end

    # define integration method
    step_model! = rk4_step!

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # set seed
    Random.seed!(seed)

    # define the initialization
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation range and truth reference solution
    obs = obs[:, 1:nanl + 3 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "dx_params" => dx_params,
                              "h" => h,
                              "diffusion" => diffusion,
                              "s_infl" => s_infl,
                              "γ" => γ,
                              "shift" => shift,
                              "mda" => mda
                             )

    # define the observation operator, observation error covariance and observations
    # with error, observation covariance operator taken as a uniform scaling by default,
    # can be changed in the definition below
    obs = H_obs(obs, obs_dim, kwargs)
    obs += obs_un * rand(Normal(), size(obs))
    obs_cov = obs_un^2.0 * I

    # check if there is a diffusion structure matrix
    if haskey(ts, "diff_mat")
        kwargs["diff_mat"] = ts["diff_mat"]
    end

    # create storage for the forecast and analysis statistics, indexed in relative time
    # the first index corresponds to time 1
    # last index corresponds to index nanl + 3 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)

    fore_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)

    # posterior array of length lag + shift will be loaded with filtered states as they
    # arrive in the DAW, with the shifting time index
    post = Array{Float64}(undef, sys_dim, N_ens, lag + shift)

    # we will run through nanl total analyses, i ranges in the absolute analysis-time index,
    # we perform assimilation of the observation window from time 2 to time nanl + 1 + lag
    # at increments of shift starting at time 2 because of no observations at time 1
    # only the interval 2 : nanl + 1 is stored later for all statistics
    for i in 2: shift : nanl + 1 + lag
        kwargs["posterior"] = post
        # observations indexed in absolute time
        analysis = ls_smoother_classic(method, ens, obs[:, i: i + shift - 1],
                                       H_obs, obs_cov, kwargs)
        ens = analysis["ens"]::Array{Float64}
        fore = analysis["fore"]::Array{Float64}
        filt = analysis["filt"]::Array{Float64}
        post = analysis["post"]::Array{Float64}

        for j in 1:shift
            # compute the forecast, filter and analysis statistics
            # indices for the forecast, filter, analysis statistics storage index starts
            # at absolute time 1, truth index starts at absolute time 1
            fore_rmse[i + j - 1],
            fore_spread[i + j - 1] = analyze_ens(
                                                 fore[:, :, j],
                                                 truth[:, i + j - 1]
                                                )
            filt_rmse[i + j - 1],
            filt_spread[i + j - 1] = analyze_ens(
                                                 filt[:, :, j],
                                                 truth[:, i + j - 1]
                                                )

            # we analyze the posterior states to be be discarded in the non-overlapping DAWs
            if shift == lag
                # for the shift=lag, all states are analyzed and discared,
                # no dummy past states are used, truth follows times minus 1
                # from the filter and forecast stastistics
                post_rmse[i + j - 2],
                post_spread[i + j - 2] = analyze_ens(
                                                     post[:, :, j],
                                                     truth[:, i + j - 2]
                                                    )

            elseif i > lag
                # for lag > shift, we wait for the dummy lag-1-total posterior states to be
                # cycled out, the first posterior starts with the first prior at time 1,
                # later discarded to align stats
                post_rmse[i - lag + j - 1],
                post_spread[i - lag + j - 1] = analyze_ens(
                                                           post[:, :, j],
                                                           truth[:, i - lag + j - 1]
                                                          )
            end
        end
    end

    # cut the statistics so that they align on the same absolute time points
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    post_rmse = post_rmse[2: nanl + 1]
    post_spread = post_spread[2: nanl + 1]

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "method" => method,
                            "seed"  => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "h" => h,
                            "N_ens" => N_ens,
                            "mda"  => mda,
                            "s_infl" => round(s_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "-classic/"
    name = method * "-classic_" * model *
                    "_state_seed_" * lpad(seed, 4, "0") *
                    "_diff_" * rpad(diffusion, 5, "0") *
                    "_sysD_" * lpad(sys_dim, 2, "0") *
                    "_obsD_" * lpad(obs_dim, 2, "0") *
                    "_obsU_" * rpad(obs_un, 4, "0") *
                    "_gamma_" * lpad(γ, 5, "0") *
                    "_nanl_" * lpad(nanl, 5, "0") *
                    "_tanl_" * rpad(tanl, 4, "0") *
                    "_h_" * rpad(h, 4, "0") *
                    "_lag_" * lpad(lag, 3, "0") *
                    "_shift_" * lpad(shift, 3, "0") *
                    "_mda_" * string(mda) *
                    "_nens_" * lpad(N_ens, 3,"0") *
                    "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
                    ".jld2"

    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


##############################################################################################
"""
    classic_ensemble_param((time_series::String, method::String, seed::Int64, nanl::Int64,
                            lag::Int64, shift::Int64, obs_un::Float64, obs_dim::Int64,
                            γ::Float64, p_err::Float64, p_wlk::Float64, N_ens::Int64,
                            s_infl::Float64, s_infl::Float64})::NamedTuple)

Classic ensemble Kalman smoother joint state-parameter estimation twin experiment.

NOTE: the classic scheme does not use multiple data assimilation and we hard code `mda=false`
in the function for consistency with other methods.

Output from the experiment is saved in a dictionary of the form,

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "param_rmse" => para_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "param_spread" => para_spread,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "param_truth" => param_truth,
                            "sys_dim" => sys_dim,
                            "state_dim" => state_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "p_err" => p_err,
                            "p_wlk" => p_wlk,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2),
                            "p_infl"  => round(p_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

Experiment output is written to a directory defined by

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "-classic/"

where the file name is written dynamically according to the selected parameters as follows:

    method * "-classic_" * model *
             "_state_seed_" * lpad(seed, 4, "0") *
             "_diff_" * rpad(diffusion, 5, "0") *
             "_sysD_" * lpad(sys_dim, 2, "0") *
             "_obsD_" * lpad(obs_dim, 2, "0") *
             "_obsU_" * rpad(obs_un, 4, "0") *
             "_gamma_" * lpad(γ, 5, "0") *
             "_nanl_" * lpad(nanl, 5, "0") *
             "_tanl_" * rpad(tanl, 4, "0") *
             "_h_" * rpad(h, 4, "0") *
             "_lag_" * lpad(lag, 3, "0") *
             "_shift_" * lpad(shift, 3, "0") *
             "_mda_" * string(mda) *
             "_nens_" * lpad(N_ens, 3,"0") *
             "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
             ".jld2"
"""
function classic_ensemble_param((time_series, method, seed, nanl, lag, shift, obs_un, obs_dim,
                                 γ, p_err, p_wlk, N_ens, s_infl, p_infl)::NamedTuple{
                               (:time_series,:method,:seed,:nanl,:lag,:shift,:obs_un,:obs_dim,
                                :γ,:p_err,:p_wlk,:N_ens,:s_infl,:p_infl),
                               <:Tuple{String,String,Int64,Int64,Int64,Int64,Float64,Int64,
                                           Float64,Float64,Float64,Int64,Float64,Float64}})
    # time the experiment
    t1 = time()

    # define static mda parameter, not used for classic smoother
    mda=false

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    dx_params = ts["dx_params"]::ParamDict(Float64)
    tanl = ts["tanl"]::Float64
    model = ts["model"]::String

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # set the integration step size for the ensemble at 0.01 if an SDE, if deterministic
    # simply use the same step size as the observation model
    if diffusion > 0.0
        h = 0.01
    else
        h = ts["h"]
    end

    # define the dynamical model derivative for this experiment from the name
    # supplied in the time series
    if model == "L96"
        dx_dt = L96.dx_dt
    elseif model == "IEEE39bus"
        dx_dt = IEEE39bus.dx_dt
    end

    # define integration method
    step_model! = rk4_step!

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # set seed
    Random.seed!(seed)

    # define the initialization
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    if model == "L96"
        param_truth = pop!(dx_params, "F")
    elseif model == "IEEE39bus"
        param_truth = [pop!(dx_params, "H"); pop!(dx_params, "D")]
        param_truth = param_truth[:]
    end

    state_dim = length(init)
    sys_dim = state_dim + length(param_truth)

    # define the initial ensemble
    ens = rand(MvNormal(init, I), N_ens)

    # extend this by the parameter ensemble
    # note here the covariance is supplied such that the standard deviation is a percent
    # of the parameter value
    param_ens = rand(MvNormal(param_truth[:],
                              diagm(param_truth[:] * p_err).^2.0),
                              N_ens
                    )

    # define the extended state ensemble
    ens = [ens; param_ens]

    # define the observation sequence where we map the true state into the observation space
    # and perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 3 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "h" => h,
                              "diffusion" => diffusion,
                              "dx_params" => dx_params,
                              "γ" => γ,
                              "state_dim" => state_dim,
                              "p_wlk" => p_wlk,
                              "s_infl" => s_infl,
                              "p_infl" => p_infl,
                              "shift" => shift,
                              "mda" => mda
                             )

    # define the observation operator, observation error covariance and observations
    # with error observation covariance operator taken as a uniform scaling by default,
    # can be changed in the definition below
    obs = H_obs(obs, obs_dim, kwargs)
    obs += obs_un * rand(Normal(), size(obs))
    obs_cov = obs_un^2.0 * I

    # we define the parameter sample as the key name and index
    # of the extended state vector pair, to be loaded in the
    # ensemble integration step
    if model == "L96"
        param_sample = Dict("F" => [41:41])
    elseif model == "IEEE39bus"
        param_sample = Dict("H" => [21:30], "D" => [31:40])
    end
    kwargs["param_sample"] = param_sample

    # create storage for the forecast and analysis statistics, indexed in relative time
    # first index corresponds to time 1, last index corresponds to index nanl + 3 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    para_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)

    fore_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    para_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)

    # posterior array of length lag + shift will be loaded with filtered states as they
    # arrive in the DAW, with the shifting time index
    post = Array{Float64}(undef, sys_dim, N_ens, lag + shift)

    # we will run through nanl total analyses, i ranges in the absolute analysis-time index,
    # we perform assimilation of the observation window from time 2 to time nanl + 1 + lag
    # at increments of shift starting at time 2 because of no observations at time 1
    # only the interval 2 : nanl + 1 is stored later for all statistics
    for i in 2: shift : nanl + 1 + lag
        kwargs["posterior"] = post
        # observations indexed in absolute time
        analysis = ls_smoother_classic(
                                       method, ens, obs[:, i: i + shift - 1],
                                       H_obs, obs_cov, kwargs
                                      )
        ens = analysis["ens"]::Array{Float64}
        fore = analysis["fore"]::Array{Float64}
        filt = analysis["filt"]::Array{Float64}
        post = analysis["post"]::Array{Float64}

        for j in 1:shift
            # compute the forecast, filter and analysis statistics
            # indices for the forecast, filter, analysis statistics storage index starts
            # at absolute time 1, truth index starts at absolute time 1
            fore_rmse[i + j - 1],
            fore_spread[i + j - 1] = analyze_ens(
                                                 fore[1:state_dim, :, j],
                                                 truth[:, i + j - 1]
                                                )
            filt_rmse[i + j - 1],
            filt_spread[i + j - 1] = analyze_ens(
                                                 filt[1:state_dim, :, j],
                                                 truth[:, i + j - 1]
                                                )

            # analyze the posterior states that will be discarded in the non-overlapping DAWs
            if shift == lag
                # for the shift=lag, all states are analyzed and discared,
                # no dummy past states are used truth follows times minus 1 from the
                # filter and forecast stastistics
                post_rmse[i + j - 2],
                post_spread[i + j - 2] = analyze_ens(
                                                     post[1:state_dim, :, j],
                                                     truth[:, i + j - 2]
                                                    )

                para_rmse[i + j - 2],
                para_spread[i + j - 2] = analyze_ens_param(
                                                           post[state_dim + 1: end,:, j],
                                                           param_truth
                                                          )
            elseif i > lag
                # for lag > shift, we wait for the dummy lag-1-total posterior states
                # to be cycled out the first posterior starts with the first prior at time 1,
                # later discarded to align stats
                post_rmse[i - lag + j - 1],
                post_spread[i - lag + j - 1] = analyze_ens(
                                                           post[1:state_dim, :, j],
                                                           truth[:, i - lag + j - 1]
                                                          )

                para_rmse[i - lag + j - 1],
                para_spread[i - lag + j - 1] =
                                             analyze_ens_param(
                                                               post[state_dim + 1: end, :, j],
                                                               param_truth
                                                              )
            end
        end
    end

    # cut the statistics so that they align on the same absolute time points
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    post_rmse = post_rmse[2: nanl + 1]
    post_spread = post_spread[2: nanl + 1]
    para_rmse = para_rmse[2: nanl + 1]
    para_spread = para_spread[2: nanl + 1]

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "param_rmse" => para_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "param_spread" => para_spread,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "param_truth" => param_truth,
                            "sys_dim" => sys_dim,
                            "state_dim" => state_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "p_err" => p_err,
                            "p_wlk" => p_wlk,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2),
                            "p_infl"  => round(p_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "-classic/"
    name = method * "-classic_" * model *
                    "_param_seed_" * lpad(seed, 4, "0") *
                    "_diff_" * rpad(diffusion, 5, "0") *
                    "_sysD_" * lpad(sys_dim, 2, "0") *
                    "_obsD_" * lpad(obs_dim, 2, "0") *
                    "_obsU_" * rpad(obs_un, 4, "0") *
                    "_gamma_" * lpad(γ, 5, "0") *
                    "_paramE_" * rpad(p_err, 4, "0") *
                    "_paramW_" * rpad(p_wlk, 6, "0") *
                    "_nanl_" * lpad(nanl, 5, "0") *
                    "_tanl_" * rpad(tanl, 4, "0") *
                    "_h_" * rpad(h, 4, "0") *
                    "_lag_" * lpad(lag, 3, "0") *
                    "_shift_" * lpad(shift, 3, "0") *
                    "_mda_" * string(mda) *
                    "_nens_" * lpad(N_ens, 3,"0") *
                    "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
                    "_paramInfl_" * rpad(round(p_infl, digits=2), 4, "0") *
                    ".jld2"

    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


##############################################################################################
"""
    single_iteration_ensemble_state((time_series::String, method::String, seed::Int64,
                                     nanl::Int64, lag::Int64, shift::Int64, mda::Bool,
                                     obs_un::Float64, obs_dim::Int64, γ::Float64,
                                     N_ens::Int64, s_infl::Float64})::NamedTuple)

SIEnKS state estimation twin experiment.

Output from the experiment is saved in a dictionary of the form,

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

Experiment output is written to a directory defined by

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "-single-iteration/"

where the file name is written dynamically according to the selected parameters as follows:

    method * "-single-iteration_" * model *
             "_state_seed_" * lpad(seed, 4, "0") *
             "_diff_" * rpad(diffusion, 5, "0") *
             "_sysD_" * lpad(sys_dim, 2, "0") *
             "_obsD_" * lpad(obs_dim, 2, "0") *
             "_obsU_" * rpad(obs_un, 4, "0") *
             "_gamma_" * lpad(γ, 5, "0") *
             "_nanl_" * lpad(nanl, 5, "0") *
             "_tanl_" * rpad(tanl, 4, "0") *
             "_h_" * rpad(h, 4, "0") *
             "_lag_" * lpad(lag, 3, "0") *
             "_shift_" * lpad(shift, 3, "0") *
             "_mda_" * string(mda) *
             "_nens_" * lpad(N_ens, 3,"0") *
             "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
             ".jld2"
"""
function single_iteration_ensemble_state((time_series, method, seed, nanl, lag, shift, mda,
                                          obs_un, obs_dim, γ, N_ens, s_infl)::NamedTuple{
                                        (:time_series,:method,:seed,:nanl,:lag,:shift,:mda,
                                         :obs_un,:obs_dim,:γ,:N_ens,:s_infl),
                                        <:Tuple{String,String,Int64,Int64,Int64,Int64,Bool,
                                                Float64,Int64,Float64,Int64,Float64}})

    # time the experiment
    t1 = time()

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    dx_params = ts["dx_params"]::ParamDict(Float64)
    tanl = ts["tanl"]::Float64
    model = ts["model"]::String

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # set the integration step size for the ensemble at 0.01 if an SDE, if deterministic
    # simply use the same step size as the observation model
    if diffusion > 0.0
        h = 0.01
    else
        h = ts["h"]
    end

    # define the dynamical model derivative for this experiment from the name
    # supplied in the time series
    if model == "L96"
        dx_dt = L96.dx_dt
    elseif model == "IEEE39bus"
        dx_dt = IEEE39bus.dx_dt
    end

    # define integration method
    step_model! = rk4_step!

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # number of discrete shift windows within the lag window
    n_shifts = convert(Int64, lag / shift)

    # set seed
    Random.seed!(seed)

    # define the initialization
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation sequence where we map the true state into the observation
    # space andperturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 3 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "dx_params" => dx_params,
                              "h" => h,
                              "diffusion" => diffusion,
                              "γ" => γ,
                              "shift" => shift,
                              "s_infl" => s_infl,
                              "mda" => mda
                             )

    # define the observation operator, observation error covariance and observations
    # with error observation covariance operator taken as a uniform scaling by default,
    # can be changed in the definition below
    obs = H_obs(obs, obs_dim, kwargs)
    obs += obs_un * rand(Normal(), size(obs))
    obs_cov = obs_un^2.0 * I

    # check if there is a diffusion structure matrix
    if haskey(ts, "diff_mat")
        kwargs["diff_mat"] = ts["diff_mat"]
    end

    # create storage for the forecast and analysis statistics, indexed in relative time
    # first index corresponds to time 1, last index corresponds to index nanl + 3 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)

    fore_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)

    # perform an initial spin for the smoothed re-analyzed first prior estimate while
    # handling new observations with a filtering step to prevent divergence of the
    # forecast for long lags
    spin = true
    kwargs["spin"] = spin
    posterior = Array{Float64}(undef, sys_dim, N_ens, shift)
    kwargs["posterior"] = posterior

    # we will run through nanl + 2 * lag total observations but discard the last-lag
    # forecast values and first-lag posterior values so that the statistics align on
    # the same time points after the spin
    for i in 2: shift : nanl + lag + 1
        # perform assimilation of the DAW
        # we use the observation window from current time +1 to current time +lag
        if mda
            # NOTE: mda spin weights only take lag equal to an integer multiple of shift
            if spin
                # for the first rebalancing step, all observations are new
                # and get fully assimilated, observation weights are given with
                # respect to a special window in terms of the number of times the
                # observation will be assimilated
                obs_weights = []
                for n in 1:n_shifts
                    obs_weights = [obs_weights; ones(shift) * n]
                end
                kwargs["obs_weights"] = Array{Float64}(obs_weights)
                kwargs["reb_weights"] = ones(lag)

            elseif i <= lag
                # if still processing observations from the spin cycle,
                # deal with special weights given by the number of times
                # the observation is assimilated
                n_complete =  (i - 2) / shift
                n_incomplete = n_shifts - n_complete

                # the leading terms have weights that are based upon the number of times
                # that the observation will be assimilated < n_shifts total times as in
                # the stable algorithm
                obs_weights = []
                for n in n_shifts - n_incomplete + 1 : n_shifts
                    obs_weights = [obs_weights; ones(shift) * n]
                end
                for n in 1 : n_complete
                    obs_weights = [obs_weights; ones(shift) * n_shifts]
                end
                kwargs["obs_weights"] = Array{Float64}(obs_weights)
                reb_weights = []

                # the rebalancing weights are specially constructed as above
                for n in 1:n_incomplete
                    reb_weights = [reb_weights; ones(shift) * n / (n + n_complete)]
                end
                for n in n_incomplete + 1 : n_shifts
                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
                end
                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)

            else
                # equal weights as all observations are assimilated n_shifts total times
                kwargs["obs_weights"] = ones(lag) * n_shifts

                # rebalancing weights are constructed in steady state
                reb_weights = []
                for n in 1:n_shifts
                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
                end
                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)
            end
        end

        # peform the analysis
        analysis = ls_smoother_single_iteration(
                                                method, ens, obs[:, i: i + lag - 1],
                                                H_obs, obs_cov, kwargs
                                               )
        ens = analysis["ens"]::Array{Float64}
        fore = analysis["fore"]::Array{Float64}
        filt = analysis["filt"]::Array{Float64}
        post = analysis["post"]::Array{Float64}

        if spin
            for j in 1:lag
                # compute forecast and filter statistics for spin period
                fore_rmse[i - 1 + j],
                fore_spread[i - 1 + j] = analyze_ens(
                                                     fore[:, :, j],
                                                     truth[:, i - 1 + j]
                                                    )

                filt_rmse[i - 1 + j],
                filt_spread[i - 1 + j] = analyze_ens(
                                                     filt[:, :, j],
                                                     truth[:, i - 1 + j]
                                                    )
            end

            for j in 1:shift
                # compute the reanalyzed prior and the shift-forward forecasted reanalysis
                post_rmse[i - 2 + j],
                post_spread[i - 2 + j] = analyze_ens(
                                                     post[:, :, j],
                                                     truth[:, i - 2 + j]
                                                    )
            end

            # turn off the initial spin period, continue on the normal assimilation cycle
            spin = false
            kwargs["spin"] = spin

        else
            for j in 1:shift
                # compute the forecast, filter and analysis statistics
                # indices for the forecast, filter, analysis and truth are in absolute time,
                # forecast / filter stats computed beyond the first lag period for the spin
                fore_rmse[i + lag - 1 - shift + j],
                fore_spread[i + lag - 1 - shift + j] =
                                                analyze_ens(
                                                            fore[:, :, j],
                                                            truth[:, i + lag - 1 - shift + j]
                                                           )

                filt_rmse[i + lag - 1 - shift + j],
                filt_spread[i + lag - 1 - shift + j] =
                                                analyze_ens(
                                                            filt[:, :, j],
                                                            truth[:, i + lag - 1 - shift + j]
                                                           )

                # analysis statistics computed beyond the first shift
                post_rmse[i - 2 + j],
                post_spread[i - 2 + j] = analyze_ens(
                                                     post[:, :, j],
                                                     truth[:, i - 2 + j]
                                                    )
            end
        end
    end

    # cut the statistics so that they align on the same absolute time points
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    post_rmse = post_rmse[2: nanl + 1]
    post_spread = post_spread[2: nanl + 1]

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "-single-iteration/"
    name = method * "-single-iteration_" * model *
                    "_state_seed_" * lpad(seed, 4, "0") *
                    "_diff_" * rpad(diffusion, 5, "0") *
                    "_sysD_" * lpad(sys_dim, 2, "0") *
                    "_obsD_" * lpad(obs_dim, 2, "0") *
                    "_obsU_" * rpad(obs_un, 4, "0") *
                    "_gamma_" * lpad(γ, 5, "0") *
                    "_nanl_" * lpad(nanl, 5, "0") *
                    "_tanl_" * rpad(tanl, 4, "0") *
                    "_h_" * rpad(h, 4, "0") *
                    "_lag_" * lpad(lag, 3, "0") *
                    "_shift_" * lpad(shift, 3, "0") *
                    "_mda_" * string(mda) *
                    "_nens_" * lpad(N_ens, 3,"0") *
                    "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
                    ".jld2"

    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


##############################################################################################
"""
    single_iteration_ensemble_param((time_series::String, method::String, seed:Int64,
                                     nanl::Int64, lag::Int64, shift::Int64, mda::Bool,
                                     obs_un::Float64, obs_dim::Int64, γ::Float64,
                                     p_err::Float64, p_wlk::Float64, N_ens::Int64,
                                     s_infl::Float64, p_infl::Float64)::NamedTuple)

SIEnKS joint state-parameter estimation twin experiment.

Output from the experiment is saved in a dictionary of the form,

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "param_rmse" => para_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "param_spread" => para_spread,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "param_truth" => param_truth,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "p_wlk" => p_wlk,
                            "p_infl" => p_infl,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2),
                            "p_infl" => round(p_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

Experiment output is written to a directory defined by

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "-single-iteration/"

where the file name is written dynamically according to the selected parameters as follows:

    method * "-single-iteration_" * model *
             "_param_seed_" * lpad(seed, 4, "0") *
             "_diff_" * rpad(diffusion, 5, "0") *
             "_sysD_" * lpad(sys_dim, 2, "0") *
             "_obsD_" * lpad(obs_dim, 2, "0") *
             "_obsU_" * rpad(obs_un, 4, "0") *
             "_gamma_" * lpad(γ, 5, "0") *
             "_paramE_" * rpad(p_err, 4, "0") *
             "_paramW_" * rpad(p_wlk, 6, "0") *
             "_nanl_" * lpad(nanl, 5, "0") *
             "_tanl_" * rpad(tanl, 4, "0") *
             "_h_" * rpad(h, 4, "0") *
             "_lag_" * lpad(lag, 3, "0") *
             "_shift_" * lpad(shift, 3, "0") *
             "_mda_" * string(mda) *
             "_nens_" * lpad(N_ens, 3,"0") *
             "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
             "_paramInfl_" * rpad(round(p_infl, digits=2), 4, "0") *
             ".jld2"
"""
function single_iteration_ensemble_param((time_series, method, seed, nanl, lag, shift, mda,
                                          obs_un, obs_dim, γ, p_err, p_wlk, N_ens, s_infl,
                                          p_infl)::NamedTuple{
                                        (:time_series,:method,:seed,:nanl,:lag,:shift,:mda,
                                         :obs_un,:obs_dim,:γ,:p_err,:p_wlk,:N_ens,:s_infl,
                                         :p_infl),
                                        <:Tuple{String,String,Int64,Int64,Int64,Int64,Bool,
                                                Float64,Int64,Float64,Float64,Float64,Int64,
                                                Float64,Float64}})

    # time the experiment
    t1 = time()

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    dx_params = ts["dx_params"]::ParamDict(Float64)
    tanl = ts["tanl"]::Float64
    model = ts["model"]::String

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # set the integration step size for the ensemble at 0.01 if an SDE, if deterministic
    # simply use the same step size as the observation model
    if diffusion > 0.0
        h = 0.01
    else
        h = ts["h"]
    end

    # define the dynamical model derivative for this experiment from the name
    # supplied in the time series
    if model == "L96"
        dx_dt = L96.dx_dt
    elseif model == "IEEE39bus"
        dx_dt = IEEE39bus.dx_dt
    end

    # define integration method
    step_model! = rk4_step!

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # number of discrete shift windows within the lag window
    n_shifts = convert(Int64, lag / shift)

    # set seed
    Random.seed!(seed)

    # define the initialization
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    if model == "L96"
        param_truth = pop!(dx_params, "F")
    elseif model == "IEEE39bus"
        param_truth = [pop!(dx_params, "H"); pop!(dx_params, "D")]
        param_truth = param_truth[:]
    end

    state_dim = length(init)
    sys_dim = state_dim + length(param_truth)

    # define the initial ensemble
    ens = rand(MvNormal(init, I), N_ens)

    # extend this by the parameter ensemble
    # note here the covariance is supplied such that the standard deviation is a percent
    # of the parameter value
    param_ens = rand(MvNormal(param_truth[:],
                              diagm(param_truth[:] * p_err).^2.0), N_ens)

    # define the extended state ensemble
    ens = [ens; param_ens]

    obs = obs[:, 1:nanl + 3 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "dx_params" => dx_params,
                              "h" => h,
                              "diffusion" => diffusion,
                              "γ" => γ,
                              "state_dim" => state_dim,
                              "shift" => shift,
                              "p_wlk" => p_wlk,
                              "s_infl" => s_infl,
                              "p_infl" => p_infl,
                              "mda" => mda
                             )

    # define the observation operator, observation error covariance and observations
    # with error observation covariance operator taken as a uniform scaling by default,
    # can be changed in the definition below
    obs = H_obs(obs, obs_dim, kwargs)
    obs += obs_un * rand(Normal(), size(obs))
    obs_cov = obs_un^2.0 * I

    # we define the parameter sample as the key name and index
    # of the extended state vector pair, to be loaded in the
    # ensemble integration step
    if model == "L96"
        param_sample = Dict("F" => [41:41])
    elseif model == "IEEE39bus"
        param_sample = Dict("H" => [21:30], "D" => [31:40])
    end
    kwargs["param_sample"] = param_sample

    # create storage for the forecast and analysis statistics, indexed in relative time
    # first index corresponds to time 1, last index corresponds to index nanl + 3 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    para_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)

    fore_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    para_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)

    # perform an initial spin for the smoothed re-analyzed first prior estimate while
    # handling new observations with a filtering step to prevent divergence of the forecast
    # for long lags
    spin = true
    kwargs["spin"] = spin
    posterior = zeros(sys_dim, N_ens, shift)
    kwargs["posterior"] = posterior

    # we will run through nanl + 2 * lag total analyses but discard the last-lag
    # forecast values and first-lag posterior values so that the statistics align on
    # the same time points after the spin
    for i in 2: shift : nanl + lag + 1
        # perform assimilation of the DAW
        # we use the observation window from current time +1 to current time +lag
        if mda
            # NOTE: mda spin weights only take lag equal to an integer multiple of shift
            if spin
                # for the first rebalancing step, all observations are new
                # and get fully assimilated, observation weights are given with
                # respect to a special window in terms of the number of times the
                # observation will be assimilated
                obs_weights = []
                for n in 1:n_shifts
                    obs_weights = [obs_weights; ones(shift) * n]
                end
                kwargs["obs_weights"] = Array{Float64}(obs_weights)
                kwargs["reb_weights"] = ones(lag)

            elseif i <= lag
                # if still processing observations from the spin cycle,
                # deal with special weights given by the number of times
                # the observation is assimilated
                n_complete =  (i - 2) / shift
                n_incomplete = n_shifts - n_complete

                # the leading terms have weights that are based upon the number of times
                # that the observation will be assimilated < n_shifts total times as in
                # the stable algorithm
                obs_weights = []
                for n in n_shifts - n_incomplete + 1 : n_shifts
                    obs_weights = [obs_weights; ones(shift) * n]
                end
                for n in 1 : n_complete
                    obs_weights = [obs_weights; ones(shift) * n_shifts]
                end
                kwargs["obs_weights"] = Array{Float64}(obs_weights)
                reb_weights = []

                # the rebalancing weights are specially constructed as above
                for n in 1:n_incomplete
                    reb_weights = [reb_weights; ones(shift) * n / (n + n_complete)]
                end
                for n in n_incomplete + 1 : n_shifts
                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
                end
                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)

            else
                # equal weights as all observations are assimilated n_shifts total times
                kwargs["obs_weights"] = ones(lag) * n_shifts

                # rebalancing weights are constructed in steady state
                reb_weights = []
                for n in 1:n_shifts
                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
                end
                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)
            end
        end

        # peform the analysis
        analysis = ls_smoother_single_iteration(
                                                method,
                                                ens,
                                                obs[:, i: i + lag - 1],
                                                H_obs,
                                                obs_cov,
                                                kwargs
                                               )
        ens = analysis["ens"]::Array{Float64}
        fore = analysis["fore"]::Array{Float64}
        filt = analysis["filt"]::Array{Float64}
        post = analysis["post"]::Array{Float64}

        if spin
            for j in 1:lag
                # compute forecast and filter statistics for the spin period
                fore_rmse[i - 1 + j],
                fore_spread[i - 1 + j] = analyze_ens(
                                                     fore[1:state_dim, :, j],
                                                          truth[:, i - 1 + j]
                                                    )

                filt_rmse[i - 1 + j],
                filt_spread[i - 1 + j] = analyze_ens(
                                                     filt[1:state_dim, :, j],
                                                          truth[:, i - 1 + j]
                                                    )

            end

            for j in 1:shift
                # compute the reanalyzed prior and the shift-forward forecasted reanalysis
                post_rmse[i - 2 + j],
                post_spread[i - 2 + j] = analyze_ens(
                                                     post[1:state_dim, :, j],
                                                     truth[:, i - 2 + j]
                                                    )

                para_rmse[i - 2 + j],
                para_spread[i - 2 + j] = analyze_ens_param(
                                                           post[state_dim+1:end, :, j],
                                                           param_truth
                                                          )

            end

            # turn off the initial spin period, continue on the normal assimilation cycle
            spin = false
            kwargs["spin"] = spin

        else
            for j in 1:shift
                # compute the forecast, filter and analysis statistics
                # indices for the forecast, filter, analysis and truth arrays are
                # in absolute time, forecast / filter stats computed beyond the
                # first lag period for the spin
                fore_rmse[i + lag - 1 - shift + j],
                fore_spread[i + lag - 1 - shift + j] =
                                                analyze_ens(
                                                            fore[1:state_dim, :, j],
                                                            truth[:, i + lag - 1 - shift + j]
                                                           )

                filt_rmse[i + lag - 1 - shift + j],
                filt_spread[i + lag - 1 - shift + j] =
                                                analyze_ens(
                                                            filt[1:state_dim, :, j],
                                                            truth[:, i + lag - 1 - shift + j]
                                                           )

                # analysis statistics computed beyond the first shift
                post_rmse[i - 2 + j],
                post_spread[i - 2 + j] = analyze_ens(
                                                     post[1:state_dim, :, j],
                                                          truth[:, i - 2 + j]
                                                    )

                para_rmse[i - 2 + j],
                para_spread[i - 2 + j] = analyze_ens_param(
                                                           post[state_dim+1:end, :, j],
                                                           param_truth
                                                          )

            end

        end

    end

    # cut the statistics so that they align on the same absolute time points
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    post_rmse = post_rmse[2: nanl + 1]
    post_spread = post_spread[2: nanl + 1]
    para_rmse = para_rmse[2: nanl + 1]
    para_spread = para_spread[2: nanl + 1]

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "param_rmse" => para_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "param_spread" => para_spread,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "param_truth" => param_truth,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "p_wlk" => p_wlk,
                            "p_infl" => p_infl,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2),
                            "p_infl" => round(p_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "-single-iteration/"
    name = method * "-single-iteration_" * model *
                    "_param_seed_" * lpad(seed, 4, "0") *
                    "_diff_" * rpad(diffusion, 5, "0") *
                    "_sysD_" * lpad(sys_dim, 2, "0") *
                    "_obsD_" * lpad(obs_dim, 2, "0") *
                    "_obsU_" * rpad(obs_un, 4, "0") *
                    "_gamma_" * lpad(γ, 5, "0") *
                    "_paramE_" * rpad(p_err, 4, "0") *
                    "_paramW_" * rpad(p_wlk, 6, "0") *
                    "_nanl_" * lpad(nanl, 5, "0") *
                    "_tanl_" * rpad(tanl, 4, "0") *
                    "_h_" * rpad(h, 4, "0") *
                    "_lag_" * lpad(lag, 3, "0") *
                    "_shift_" * lpad(shift, 3, "0") *
                    "_mda_" * string(mda) *
                    "_nens_" * lpad(N_ens, 3,"0") *
                    "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
                    "_paramInfl_" * rpad(round(p_infl, digits=2), 4, "0") *
                    ".jld2"


    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


##############################################################################################
"""
    iterative_ensemble_state((time_series::String, method::String, seed::Int64, nanl::Int64,
                              lag::Int64, shift::Int64, mda::Bool, obs_un::Float64,
                              obs_dim::Int64, γ::Float64, N_ens::Int64,
                              s_infl::Float64)::NamedTuple)

4DEnVAR state estimation twin experiment using the IEnKS formalism.

Output from the experiment is saved in a dictionary of the form,

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "iteration_sequence" => iteration_sequence,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

Experiment output is written to a directory defined by

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "/"

where the file name is written dynamically according to the selected parameters as follows:

    method * "_" * model *
             "_state_seed_" * lpad(seed, 4, "0") *
             "_diff_" * rpad(diffusion, 5, "0") *
             "_sysD_" * lpad(sys_dim, 2, "0") *
             "_obsD_" * lpad(obs_dim, 2, "0") *
             "_obsU_" * rpad(obs_un, 4, "0") *
             "_gamma_" * lpad(γ, 5, "0") *
             "_nanl_" * lpad(nanl, 5, "0") *
             "_tanl_" * rpad(tanl, 4, "0") *
             "_h_" * rpad(h, 4, "0") *
             "_lag_" * lpad(lag, 3, "0") *
             "_shift_" * lpad(shift, 3, "0") *
             "_mda_" * string(mda) *
             "_nens_" * lpad(N_ens, 3,"0") *
             "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
             ".jld2"
"""
function iterative_ensemble_state((time_series, method, seed, nanl, lag, shift, mda, obs_un,
                                   obs_dim,γ, N_ens, s_infl)::NamedTuple{
                                 (:time_series,:method,:seed,:nanl,:lag,:shift,:mda,:obs_un,
                                  :obs_dim,:γ,:N_ens,:s_infl),
                                 <:Tuple{String,String,Int64,Int64,Int64,Int64,Bool,Float64,
                                         Int64,Float64,Int64,Float64}})

    # time the experiment
    t1 = time()

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    dx_params = ts["dx_params"]::ParamDict(Float64)
    tanl = ts["tanl"]::Float64
    model = ts["model"]::String

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # set the integration step size for the ensemble at 0.01 if an SDE, if deterministic
    # simply use the same step size as the observation model
    if diffusion > 0.0
        h = 0.01
    else
        h = ts["h"]
    end

    # define the dynamical model derivative for this experiment from the name
    # supplied in the time series
    if model == "L96"
        dx_dt = L96.dx_dt
    elseif model == "IEEE39bus"
        dx_dt = IEEE39bus.dx_dt
    end

    # define integration method
    step_model! = rk4_step!

    # define the iterative smoother method HARD-CODED here
    ls_smoother_iterative = ls_smoother_gauss_newton

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # number of discrete shift windows within the lag window
    n_shifts = convert(Int64, lag / shift)

    # set seed
    Random.seed!(seed)

    # define the initialization
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    sys_dim = length(init)
    ens = rand(MvNormal(init, I), N_ens)

    # define the observation range and truth reference solution
    obs = obs[:, 1:nanl + 3 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "dx_params" => dx_params,
                              "h" => h,
                              "diffusion" => diffusion,
                              "γ" => γ,
                              "s_infl" => s_infl,
                              "shift" => shift,
                              "mda" => mda
                             )

    # define the observation operator, observation error covariance and observations
    # with error observation covariance operator taken as a uniform scaling by default,
    # can be changed in the definition below
    obs = H_obs(obs, obs_dim, kwargs)
    obs += obs_un * rand(Normal(), size(obs))
    obs_cov = obs_un^2.0 * I

    # check if there is a diffusion structure matrix
    if haskey(ts, "diff_mat")
        kwargs["diff_mat"] = ts["diff_mat"]
    end

    # create storage for the forecast and analysis statistics, indexed in relative time
    # first index corresponds to time 1, last index corresponds to index nanl + 3 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)

    fore_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)

    # create storage for the iteration sequence, where we will append the number
    # of iterations on the fly, due to the miss-match between the number of observations
    # and the number of analyses with shift > 1
    iteration_sequence = Vector{Float64}[]

    # create counter for the analyses
    m = 1

    # perform an initial spin for the smoothed re-analyzed first prior estimate while
    # handling new observations with a filtering step to prevent divergence of the
    # forecast for long lags
    spin = true
    kwargs["spin"] = spin
    posterior = zeros(sys_dim, N_ens, shift)
    kwargs["posterior"] = posterior

    # we will run through nanl + 2 * lag total observations but discard the
    # last-lag forecast values and first-lag posterior values so that the statistics
    # align on the same observation time points after the spin
    for i in 2: shift : nanl + lag + 1
        # perform assimilation of the DAW
        # we use the observation window from current time +1 to current time +lag
        if mda
            # NOTE: mda spin weights are only designed for lag equal to an integer
            # multiple of shift
            if spin
                # for the first rebalancing step, all observations are new and get
                # fully assimilated observation weights are given with respect to a
                # special window in terms of the number of times the observation will
                # be assimilated
                obs_weights = []
                for n in 1:n_shifts
                    obs_weights = [obs_weights; ones(shift) * n]
                end
                kwargs["obs_weights"] = Array{Float64}(obs_weights)
                kwargs["reb_weights"] = ones(lag)

            elseif i <= lag
                # if still processing observations from the spin cycle,
                # deal with special weights given by the number of times the observation
                # is assimilated
                n_complete =  (i - 2) / shift
                n_incomplete = n_shifts - n_complete

                # the leading terms have weights that are based upon the number of times
                # that the observation will be assimilated < n_shifts total times as in
                # the stable algorithm
                obs_weights = []
                for n in n_shifts - n_incomplete + 1 : n_shifts
                    obs_weights = [obs_weights; ones(shift) * n]
                end
                for n in 1 : n_complete
                    obs_weights = [obs_weights; ones(shift) * n_shifts]
                end
                kwargs["obs_weights"] = Array{Float64}(obs_weights)
                reb_weights = []

                # the rebalancing weights are specially constructed as above
                for n in 1:n_incomplete
                    reb_weights = [reb_weights; ones(shift) * n / (n + n_complete)]
                end
                for n in n_incomplete + 1 : n_shifts
                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
                end
                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)

            else
                # otherwise equal weights as all observations are assimilated n_shifts
                # total times
                kwargs["obs_weights"] = ones(lag) * n_shifts

                # rebalancing weights are constructed in steady state
                reb_weights = []
                for n in 1:n_shifts
                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
                end
                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)
            end
        end

        if method[1:4] == "lin-"
            if spin
                # on the spin cycle, there are the standard number of iterations allowed
                # to warm up
                analysis = ls_smoother_iterative(method[5:end], ens, obs[:, i: i + lag - 1],
                                                 H_obs, obs_cov, kwargs)
            else
                # after this, the number of iterations allowed is set to one
                analysis = ls_smoother_iterative(method[5:end], ens, obs[:, i: i + lag - 1],
                                                 H_obs, obs_cov, kwargs, max_iter=1)
            end
        else
            analysis = ls_smoother_iterative(method, ens, obs[:, i: i + lag - 1],
                                             H_obs, obs_cov, kwargs)
        end
        ens = analysis["ens"]::Array{Float64}
        fore = analysis["fore"]::Array{Float64}
        filt = analysis["filt"]::Array{Float64}
        post = analysis["post"]::Array{Float64}
        iteration_sequence = [iteration_sequence; analysis["iterations"]]
        m+=1

        if spin
            for j in 1:lag
                # compute filter statistics on the first lag states during spin period
                filt_rmse[i - 1 + j],
                filt_spread[i - 1 + j] = analyze_ens(
                                                     filt[:, :, j],
                                                     truth[:, i - 1 + j]
                                                    )
            end
            for j in 1:lag+shift
                # compute the forecast statistics on the first lag+shift states during
                # the spin period
                fore_rmse[i - 1 + j],
                fore_spread[i - 1 + j] = analyze_ens(
                                                     fore[:, :, j],
                                                     truth[:, i - 1 + j]
                                                    )
            end
            for j in 1:shift
                # compute only the reanalyzed prior and the shift-forward forecasted
                # reanalysis
                post_rmse[i - 2 + j],
                post_spread[i - 2 + j] = analyze_ens(
                                                     post[:, :, j],
                                                     truth[:, i - 2 + j]
                                                    )
            end

            # turn off the initial spin period, continue hereafter on the normal
            # assimilation cycle
            spin = false
            kwargs["spin"] = spin

        else
            for j in 1:shift
                # compute the forecast, filter and analysis statistics
                # indices for the forecast, filter, analysis and truth arrays
                # are in absolute time, forecast / filter stats computed beyond
                # the first lag period for the spin
                fore_rmse[i + lag - 1 + j],
                fore_spread[i + lag - 1+ j] = analyze_ens(
                                                          fore[:, :, j],
                                                          truth[:, i + lag - 1 + j]
                                                         )

                filt_rmse[i + lag - 1 - shift + j],
                filt_spread[i + lag - 1 - shift + j] =
                                                analyze_ens(
                                                            filt[:, :, j],
                                                            truth[:, i + lag - 1 - shift + j]
                                                           )

                # analysis statistics computed beyond the first shift
                post_rmse[i - 2 + j],
                post_spread[i - 2 + j] = analyze_ens(
                                                     post[:, :, j],
                                                     truth[:, i - 2 + j]
                                                    )

            end

        end

    end

    # cut the statistics so that they align on the same absolute time points
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    post_rmse = post_rmse[2: nanl + 1]
    post_spread = post_spread[2: nanl + 1]
    iteration_sequence = Array{Float64}(iteration_sequence)

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "iteration_sequence" => iteration_sequence,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "/"
    name = method * "_" * model *
                    "_state_seed_" * lpad(seed, 4, "0") *
                    "_diff_" * rpad(diffusion, 5, "0") *
                    "_sysD_" * lpad(sys_dim, 2, "0") *
                    "_obsD_" * lpad(obs_dim, 2, "0") *
                    "_obsU_" * rpad(obs_un, 4, "0") *
                    "_gamma_" * lpad(γ, 5, "0") *
                    "_nanl_" * lpad(nanl, 5, "0") *
                    "_tanl_" * rpad(tanl, 4, "0") *
                    "_h_" * rpad(h, 4, "0") *
                    "_lag_" * lpad(lag, 3, "0") *
                    "_shift_" * lpad(shift, 3, "0") *
                    "_mda_" * string(mda) *
                    "_nens_" * lpad(N_ens, 3,"0") *
                    "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
                    ".jld2"

    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")

end


##############################################################################################
"""
    iterative_ensemble_param((time_series:String, method:String, seed::Int64, nanl::Int64,
                              lag::Int64, shift::Int64, mda::Bool, obs_un::Float64,
                              obs_dim::Int64, γ::Float64, p_err::Float64, p_wlk::Float64,
                              N_ens::Int64, s_infl::Float64, p_infl::Float64)::NamedTuple)

4DEnVAR joint state-parameter estimation twin experiment using the IEnKS formalism.

Output from the experiment is saved in a dictionary of the form,

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "param_rmse" => para_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "param_spread" => para_spread,
                            "iteration_sequence" => iteration_sequence,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "p_wlk" => p_wlk,
                            "p_infl" => p_infl,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2),
                            "p_infl" => round(p_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

Experiment output is written to a directory defined by

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "/"

where the file name is written dynamically according to the selected parameters as follows:

    method * "_" * model *
             "_param_seed_" * lpad(seed, 4, "0") *
             "_diff_" * rpad(diffusion, 5, "0") *
             "_sysD_" * lpad(sys_dim, 2, "0") *
             "_obsD_" * lpad(obs_dim, 2, "0") *
             "_obsU_" * rpad(obs_un, 4, "0") *
             "_gamma_" * lpad(γ, 5, "0") *
             "_paramE_" * rpad(p_err, 4, "0") *
             "_paramW_" * rpad(p_wlk, 6, "0") *
             "_nanl_" * lpad(nanl, 5, "0") *
             "_tanl_" * rpad(tanl, 4, "0") *
             "_h_" * rpad(h, 4, "0") *
             "_lag_" * lpad(lag, 3, "0") *
             "_shift_" * lpad(shift, 3, "0") *
             "_mda_" * string(mda) *
             "_nens_" * lpad(N_ens, 3,"0") *
             "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
             "_paramInfl_" * rpad(round(p_infl, digits=2), 4, "0") *
             ".jld2"
"""
function iterative_ensemble_param((time_series, method, seed, nanl, lag, shift, mda, obs_un,
                                   obs_dim, γ, p_err, p_wlk, N_ens, s_infl,
                                   p_infl)::NamedTuple{
                                 (:time_series,:method,:seed,:nanl,:lag,:shift,:mda,:obs_un,
                                  :obs_dim,:γ,:p_err,:p_wlk,:N_ens,:s_infl,:p_infl),
                                 <:Tuple{String,String,Int64,Int64,Int64,Int64,Bool,Float64,
                                         Int64,Float64,Float64,Float64,Int64,Float64,
                                         Float64}})

    # time the experiment
    t1 = time()

    # load the timeseries and associated parameters
    ts = load(time_series)::Dict{String,Any}
    diffusion = ts["diffusion"]::Float64
    dx_params = ts["dx_params"]::ParamDict(Float64)
    tanl = ts["tanl"]::Float64
    model = ts["model"]::String

    # define the observation operator HARD-CODED in this line
    H_obs = alternating_obs_operator

    # set the integration step size for the ensemble at 0.01 if an SDE, if deterministic
    # simply use the same step size as the observation model
    if diffusion > 0.0
        h = 0.01
    else
        h = ts["h"]
    end

    # define the dynamical model derivative for this experiment from the name
    # supplied in the time series
    if model == "L96"
        dx_dt = L96.dx_dt
    elseif model == "IEEE39bus"
        dx_dt = IEEE39bus.dx_dt
    end

    # define integration method
    step_model! = rk4_step!

    # define the iterative smoother method
    ls_smoother_iterative = ls_smoother_gauss_newton

    # number of discrete forecast steps
    f_steps = convert(Int64, tanl / h)

    # number of discrete shift windows within the lag window
    n_shifts = convert(Int64, lag / shift)

    # set seed
    Random.seed!(seed)

    # define the initialization
    obs = ts["obs"]::Array{Float64, 2}
    init = obs[:, 1]
    if model == "L96"
        param_truth = pop!(dx_params, "F")
    elseif model == "IEEE39bus"
        param_truth = [pop!(dx_params, "H"); pop!(dx_params, "D")]
        param_truth = param_truth[:]
    end

    state_dim = length(init)
    sys_dim = state_dim + length(param_truth)

    # define the initial ensemble
    ens = rand(MvNormal(init, I), N_ens)

    # extend this by the parameter ensemble
    # note here the covariance is supplied such that the standard deviation is a
    # percent of the parameter value
    param_ens = rand(MvNormal(param_truth[:], diagm(param_truth[:] * p_err).^2.0), N_ens)

    # define the extended state ensemble
    ens = [ens; param_ens]

    # define the observation range and truth reference solution
    obs = obs[:, 1:nanl + 3 * lag + 1]
    truth = copy(obs)

    # define kwargs
    kwargs = Dict{String,Any}(
                              "dx_dt" => dx_dt,
                              "f_steps" => f_steps,
                              "step_model" => step_model!,
                              "dx_params" => dx_params,
                              "h" => h,
                              "diffusion" => diffusion,
                              "γ" => γ,
                              "state_dim" => state_dim,
                              "shift" => shift,
                              "p_wlk" => p_wlk,
                              "s_infl" => s_infl,
                              "p_infl" => p_infl,
                              "mda" => mda
                             )

    # define the observation operator, observation error covariance and observations
    # with error observation covariance operator taken as a uniform scaling by default,
    # can be changed in the definition below
    obs = H_obs(obs, obs_dim, kwargs)
    obs += obs_un * rand(Normal(), size(obs))
    obs_cov = obs_un^2.0 * I

    # we define the parameter sample as the key name and index
    # of the extended state vector pair, to be loaded in the
    # ensemble integration step
    if model == "L96"
        param_sample = Dict("F" => [41:41])
    elseif model == "IEEE39bus"
        param_sample = Dict("H" => [21:30], "D" => [31:40])
    end
    kwargs["param_sample"] = param_sample

    # create storage for the forecast and analysis statistics, indexed in relative time
    # first index corresponds to time 1, last index corresponds to index nanl + 3 * lag + 1
    fore_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
    para_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)

    fore_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    filt_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    post_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
    para_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)

    # create storage for the iteration sequence, where we will append the number
    # of iterations on the fly, due to the miss-match between the number of observations
    # and the number of analyses with shift > 1
    iteration_sequence = Vector{Float64}[]

    # create counter for the analyses
    m = 1

    # perform an initial spin for the smoothed re-analyzed first prior estimate while
    # handling new observations with a filtering step to prevent divergence of the
    # forecast for long lags
    spin = true
    kwargs["spin"] = spin
    posterior = zeros(sys_dim, N_ens, shift)
    kwargs["posterior"] = posterior

    # we will run through nanl + 2 * lag total observations but discard the
    # last-lag forecast values and first-lag posterior values so that the statistics
    # align on the same observation time points after the spin
    for i in 2: shift : nanl + lag + 1
        # perform assimilation of the DAW
        # we use the observation window from current time +1 to current time +lag
        if mda
            # NOTE: mda spin weights only take lag equal to an integer multiple of shift
            if spin
                # all observations are new and get fully assimilated
                # observation weights are given in terms of the
                # number of times the observation will be assimilated
                obs_weights = []
                for n in 1:n_shifts
                    obs_weights = [obs_weights; ones(shift) * n]
                end
                kwargs["obs_weights"] = Array{Float64}(obs_weights)
                kwargs["reb_weights"] = ones(lag)

            elseif i <= lag
                # still processing observations from the spin cycle, deal with special weights
                # given by the number of times the observation is assimilated
                n_complete =  (i - 2) / shift
                n_incomplete = n_shifts - n_complete

                # the leading terms have weights that are based upon the number of times
                # that the observation will be assimilated < n_shifts total times as in
                # the stable algorithm
                obs_weights = []
                for n in n_shifts - n_incomplete + 1 : n_shifts
                    obs_weights = [obs_weights; ones(shift) * n]
                end
                for n in 1 : n_complete
                    obs_weights = [obs_weights; ones(shift) * n_shifts]
                end
                kwargs["obs_weights"] = Array{Float64}(obs_weights)
                reb_weights = []

                # the rebalancing weights are specially constructed as above
                for n in 1:n_incomplete
                    reb_weights = [reb_weights; ones(shift) * n / (n + n_complete)]
                end
                for n in n_incomplete + 1 : n_shifts
                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
                end
                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)

            else
                # equal weights as all observations are assimilated n_shifts total times
                kwargs["obs_weights"] = ones(lag) * n_shifts

                # rebalancing weights are constructed in steady state
                reb_weights = []
                for n in 1:n_shifts
                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
                end
                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)
            end
        end

        if method[1:4] == "lin-"
            if spin
                # on the spin cycle, a standard number of iterations allowed to warm up
                analysis = ls_smoother_iterative(method[5:end], ens, obs[:, i: i + lag - 1],
                                                 H_obs, obs_cov, kwargs)
            else
                # after this, the number of iterations allowed is set to one
                analysis = ls_smoother_iterative(method[5:end], ens, obs[:, i: i + lag - 1],
                                                 H_obs, obs_cov, kwargs, max_iter=1)
            end
        else
            analysis = ls_smoother_iterative(method, ens, obs[:, i: i + lag - 1],
                                             H_obs, obs_cov, kwargs)
        end
        ens = analysis["ens"]::Array{Float64}
        fore = analysis["fore"]::Array{Float64}
        filt = analysis["filt"]::Array{Float64}
        post = analysis["post"]::Array{Float64}
        iteration_sequence = [iteration_sequence; analysis["iterations"]]
        m+=1

        if spin
            for j in 1:lag
                # compute filter statistics on the first lag states during spin period
                filt_rmse[i - 1 + j],
                filt_spread[i - 1 + j] = analyze_ens(filt[1:state_dim, :, j],
                                                          truth[:, i - 1 + j])
            end
            for j in 1:lag+shift
                # compute the forecast statistics on the first lag+shift states
                # during the spin period
                fore_rmse[i - 1 + j],
                fore_spread[i - 1 + j] = analyze_ens(fore[1:state_dim, :, j],
                                                          truth[:, i - 1 + j])
            end
            for j in 1:shift
                # compute the reanalyzed prior and the shift-forward forecasted reanalysis
                post_rmse[i - 2 + j],
                post_spread[i - 2 + j] = analyze_ens(post[1:state_dim, :, j],
                                                          truth[:, i - 2 + j])

                para_rmse[i - 2 + j],
                para_spread[i - 2 + j] = analyze_ens_param(post[state_dim+1:end, :, j],
                                                                     param_truth)

            end

            # turn off the initial spin period, continue with the normal assimilation cycle
            spin = false
            kwargs["spin"] = spin

        else
            for j in 1:shift
                # compute the forecast, filter and analysis statistics
                # indices for the forecast, filter, analysis and truth are in absolute time,
                # forecast / filter stats computed beyond the first lag period for the spin
                fore_rmse[i + lag - 1 + j],
                fore_spread[i + lag - 1+ j] = analyze_ens(
                                                          fore[1:state_dim, :, j],
                                                          truth[:, i + lag - 1 + j]
                                                         )

                filt_rmse[i + lag - 1 - shift + j],
                filt_spread[i + lag - 1 - shift + j] =
                                                analyze_ens(
                                                            filt[1:state_dim, :, j],
                                                            truth[:, i + lag - 1 - shift + j]
                                                           )

                # analysis statistics computed beyond the first shift
                post_rmse[i - 2 + j],
                post_spread[i - 2 + j] = analyze_ens(
                                                     post[1:state_dim, :, j],
                                                     truth[:, i - 2 + j]
                                                    )

                para_rmse[i - 2 + j],
                para_spread[i - 2 + j] = analyze_ens_param(
                                                           post[state_dim+1:end, :, j],
                                                           param_truth
                                                          )

            end

        end

    end

    # cut the statistics so that they align on the same absolute time points
    fore_rmse = fore_rmse[2: nanl + 1]
    fore_spread = fore_spread[2: nanl + 1]
    filt_rmse = filt_rmse[2: nanl + 1]
    filt_spread = filt_spread[2: nanl + 1]
    post_rmse = post_rmse[2: nanl + 1]
    post_spread = post_spread[2: nanl + 1]
    para_rmse = para_rmse[2: nanl + 1]
    para_spread = para_spread[2: nanl + 1]
    iteration_sequence = Array{Float64}(iteration_sequence)

    data = Dict{String,Any}(
                            "fore_rmse" => fore_rmse,
                            "filt_rmse" => filt_rmse,
                            "post_rmse" => post_rmse,
                            "param_rmse" => para_rmse,
                            "fore_spread" => fore_spread,
                            "filt_spread" => filt_spread,
                            "post_spread" => post_spread,
                            "param_spread" => para_spread,
                            "iteration_sequence" => iteration_sequence,
                            "method" => method,
                            "seed" => seed,
                            "diffusion" => diffusion,
                            "dx_params" => dx_params,
                            "sys_dim" => sys_dim,
                            "obs_dim" => obs_dim,
                            "obs_un" => obs_un,
                            "γ" => γ,
                            "p_wlk" => p_wlk,
                            "p_infl" => p_infl,
                            "nanl" => nanl,
                            "tanl" => tanl,
                            "lag" => lag,
                            "shift" => shift,
                            "mda" => mda,
                            "h" => h,
                            "N_ens" => N_ens,
                            "s_infl" => round(s_infl, digits=2),
                            "p_infl" => round(p_infl, digits=2)
                           )

    if haskey(ts, "diff_mat")
        data["diff_mat"] = ts["diff_mat"]
    end

    path = pkgdir(DataAssimilationBenchmarks) * "/src/data/" * method * "/"
    name = method * "_" * model *
                    "_param_seed_" * lpad(seed, 4, "0") *
                    "_diff_" * rpad(diffusion, 5, "0") *
                    "_sysD_" * lpad(sys_dim, 2, "0") *
                    "_obsD_" * lpad(obs_dim, 2, "0") *
                    "_obsU_" * rpad(obs_un, 4, "0") *
                    "_gamma_" * lpad(γ, 5, "0") *
                    "_paramE_" * rpad(p_err, 4, "0") *
                    "_paramW_" * rpad(p_wlk, 6, "0") *
                    "_nanl_" * lpad(nanl, 5, "0") *
                    "_tanl_" * rpad(tanl, 4, "0") *
                    "_h_" * rpad(h, 4, "0") *
                    "_lag_" * lpad(lag, 3, "0") *
                    "_shift_" * lpad(shift, 3, "0") *
                    "_mda_" * string(mda) *
                    "_nens_" * lpad(N_ens, 3,"0") *
                    "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
                    "_paramInfl_" * rpad(round(p_infl, digits=2), 4, "0") *
                    ".jld2"

    save(path * name, data)
    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
end


##############################################################################################
# end module

end

##############################################################################################
# NOTE STILL DEBUGGING THIS EXPERIMENT
#function single_iteration_adaptive_state(args::Tuple{String,String,Int64,Int64,Int64,Bool,Float64,Int64,
#                                                     Float64,Int64,Float64};tail::Int64=3)
#
#    # time the experiment
#    t1 = time()
#
#    # Define experiment parameters
#    time_series, method, seed, lag, shift, mda, obs_un, obs_dim, γ, N_ens, s_infl = args
#
#    # load the timeseries and associated parameters
#    ts = load(time_series)::Dict{String,Any}
#    diffusion = ts["diffusion"]::Float64
#    f = ts["F"]::Float64
#    tanl = ts["tanl"]::Float64
#    h = 0.01
#    dx_dt = L96.dx_dt
#    step_model! = rk4_step!
#
#    # number of discrete forecast steps
#    f_steps = convert(Int64, tanl / h)
#
#    # number of discrete shift windows within the lag window
#    n_shifts = convert(Int64, lag / shift)
#
#    # number of analyses
#    nanl = 2500
#
#    # set seed
#    Random.seed!(seed)
#
#    # define the initial ensembles
#    obs = ts["obs"]::Array{Float64, 2}
#    init = obs[:, 1]
#    sys_dim = length(init)
#    ens = rand(MvNormal(init, I), N_ens)
#
#    # define the observation sequence where we map the true state into the observation space and
#    # perturb by white-in-time-and-space noise with standard deviation obs_un
#    obs = obs[:, 1:nanl + 3 * lag + 1]
#    truth = copy(obs)
#
#    # define kwargs
#    kwargs = Dict{String,Any}(
#                "dx_dt" => dx_dt,
#                "f_steps" => f_steps,
#                "step_model" => step_model!,
#                "dx_params" => [f],
#                "h" => h,
#                "diffusion" => diffusion,
#                "γ" => γ,
#                "shift" => shift,
#                "mda" => mda
#                             )
#
#    if method == "etks_adaptive"
#        tail_spin = true
#        kwargs["tail_spin"] = tail_spin
#        kwargs["analysis"] = Array{Float64}(undef, sys_dim, N_ens, lag)
#        kwargs["analysis_innovations"] = Array{Float64}(undef, sys_dim, lag)
#    end
#
#    # define the observation operator, observation error covariance and observations with error
#    obs = H_obs(obs, obs_dim, kwargs)
#    obs += obs_un * rand(Normal(), size(obs))
#    obs_cov = obs_un^2.0 * I
#
#    # create storage for the forecast and analysis statistics, indexed in relative time
#    # the first index corresponds to time 1, last index corresponds to index nanl + 3 * lag + 1
#    fore_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
#    filt_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
#    post_rmse = Vector{Float64}(undef, nanl + 3 * lag + 1)
#
#    fore_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
#    filt_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
#    post_spread = Vector{Float64}(undef, nanl + 3 * lag + 1)
#
#    # perform an initial spin for the smoothed re-analyzed first prior estimate while handling
#    # new observations with a filtering step to prevent divergence of the forecast for long lags
#    spin = true
#    kwargs["spin"] = spin
#    posterior = Array{Float64}(undef, sys_dim, N_ens, shift)
#    kwargs["posterior"] = posterior
#
#    # we will run through nanl + 2 * lag total observations but discard the last-lag
#    # forecast values and first-lag posterior values so that the statistics align on
#    # the same time points after the spin
#    for i in 2: shift : nanl + lag + 1
#        # perform assimilation of the DAW
#        # we use the observation window from current time +1 to current time +lag
#        if mda
#            # NOTE: mda spin weights are only designed for lag equal to an integer multiple of shift
#            if spin
#                # for the first rebalancing step, all observations are new and get fully assimilated
#                # observation weights are given with respect to a special window in terms of the
#                # number of times the observation will be assimilated
#                obs_weights = []
#                for n in 1:n_shifts
#                    obs_weights = [obs_weights; ones(shift) * n]
#                end
#                kwargs["obs_weights"] = Array{Float64}(obs_weights)
#                kwargs["reb_weights"] = ones(lag)
#
#            elseif i <= lag
#                # if still processing observations from the spin cycle, deal with special weights
#                # given by the number of times the observation is assimilated
#                n_complete =  (i - 2) / shift
#                n_incomplete = n_shifts - n_complete
#
#                # the leading terms have weights that are based upon the number of times
#                # that the observation will be assimilated < n_shifts total times as in
#                # the stable algorithm
#                obs_weights = []
#                for n in n_shifts - n_incomplete + 1 : n_shifts
#                    obs_weights = [obs_weights; ones(shift) * n]
#                end
#                for n in 1 : n_complete
#                    obs_weights = [obs_weights; ones(shift) * n_shifts]
#                end
#                kwargs["obs_weights"] = Array{Float64}(obs_weights)
#                reb_weights = []
#
#                # the rebalancing weights are specially constructed as above
#                for n in 1:n_incomplete
#                    reb_weights = [reb_weights; ones(shift) * n / (n + n_complete)]
#                end
#                for n in n_incomplete + 1 : n_shifts
#                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
#                end
#                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)
#
#            else
#                # otherwise equal weights as all observations are assimilated n_shifts total times
#                kwargs["obs_weights"] = ones(lag) * n_shifts
#
#                # rebalancing weights are constructed in steady state
#                reb_weights = []
#                for n in 1:n_shifts
#                    reb_weights = [reb_weights; ones(shift) * n / n_shifts]
#                end
#                kwargs["reb_weights"] = 1.0 ./ Array{Float64}(reb_weights)
#            end
#        end
#
#        # peform the analysis
#        analysis = ls_smoother_single_iteration(method, ens, obs[:, i: i + lag - 1],
#                                                obs_cov, s_infl, kwargs)
#        ens = analysis["ens"]
#        fore = analysis["fore"]
#        filt = analysis["filt"]
#        post = analysis["post"]
#
#        if method == "etks_adaptive"
#            # cycle the analysis states for the new DAW
#            kwargs["analysis"] = analysis["anal"]
#            if tail_spin
#                # check if we have reached a long enough tail of innovation statistics
#                analysis_innovations = analysis["inno"]
#                if size(analysis_innovations, 2) / lag >= tail
#                    # if so, stop the tail spin
#                    tail_spin = false
#                    kwargs["tail_spin"] = tail_spin
#                end
#            end
#            # cycle  the analysis states for the new DAW
#            kwargs["analysis_innovations"] = analysis["inno"]
#        end
#
#        if spin
#            for j in 1:lag
#                # compute forecast and filter statistics on the first lag states during spin period
#                fore_rmse[i - 1 + j], fore_spread[i - 1 + j] = analyze_ens(fore[:, :, j],
#                                                                                    truth[:, i - 1 + j])
#
#                filt_rmse[i - 1 + j], filt_spread[i - 1 + j] = analyze_ens(filt[:, :, j],
#                                                                                    truth[:, i - 1 + j])
#            end
#
#            for j in 1:shift
#                # compute only the reanalyzed prior and the shift-forward forecasted reanalysis
#                post_rmse[i - 2 + j], post_spread[i - 2 + j] = analyze_ens(post[:, :, j], truth[:, i - 2 + j])
#            end
#
#            # turn off the initial spin period, continue hereafter on the normal assimilation cycle
#            spin = false
#            kwargs["spin"] = spin
#
#        else
#            for j in 1:shift
#                # compute the forecast, filter and analysis statistics
#                # indices for the forecast, filter, analysis and truth arrays are in absolute time,
#                # forecast / filter stats computed beyond the first lag period for the spin
#                fore_rmse[i + lag - 1 - shift + j],
#                fore_spread[i + lag - 1 - shift + j] = analyze_ens(fore[:, :, j],
#                                                                                    truth[:, i + lag - 1 - shift + j])
#
#                filt_rmse[i + lag - 1 - shift + j],
#                filt_spread[i + lag - 1 - shift + j] = analyze_ens(filt[:, :, j],
#                                                                                    truth[:, i + lag - 1 - shift + j])
#
#                # analysis statistics computed beyond the first shift
#                post_rmse[i - 2 + j], post_spread[i - 2 + j] = analyze_ens(post[:, :, j], truth[:, i - 2 + j])
#            end
#        end
#    end
#
#    # cut the statistics so that they align on the same absolute time points
#    fore_rmse = fore_rmse[2: nanl + 1]
#    fore_spread = fore_spread[2: nanl + 1]
#    filt_rmse = filt_rmse[2: nanl + 1]
#    filt_spread = filt_spread[2: nanl + 1]
#    post_rmse = post_rmse[2: nanl + 1]
#    post_spread = post_spread[2: nanl + 1]
#
#    data = Dict{String,Any}(
#            "fore_rmse" => fore_rmse,
#            "filt_rmse" => filt_rmse,
#            "post_rmse" => post_rmse,
#            "fore_spread" => fore_spread,
#            "filt_spread" => filt_spread,
#            "post_spread" => post_spread,
#            "method" => method,
#            "seed" => seed,
#            "diffusion" => diffusion,
#            "sys_dim" => sys_dim,
#            "obs_dim" => obs_dim,
#            "obs_un" => obs_un,
#            "γ" => γ,
#            "nanl" => nanl,
#            "tanl" => tanl,
#            "lag" => lag,
#            "shift" => shift,
#            "mda" => mda,
#            "h" => h,
#            "N_ens" => N_ens,
#            "s_infl" => round(s_infl, digits=2)
#           )
#
#    if method == "etks_adaptive"
#        data["tail"] = tail
#    end
#
#    path = "../data/" * method * "_single_iteration/"
#    name = method * "_single_iteration" *
#            "_l96_state_benchmark_seed_" * lpad(seed, 4, "0") *
#            "_diffusion_" * rpad(diffusion, 4, "0") *
#            "_sys_dim_" * lpad(sys_dim, 2, "0") *
#            "_obs_dim_" * lpad(obs_dim, 2, "0") *
#            "_obs_un_" * rpad(obs_un, 4, "0") *
#            "_gamma_" * lpad(γ, 5, "0") *
#            "_nanl_" * lpad(nanl, 5, "0") *
#            "_tanl_" * rpad(tanl, 4, "0") *
#            "_h_" * rpad(h, 4, "0") *
#            "_lag_" * lpad(lag, 3, "0") *
#            "_shift_" * lpad(shift, 3, "0") *
#            "_mda_" * string(mda) *
#            "_N_ens_" * lpad(N_ens, 3,"0") *
#            "_s_infl_" * rpad(round(s_infl, digits=2), 4, "0") *
#            ".jld2"
#
#
#    save(path * name, data)
#    print("Runtime " * string(round((time() - t1)  / 60.0, digits=4))  * " minutes\n")
#
#end
#
#
#########################################################################################################################

