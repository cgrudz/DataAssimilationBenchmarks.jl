##############################################################################################
module ParallelExperimentDriver
##############################################################################################
# imports and exports
using ..DataAssimilationBenchmarks
export ensemble_filter_adaptive_inflation, ensemble_filter_param, classic_ensemble_state,
       classic_ensemble_param, single_iteration_ensemble_state, iterative_ensemble_state 

##############################################################################################
# Utility methods and definitions
##############################################################################################

path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"

##############################################################################################
# Filters
##############################################################################################
# compare adaptive inflation methods

"""
    args, exp = ensemble_filter_adaptive_inflation()

Constucts a parameter map and experiment wrapper for sensitivity test of adaptive inflation.
"""
function ensemble_filter_adaptive_inflation()

    exp = DataAssimilationBenchmarks.FilterExps.ensemble_filter_state
    function wrap_exp(arguments)
        try
            exp(arguments)
        catch
            print("Error on " * string(arguments) * "\n")
        end
    end

    # set time series parameters
    seed      = 123
    h         = 0.05
    state_dim = 40
    tanl      = 0.05
    nanl      = 6500
    spin      = 1500
    diffusion = 0.00
    F         = 8.0

    # generate truth twin time series
    GenerateTimeSeries.L96_time_series(
                                       (
                                         seed      = seed,
                                         h         = h,
                                         state_dim = state_dim,
                                         tanl      = tanl,
                                         nanl      = nanl,
                                         spin      = spin,
                                         diffusion = diffusion,
                                         F         = F,
                                        )
                                      )

    # define load path to time series
    time_series = path * "L96_time_series_seed_" * lpad(seed, 4, "0") *
                         "_dim_" * lpad(state_dim, 2, "0") *
                         "_diff_" * rpad(diffusion, 5, "0") *
                         "_F_" * lpad(F, 4, "0") *
                         "_tanl_" * rpad(tanl, 4, "0") *
                         "_nanl_" * lpad(nanl, 5, "0") *
                         "_spin_" * lpad(spin, 4, "0") *
                         "_h_" * rpad(h, 5, "0") *
                         ".jld2"

    # define ranges for filter parameters
    methods = ["enkf-n-primal", "enkf-n-primal-ls", "enkf-n-dual"]
    seed = 1234
    obs_un = 1.0
    obs_dim = 40
    N_enss = 15:3:42
    s_infls = [1.0]
    nanl = 4000
    γ = 1.0
    
    # load the experiments
    args = Vector{Any}()
    for method in methods
        for N_ens in N_enss
            for s_infl in s_infls
                tmp = (
                       time_series = time_series,
                       method = method,
                       seed = seed,
                       nanl = nanl,
                       obs_un = obs_un,
                       obs_dim = obs_dim,
                       γ = γ,
                       N_ens = N_ens,
                       s_infl = s_infl
                      )
                push!(args, tmp)
            end
        end
    end
    return args, wrap_exp
end


##############################################################################################
# parameter estimation, different random walk and inflation settings for parameter resampling

"""
    args, exp = ensemble_filter_param()

Constucts a parameter map and experiment wrapper for sensitivity test of parameter estimation.

Ensemble schemes sample the forcing parameter for the Lorenz-96 system and vary the random
walk parameter model for its time evolution / search over parameter space.
"""
function ensemble_filter_param()

    exp = DataAssimilationBenchmarks.FilterExps.ensemble_filter_param
    function wrap_exp(arguments)
        try
            exp(arguments)
        catch
            print("Error on " * string(arguments) * "\n")
        end
    end

    # set time series parameters
    seed      = 123
    h         = 0.05
    state_dim = 40
    tanl      = 0.05
    nanl      = 7500
    spin      = 1500
    diffusion = 0.00
    F         = 8.0

    # generate truth twin time series
    GenerateTimeSeries.L96_time_series(
                                       (
                                         seed      = seed,
                                         h         = h,
                                         state_dim = state_dim,
                                         tanl      = tanl,
                                         nanl      = nanl,
                                         spin      = spin,
                                         diffusion = diffusion,
                                         F         = F,
                                        )
                                      )

    # define load path to time series
    time_series = path * "L96_time_series_seed_" * lpad(seed, 4, "0") *
                         "_dim_" * lpad(state_dim, 2, "0") *
                         "_diff_" * rpad(diffusion, 5, "0") *
                         "_F_" * lpad(F, 4, "0") *
                         "_tanl_" * rpad(tanl, 4, "0") *
                         "_nanl_" * lpad(nanl, 5, "0") *
                         "_spin_" * lpad(spin, 4, "0") *
                         "_h_" * rpad(h, 5, "0") *
                         ".jld2"

    # define ranges for filter parameters
    methods = ["etkf", "mlef-transform"]
    seed = 1234
    obs_un = 1.0
    obs_dim = 40
    p_err = 0.03
    p_wlks = [0.0000, 0.0001, 0.0010, 0.0100]
    N_enss = 15:3:42
    s_infls = [1.0]
    nanl = 4000
    s_infls = LinRange(1.0, 1.10, 11)
    p_infls = LinRange(1.0, 1.05, 6)
    γ = 1.0

    # load the experiments
    args = Vector{Any}() 
    for method in methods
        for p_wlk in p_wlks
            for N_ens in N_enss
                for s_infl in s_infls
                    for p_infl in p_infls
                        tmp = (
                               time_series = time_series,
                               method = method,
                               seed = seed,
                               nanl = nanl,
                               obs_un = obs_un,
                               obs_dim = obs_dim,
                               γ = γ,
                               p_err = p_err,
                               p_wlk = p_wlk,
                               N_ens = N_ens,
                               s_infl = s_infl,
                               p_infl = p_infl
                              )
                        push!(args, tmp)
                    end
                end
            end
        end
    end
    return args, wrap_exp
end


##############################################################################################
# Classic smoothers
##############################################################################################

"""
    args, exp = classic_ensemble_state()

Constucts a parameter map and experiment wrapper for sensitivity test of nonlinear obs.

The ETKS / MLES estimators vary over different multiplicative inflation parameters, smoother
lag lengths and the nonlinearity of the observation operator.
"""
function classic_ensemble_state()

    exp = DataAssimilationBenchmarks.SmootherExps.classic_ensemble_state
    function wrap_exp(arguments)
        try
            exp(arguments)
        catch
            print("Error on " * string(arguments) * "\n")
        end
    end

    # set time series parameters
    seed      = 123
    h         = 0.05
    state_dim = 40
    tanl      = 0.05
    nanl      = 7500
    spin      = 1500
    diffusion = 0.00
    F         = 8.0

    # generate truth twin time series
    GenerateTimeSeries.L96_time_series(
                                       (
                                         seed      = seed,
                                         h         = h,
                                         state_dim = state_dim,
                                         tanl      = tanl,
                                         nanl      = nanl,
                                         spin      = spin,
                                         diffusion = diffusion,
                                         F         = F,
                                        )
                                      )

    # define load path to time series
    time_series = path * "L96_time_series_seed_" * lpad(seed, 4, "0") *
                         "_dim_" * lpad(state_dim, 2, "0") *
                         "_diff_" * rpad(diffusion, 5, "0") *
                         "_F_" * lpad(F, 4, "0") *
                         "_tanl_" * rpad(tanl, 4, "0") *
                         "_nanl_" * lpad(nanl, 5, "0") *
                         "_spin_" * lpad(spin, 4, "0") *
                         "_h_" * rpad(h, 5, "0") *
                         ".jld2"

    # define ranges for filter parameters
    methods = ["etks", "mles-transform"]
    seed = 1234
    lags = 1:3:52
    shifts = [1]
    gammas = Vector{Float64}(1:10)
    shift = 1
    obs_un = 1.0
    obs_dim = 40
    N_enss = 15:3:42
    s_infls = LinRange(1.0, 1.10, 11)
    nanl = 4000

    # load the experiments
    args = Vector{Any}() 
    for method in methods
        for γ in gammas
            for lag in lags
                for shift in shifts
                    for N_ens in N_enss
                        for s_infl in s_infls
                            tmp = (
                                   time_series = time_series,
                                   method = method,
                                   seed = seed,
                                   nanl = nanl,
                                   lag = lag,
                                   shift = shift,
                                   obs_un = obs_un,
                                   obs_dim = obs_dim,
                                   γ = γ,
                                   N_ens = N_ens,
                                   s_infl = s_infl
                                  )
                            push!(args, tmp)
                        end
                    end
                end
            end
        end
    end
    return args, wrap_exp
end

#############################################################################################

"""
    args, exp = ensemble_filter_adaptive_inflation()

Constucts a parameter map and experiment wrapper for sensitivity test of parameter estimation.

Ensemble schemes sample the forcing parameter for the Lorenz-96 system and vary the random
walk parameter model for its time evolution / search over parameter space.  Methods vary
the ETKS and MLES analysis, with different lag lengths, multiplicative inflation parameters,
and different pameter models.
"""
function classic_ensemble_param()
    
    exp = DataAssimilationBenchmarks.SmootherExps.classic_ensemble_param
    function wrap_exp(arguments)
        try
            exp(arguments)
        catch
            print("Error on " * string(arguments) * "\n")
        end
    end

    # set time series parameters
    seed      = 123
    h         = 0.05
    state_dim = 40
    tanl      = 0.05
    nanl      = 7500
    spin      = 1500
    diffusion = 0.00
    F         = 8.0

    # generate truth twin time series
    GenerateTimeSeries.L96_time_series(
                                       (
                                         seed      = seed,
                                         h         = h,
                                         state_dim = state_dim,
                                         tanl      = tanl,
                                         nanl      = nanl,
                                         spin      = spin,
                                         diffusion = diffusion,
                                         F         = F,
                                        )
                                      )

    # define load path to time series
    time_series = path * "L96_time_series_seed_" * lpad(seed, 4, "0") *
                         "_dim_" * lpad(state_dim, 2, "0") *
                         "_diff_" * rpad(diffusion, 5, "0") *
                         "_F_" * lpad(F, 4, "0") *
                         "_tanl_" * rpad(tanl, 4, "0") *
                         "_nanl_" * lpad(nanl, 5, "0") *
                         "_spin_" * lpad(spin, 4, "0") *
                         "_h_" * rpad(h, 5, "0") *
                         ".jld2"

    # define ranges for filter parameters
    methods = ["etks", "mles-transform"]
    seed = 1234
    lags = 1:3:52
    shifts = [1]
    gammas = [1.0]
    shift = 1
    obs_un = 1.0
    obs_dim = 40
    N_enss = 15:3:42
    p_err = 0.03
    p_wlks = [0.0000, 0.0001, 0.0010, 0.0100]
    s_infls = LinRange(1.0, 1.10, 11)
    p_infls = LinRange(1.0, 1.05, 6)
    nanl = 4000
    
    # load the experiments
    args = Vector{Any}() 
    for method in methods
        for lag in lags
            for γ in gammas
                for N_ens in N_enss
                    for p_wlk in p_wlks
                        for s_infl in s_infls
                            for p_infl in p_infls
                                tmp = (
                                       time_series = time_series,
                                       method = method,
                                       seed = seed,
                                       nanl = nanl,
                                       lag = lag,
                                       shift = shift,
                                       obs_un = obs_un,
                                       obs_dim = obs_dim,
                                       γ = γ,
                                       p_err = p_err,
                                       p_wlk = p_wlk,
                                       N_ens = N_ens,
                                       s_infl = s_infl,
                                       p_infl = p_infl
                                      )
                                push!(args, tmp)
                            end
                        end
                    end
                end
            end
        end
    end
   return args, wrap_exp 
end
    
    
#############################################################################################
# SIEnKS
#############################################################################################

function single_iteration_ensemble_state()
        
    exp = DataAssimilationBenchmarks.SmootherExps.single_iteration_ensemble_state
    function wrap_exp(arguments)
        try
            exp(arguments)
        catch
            print("Error on " * string(arguments) * "\n")
        end
    end

    # set time series parameters
    seed      = 123
    h         = 0.05
    state_dim = 40
    tanl      = 0.05
    nanl      = 7500
    spin      = 1500
    diffusion = 0.00
    F         = 8.0

    # generate truth twin time series
    GenerateTimeSeries.L96_time_series(
                                       (
                                         seed      = seed,
                                         h         = h,
                                         state_dim = state_dim,
                                         tanl      = tanl,
                                         nanl      = nanl,
                                         spin      = spin,
                                         diffusion = diffusion,
                                         F         = F,
                                        )
                                      )

    # define load path to time series
    time_series = path * "L96_time_series_seed_" * lpad(seed, 4, "0") *
                         "_dim_" * lpad(state_dim, 2, "0") *
                         "_diff_" * rpad(diffusion, 5, "0") *
                         "_F_" * lpad(F, 4, "0") *
                         "_tanl_" * rpad(tanl, 4, "0") *
                         "_nanl_" * lpad(nanl, 5, "0") *
                         "_spin_" * lpad(spin, 4, "0") *
                         "_h_" * rpad(h, 5, "0") *
                         ".jld2"

    # define ranges for filter parameters
    methods = ["etks"]
    seed = 1234
    lags = 1:3:52
    shifts = [1]
    gammas = [1.0]
    shift = 1
    obs_un = 1.0
    obs_dim = 40
    N_enss = 15:3:42
    s_infls = LinRange(1.0, 1.10, 11)
    nanl = 4000
    mdas = [false, true]

    # load the experiments
    args = Vector{Any}() 
    for mda in mdas
        for γ in gammas
            for method in methods
                for lag in lags
                    for shift in shifts
                        for N_ens in N_enss
                            for s_infl in s_infls
                                tmp = (
                                       time_series = time_series,
                                       method = method,
                                       seed = seed,
                                       nanl = nanl,
                                       lag = lag,
                                       shift = shift,
                                       mda = mda,
                                       obs_un = obs_un,
                                       obs_dim = obs_dim,
                                       γ = γ,
                                       N_ens = N_ens,
                                       s_infl = s_infl
                                      )
                                push!(args, tmp)
                            end
                        end
                    end
                end
            end
        end
    end
    return args, wrap_exp
end


#############################################################################################
# IEnKS
#############################################################################################

function iterative_ensemble_state()
        
    exp = DataAssimilationBenchmarks.SmootherExps.iterative_ensemble_state
    function wrap_exp(arguments)
        try
            exp(arguments)
        catch
            print("Error on " * string(arguments) * "\n")
        end
    end

    # set time series parameters
    seed      = 123
    h         = 0.05
    state_dim = 40
    tanl      = 0.05
    nanl      = 7500
    spin      = 1500
    diffusion = 0.00
    F         = 8.0

    # generate truth twin time series
    GenerateTimeSeries.L96_time_series(
                                       (
                                         seed      = seed,
                                         h         = h,
                                         state_dim = state_dim,
                                         tanl      = tanl,
                                         nanl      = nanl,
                                         spin      = spin,
                                         diffusion = diffusion,
                                         F         = F,
                                        )
                                      )

    # define load path to time series
    time_series = path * "L96_time_series_seed_" * lpad(seed, 4, "0") *
                         "_dim_" * lpad(state_dim, 2, "0") *
                         "_diff_" * rpad(diffusion, 5, "0") *
                         "_F_" * lpad(F, 4, "0") *
                         "_tanl_" * rpad(tanl, 4, "0") *
                         "_nanl_" * lpad(nanl, 5, "0") *
                         "_spin_" * lpad(spin, 4, "0") *
                         "_h_" * rpad(h, 5, "0") *
                         ".jld2"

    # define ranges for filter parameters
    methods = ["ienks-transform", "lin-ienks-transform"]
    seed = 1234
    lags = 1:3:52
    gammas = [1.0]
    shift = 1
    obs_un = 1.0
    obs_dim = 40
    N_enss = 15:3:42
    s_infls = LinRange(1.0, 1.10, 11)
    nanl = 4000
    mdas = [false, true]

    # load the experiments
    args = Vector{Any}() 
    for mda in mdas
        for γ in gammas
            for method in methods
                for lag in lags
                    for N_ens in N_enss
                        for s_infl in s_infls
                            tmp = (
                                   time_series = time_series,
                                   method = method,
                                   seed = seed,
                                   nanl = nanl,
                                   lag = lag,
                                   shift = shift,
                                   mda = mda,
                                   obs_un = obs_un,
                                   obs_dim = obs_dim,
                                   γ = γ,
                                   N_ens = N_ens,
                                   s_infl = s_infl
                                  )
                            push!(args, tmp)
                        end
                    end
                end
            end
        end
    end
    return args, exp
end


##############################################################################################
# end module

end
