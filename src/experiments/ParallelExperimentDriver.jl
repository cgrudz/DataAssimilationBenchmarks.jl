##############################################################################################
module ParallelExperimentDriver
##############################################################################################
# imports and exports
using ..DataAssimilationBenchmarks
using ..DataAssimilationBenchmarks.GenerateTimeSeries
using ..DataAssimilationBenchmarks.FilterExps
using ..DataAssimilationBenchmarks.SmootherExps
export adaptive_inflation_comp

##############################################################################################
# Utility methods and definitions
##############################################################################################

path = pkgdir(DataAssimilationBenchmarks) * "/src/data/time_series/"

##############################################################################################
# FIlters
##############################################################################################

function adaptive_inflation_comp()

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
    L96_time_series(
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
    N_enss = 15:2:43
    s_infls = [1.0]
    nanl = 4000
    
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
# filter_param
## [time_series, scheme, seed, nanl, obs_un, obs_dim, param_err, param_wlk, N_ens,
##  state_infl, param_infl] = args
#
#schemes = ["enkf", "etkf"]
#seed = 0
#obs_un = 1.0
#obs_dim = 40
#param_err = 0.03
#param_wlk = [0.0000, 0.0001, 0.0010, 0.0100]
#N_ens = 14:41
#state_infl = LinRange(1.0, 1.20, 21)
#param_infl = LinRange(1.0, 1.00, 1)
#nanl = 2500
#
## load the experiments
#args = Tuple[]
#for scheme in schemes
#    for wlk in param_wlk
#        for N in N_ens
#            for s_infl in state_infl
#                for p_infl in param_infl
#                    tmp = (time_series, scheme, seed, nanl, obs_un, obs_dim,
#                           param_err, wlk, N, s_infl, p_infl)
#                    push!(args, tmp)
#                end
#            end
#        end
#    end
#end
#
#experiment = FilterExps.filter_param
#
#
##############################################################################################
# Classic smoothers
##############################################################################################
## classic_state parallel run, arguments are
## time_series, method, seed, nanl, lag, shift, obs_un, obs_dim, γ, N_ens, state_infl = args
#
#schemes = ["etks"]
#seed = 0
#lags = 1:3:52
#shifts = [1]
##lags = [1, 2, 4, 8, 16, 32, 64]
##gammas = Array{Float64}(1:11)
#gammas = [1.0]
#shift = 1
#obs_un = 1.0
#obs_dim = 40
##N_ens = 15:2:41
#N_ens = [21]
##state_infl = [1.0]
#state_infl = LinRange(1.0, 1.10, 11)
#time_series = [ts1, ts2, ts3, ts4, ts5]
#nanl = 2500
#
## load the experiments
#args = Tuple[]
#for ts in time_series
#    for scheme in schemes
#        for γ in gammas
#            for l in 1:length(lags)
#                # optional definition of shifts in terms of the current lag parameter for a
#                # range of shift values
#                lag = lags[l]
#                #shifts = lags[1:l]
#                for shift in shifts
#                    for N in N_ens
#                        for s_infl in state_infl
#                            tmp = (ts, scheme, seed, nanl, lag, shift, obs_un, obs_dim,
#                                   γ, N, s_infl)
#                            push!(args, tmp)
#                        end
#                    end
#                end
#            end
#        end
#    end
#end
#
#
#experiment = SmootherExps.classic_state
##experiment = wrap_exp
#
#
##############################################################################################
## classic_param single run for debugging, arguments are
##  [time_series, method, seed, nanl, lag, shift, obs_un, obs_dim, param_err,
##   param_wlk, N_ens, state_infl, param_infl = args
#
#schemes = ["enks", "etks"]
#seed = 0
#lag = 1:5:51
#shift = 1
#obs_un = 1.0
#obs_dim = 40
#N_ens = 14:41
#param_err = 0.03
#param_wlk = [0.0000, 0.0001, 0.0010, 0.0100]
#state_infl = LinRange(1.0, 1.20, 21)
#param_infl = LinRange(1.0, 1.00, 1)
#nanl = 2500
#
## load the experiments
#args = Tuple[]
#for scheme in schemes
#    for l in lag
#        for N in N_ens
#            for wlk in param_wlk
#                for s_infl in state_infl
#                    for p_infl in param_infl
#                        tmp = (time_series, scheme, seed, nanl, l, shift, obs_un, obs_dim,
#                               param_err, wlk, N, s_infl, p_infl)
#                        push!(args, tmp)
#                    end
#                end
#            end
#        end
#    end
#end
#
#experiment = SmootherExps.classic_param
#
#
##############################################################################################
# Single iteration smoothers
##############################################################################################
## single iteration single run for degbugging, arguments are
## [time_series, method, seed, nanl, lag, shift, mda, obs_un, obs_dim,
##  N_ens, state_infl = args
#
#schemes = ["enks-n-primal"]
#seed = 0
##lags = 1:3:52
#lags = [1, 2, 4, 8, 16, 32, 64]
##gammas = Array{Float64}(1:11)
#gammas = [1.0]
##shift = 1
#obs_un = 1.0
#obs_dim = 40
##N_ens = 15:2:41
#N_ens = [21]
#state_infl = [1.0]
##state_infl = LinRange(1.0, 1.10, 11)
#time_series = [time_series_1, time_series_2]
#mdas = [false]
#nanl = 2500
#
## load the experiments
#args = Tuple[]
#for m in mdas
#    for ts in time_series
#        for γ in gammas
#            for scheme in schemes
#                for l in 1:length(lags)
#                    # optional definition of shift in terms of the current lag parameter
#                    # for a range of shift values
#                    lag = lags[l]
#                    shifts = lags[1:l]
#                    for shift in shifts
#                        for N in N_ens
#                            for s_infl in state_infl
#                                tmp = (ts, scheme, seed, nanl, lag, shift, m, obs_un,
#                                       obs_dim, γ, N, s_infl)
#                                push!(args, tmp)
#                            end
#                        end
#                    end
#                end
#            end
#        end
#    end
#end
#
## define the robust to failure wrapper
#function wrap_exp(arguments)
#    try
#        SmootherExps.single_iteration_state(arguments)
#    catch
#        print("Error on " * string(args) * "\n")
#    end
#end
#
#experiment = wrap_exp
#
##############################################################################################
# end module

end
