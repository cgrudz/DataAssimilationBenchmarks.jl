##############################################################################################
module ParallelExperimentDriver 
##############################################################################################
# imports and exports
using Distributed
@everywhere push!(LOAD_PATH, "/DataAssimilationBenchmarks/src/")
@everywhere push!(LOAD_PATH, "/DataAssimilationBenchmarks/src/methods")
@everywhere push!(LOAD_PATH, "/DataAssimilationBenchmarks/src/models")
@everywhere push!(LOAD_PATH, "/DataAssimilationBenchmarks/src/experiments")
@everywhere using JLD2, ..DataAssimilationBenchmarks, ..FilterExps, ..SmootherExps,
                  ..EnsembleKalmanSchemes, ..DeSolvers, ..L96, ..ParallelExperimentDriver
@everywhere export experiment 
#@everywhere export wrap_exp

##############################################################################################
## Timeseries data 
##############################################################################################
# observation timeseries to load into the experiment as truth twin, timeseries are named by
# the model, seed to initialize, the integration scheme used to produce, number of analyses,
# the spinup length, and the time length between observation points

ts1 = "../data/time_series/L96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.05" * 
      "_nanl_50000_spin_5000_h_0.050.jld2"
ts2 = "../data/time_series/L96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.10" *
      "_nanl_50000_spin_5000_h_0.050.jld2"
ts3 = "../data/time_series/L96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.15" *
      "_nanl_50000_spin_5000_h_0.050.jld2"
ts4 = "../data/time_series/L96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.20" *
      "_nanl_50000_spin_5000_h_0.050.jld2"
ts5 = "../data/time_series/L96_time_series_seed_0000_dim_40_diff_0.00_F_08.0)tanl_0.25" *
      "_nanl_50000_spin_5000_h_0.050.jld2"

##############################################################################################
## Experiment parameter generation 
##############################################################################################

##############################################################################################
# Filters
##############################################################################################
## filter_state 
#
## [time_series, scheme, seed, nanl, obs_un, obs_dim, N_ens, infl] = args
#
#schemes = ["enkf-n-primal", "enkf-n-primal-ls", "enkf-n-dual"]
#seed = 0
#obs_un = 1.0
#obs_dim = 40
#N_ens = 15:43
#infl = [1.0]#LinRange(1.0, 1.20, 21)
#nanl = 2500
#
## load the experiments
#args = Tuple[]
#for scheme in schemes
#    for N in N_ens
#        for α in infl
#            tmp = (time_series, scheme, seed, nanl, obs_un, obs_dim, N, α)
#            push!(args, tmp)
#        end
#    end
#end
#
#experiment = FilterExps.filter_state
#
#
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
# classic_state parallel run, arguments are
# time_series, method, seed, nanl, lag, shift, obs_un, obs_dim, γ, N_ens, state_infl = args

schemes = ["etks"]
seed = 0
lags = 1:3:52
shifts = [1]
#lags = [1, 2, 4, 8, 16, 32, 64]
#gammas = Array{Float64}(1:11)
gammas = [1.0]
shift = 1
obs_un = 1.0
obs_dim = 40
#N_ens = 15:2:41
N_ens = [21]
#state_infl = [1.0]
state_infl = LinRange(1.0, 1.10, 11)
time_series = [ts1, ts2, ts3, ts4, ts5]
nanl = 2500

# load the experiments
args = Tuple[]
for ts in time_series
    for scheme in schemes
        for γ in gammas
            for l in 1:length(lags)
                # optional definition of shifts in terms of the current lag parameter for a
                # range of shift values
                lag = lags[l]
                #shifts = lags[1:l]
                for shift in shifts
                    for N in N_ens
                        for s_infl in state_infl
                            tmp = (ts, scheme, seed, nanl, lag, shift, obs_un, obs_dim,
                                   γ, N, s_infl)
                            push!(args, tmp)
                        end
                    end
                end
            end
        end
    end
end


## define the robust to failure wrapper
#function wrap_exp(arguments)
#    try
#        classic_state(arguments)
#    catch
#        print("Error on " * string(args) * "\n")
#    end
#end

experiment = SmootherExps.classic_state
#experiment = wrap_exp


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
# Run the experiments in parallel over the parameter values
##############################################################################################
pmap(experiment, args)

##############################################################################################
# end module

end
