########################################################################################################################
module ParallelExperimentDriver 
########################################################################################################################
# imports and exports
using Distributed
using Debugger
@everywhere push!(LOAD_PATH, "/media/Storage/Dropbox/Documents/scripting/julia/data_assimilation/da_benchmark")
@everywhere push!(LOAD_PATH, "/media/Storage/Dropbox/Documents/scripting/julia/data_assimilation/da_benchmark/methods")
@everywhere push!(LOAD_PATH, "/media/Storage/Dropbox/Documents/scripting/julia/data_assimilation/da_benchmark/models")
@everywhere push!(LOAD_PATH, "/media/Storage/Dropbox/Documents/scripting/julia/data_assimilation/da_benchmark/experiments")
@everywhere using FilterExps, EnsembleKalmanSchemes, DeSolvers, L96

########################################################################################################################
########################################################################################################################
## Timeseries data 
########################################################################################################################
# observation timeseries to load into the experiment as truth twin
# timeseries are named by the model, seed to initialize, the integration scheme used to produce, number of analyses,
# the spinup length, and the time length between observation points
#
time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.00_tanl_0.05_nanl_50000_spin_5000_h_0.010.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.00_tanl_0.10_nanl_50000_spin_5000_h_0.010.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.10_tanl_0.05_nanl_50000_spin_5000_h_0.005.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.10_tanl_0.10_nanl_50000_spin_5000_h_0.005.jld"
########################################################################################################################

########################################################################################################################
## Experiment parameter generation 
########################################################################################################################
########################################################################################################################

########################################################################################################################
# Filters
########################################################################################################################
########################################################################################################################
# filter_state 
########################################################################################################################
## [time_series, scheme, seed, obs_un, obs_dim, N_ens, infl] = args
#
#schemes = ["enkf", "etkf"]
#seed = 0
#obs_un = 1.0
#obs_dim = 40
#N_ens = 20:25
#infl = LinRange(1.0, 1.02, 3)
#
## load the experiments
#args = Tuple[]
#for scheme in schemes
#    for N in N_ens
#        for α in infl
#            tmp = (time_series, scheme, seed, obs_un, obs_dim, N, α)
#            push!(args, tmp)
#        end
#    end
#end
#
#experiment = FilterExps.filter_state
#
#
########################################################################################################################
# filter_param 
########################################################################################################################
## [time_series, scheme, seed, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
schemes = ["enkf", "etkf"]
seed = 0
obs_un = 1.0
obs_dim = 40
param_err = 0.03
param_wlk = [0.0000, 0.0001, 0.0010, 0.0100]
N_ens = 20:25
state_infl = LinRange(1.0, 1.02, 3)
param_infl = LinRange(1.0, 1.02, 3)

# load the experiments
args = Tuple[]
for scheme in schemes
    for wlk in param_wlk
        for N in N_ens
            for s_infl in state_infl
                for p_infl in param_infl
                    tmp = (time_series, scheme, seed, obs_un, obs_dim, param_err, wlk, N, s_infl, p_infl)
                    push!(args, tmp)
                end
            end
        end
    end
end

experiment = FilterExps.filter_param


########################################################################################################################
########################################################################################################################
# Classic smoothers
########################################################################################################################
## classic_state single run for degbugging, arguments are
## [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args
#
#args = [time_series, 'etks', 0, 11, 11, 1.0, 40, 25, 1.05]
#print(classic_state(args))
########################################################################################################################
## classic_param single run for debugging, arguments are
## [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
#args = [time_series, 'etks', 0, 1, 1, 1.0, 40, 0.03, 0.01, 24, 1.08, 1.0] 
#print(classic_param(args))
########################################################################################################################

########################################################################################################################
# Hybrid smoothers
########################################################################################################################
# hybrid_state single run for degbugging, arguments are
# [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args
#
#args = [time_series, 'etks', 0, 31, 1, 1.0, 40, 25, 1.01]
#print(hybrid_state(args))
########################################################################################################################
# hybrid_param single run for debugging, arguments are
# [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
#args = [time_series, 'etks', 0, 26, 1, 1.0, 40, 0.03, 0.01, 25, 1.01, 1.0] 
#print(hybrid_param(args))
########################################################################################################################
########################################################################################################################
# Run the experiments in parallel over the parameter values
########################################################################################################################
########################################################################################################################
pmap(experiment, args)

end
