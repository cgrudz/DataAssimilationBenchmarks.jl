##############################################################################################
module SingleExperimentDriver
##############################################################################################
##############################################################################################
# imports and exports
using ..FilterExps, ..SmootherExps, ..GenerateTimeSeries, JLD2
export filter_state_exp, filter_param_exp, classic_smoother_state_exp,
       classic_smoother_param_exp, single_iteration_smoother_state_exp,
       single_iteration_smoother_param_exp, iterative_smoother_state_exp,
       iterative_smoother_param_exp, L96_time_series_exp,
       IEEE39bus_time_series_exp, L96_time_series_test, IEEE39bus_time_series_test

##############################################################################################
# Description
##############################################################################################
# The following are used as wrappers for the experiments to work with standard inputs
# for debugging, benchmarking and profiling.  These will work simply with the standard macros,
# while the parallel submission scripts are used for performance on servers
##############################################################################################
##############################################################################################
##############################################################################################
## Generate time series data
##############################################################################################
##############################################################################################

##############################################################################################
## Lorenz-96 model
##############################################################################################
# Lorenz-96(-s) truth twin data generated as a single function call, arguments are
# seed, state_dim, tanl, nanl, spin, diffusion, F = args

function L96_time_series_exp()
    args = (0, 40, 0.05, 50000, 5000, 0.00, 8.0)
    L96_time_series(args)
end


##############################################################################################
## IEEE 39 bus test case model
##############################################################################################
# IEEE 39 bus test case truth twin data generated as a single function call, arguments are
# seed, tanl, nanl, spin, diffusion = args

function IEEE39bus_time_series_exp()
    args = (0, 0.01, 50000, 5000, 0.0)
    IEEE39bus_time_series(args)
end


##############################################################################################
##############################################################################################
## DA experiments begin here
##############################################################################################
##############################################################################################

##############################################################################################
## Time series data
##############################################################################################
# Observation timeseries to load into DA experiments as a truth twin, without
# needing to generate the timeseries with the model twin.  

# path to time series directory
path = joinpath(@__DIR__, "../data/time_series/") 

# file names
#fname = "IEEE39bus_time_series_seed_0000_diff_0.000_tanl_0.01_nanl_50000_spin_5000_" * 
#        "h_0.010.jld2"
#fname = "IEEE39bus_time_series_seed_0000_diff_0.012_tanl_0.01_nanl_50000_spin_5000_" * 
#        "h_0.010.jld2"
fname = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.05_nanl_50000_" * 
        "spin_5000_h_0.050.jld2"
#fname = "L96_time_series_seed_0000_dim_40_diff_0.000_F_08.0_tanl_0.10_nanl_50000_" *
#        "spin_5000_h_0.050.jld2"
#fname = "L96_time_series_seed_0000_dim_40_diff_0.100_F_08.0_tanl_0.05_nanl_50000_" *
#         "spin_5000_h_0.005.jld2"
#fname = "L96_time_series_seed_0000_dim_40_diff_0.100_F_08.0_tanl_0.10_nanl_50000_" *
#        "spin_5000_h_0.005.jld2"

# load the file name with the path
time_series = path * fname 


##############################################################################################

##############################################################################################
# Filters
##############################################################################################
# filter state estimation, arguments are
# time_series, scheme, seed, nanl, obs_un, obs_dim, γ, N_ens, infl = args

function filter_state_exp()
    args = (time_series, "etkf", 0, 2500, 1.0, 40, 1.00, 21, 1.02)
    filter_state(args)
end


##############################################################################################
# filter joint state-parameter estimation, arguments are
# time_series, scheme, seed, nanl, obs_un, obs_dim, γ, param_err, param_wlk, N_ens,
# state_infl, param_infl = args

function filter_param_exp()
    args = (time_series, "etkf", 0, 2500, 1.0, 40, 1.0, 0.10, 0.0010, 21, 1.02, 1.0)
    filter_param(args)
end


##############################################################################################

##############################################################################################
# Classic smoothers
##############################################################################################
# classic EnKS style smoothing for state estimation, arguments are
# time_series, method, seed, nanl, lag, shift, obs_un, obs_dim, γ, N_ens, infl = args

function classic_smoother_state_exp()
    args = (time_series, "etks", 0, 2500, 10, 1, 1.0, 40, 1.0, 21, 1.02)
    classic_state(args)
end


##############################################################################################
# classic EnKS style smoothing for joint state-parameter estimation, arguments are
# time_series, method, seed, nanl, lag, shift, obs_un, obs_dim, γ,
# param_err, param_wlk, N_ens, state_infl, param_infl = args

function classic_smoother_param_exp()
    args = (time_series, "etks", 0, 2500, 10, 1, 1.0, 40, 1.0, 0.03, 0.001, 21, 1.02, 1.0)
    classic_param(args)
end


##############################################################################################

##############################################################################################
# Single iteration smoothers
##############################################################################################
# Single iteration ensemble Kalman smoothing for state estimation, arguments are
# time_series, method, seed, nanl, lag, shift, mda, obs_un, obs_dim, γ, N_ens, infl = args

function single_iteration_smoother_state_exp()
    args = (time_series, "etks", 0, 2500, 10, 1, false, 0.1, 20, 1.0, 21, 1.02)
    single_iteration_state(args)
end


##############################################################################################
# Single iteration ensemble Kalman smoothing for joint state-parameter estimation,
# arguments are
# time_series, method, seed, nanl, lag, shift, mda, obs_un, obs_dim, γ,
# param_err, param_wlk, N_ens, state_infl, param_infl = args

function single_iteration_smoother_param_exp()
    args = (time_series, 
            "etks", 0, 2500, 10, 1, false, 0.1, 20, 1.0, 0.03, 0.0010, 21, 1.02, 1.00)
    single_iteration_param(args)
end


##############################################################################################
# Iterative smoothers
##############################################################################################
# IEnKS style iterative smoothing for state estmation, arugments are
# time_series, method, seed, nanl, lag, shift, mda, obs_un, obs_dim, γ, N_ens, infl = args

function iterative_smoother_state_exp()
    args = (time_series,
            "ienks-transform", 0, 2500, 10, 1, false, 0.1, 20, 1.0, 21, 1.02)
    iterative_state(args)
end


##############################################################################################
# IEnKS style iterative smoothing for joint state-parameter estmation, arugments are
# time_series, method, seed, nanl, lag, shift, mda, obs_un, obs_dim, γ,
# param_err, param_wlk, N_ens, state_infl, param_infl = args

function iterative_smoother_param_exp()
    args = (time_series, "ienks-transform",
            0, 2500, 10, 1, true, 1.0, 20, 1.0, 0.03, 0.0001,  21, 1.02, 1.0)
    iterative_param(args)
end


##############################################################################################
# end module

end
