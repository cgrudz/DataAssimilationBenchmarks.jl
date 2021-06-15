########################################################################################################################
module SingleExperimentDriver
########################################################################################################################
########################################################################################################################
# imports and exports
using FilterExps, SmootherExps, GenerateTimeSeries
export filter_state_exp, filter_param_exp, classic_smoother_state_exp, classic_smoother_param_exp,
        single_iteration_smoother_state_exp, single_iteration_smoother_param_exp, iterative_smoother_state_exp,
        l96_time_series_exp

########################################################################################################################
# Description
########################################################################################################################
# The following functions are used as wrappers for the experiments to work with standard inputs
# for debugging, benchmarking and profiling.  These will work simply with the standard macros,
# while the parallel submission scripts are used for performance on servers
########################################################################################################################

########################################################################################################################
########################################################################################################################
## Generate time series data
########################################################################################################################
########################################################################################################################

########################################################################################################################
## Lorenz-96 model
########################################################################################################################
# Lorenz-96(-s) truth twin data generated as a single function call, arguments are
# seed, state_dim, tanl, nanl, spin, diffusion, F = args

function l96_time_series_exp()
    args = (0, 40, 0.60, 50000, 5000, 0.00, 8.0)
    l96_time_series(args)
end

########################################################################################################################
########################################################################################################################
## DA experiments begin here
########################################################################################################################
########################################################################################################################

########################################################################################################################
## Time series data
########################################################################################################################
# Observation timeseries to load into DA experiments as a truth twin, without needing to generate the timeseries with
# the model twin.  

# path to time series directory
path = "./data/time_series/"

# file names
fname = "l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.05_nanl_50000_spin_5000_h_0.010.jld"
#fname = "/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.10_nanl_50000_spin_5000_h_0.010.jld"
#fname = "/l96_time_series_seed_0000_dim_40_diff_0.10_F_08.0_tanl_0.05_nanl_50000_spin_5000_h_0.005.jld"
#fname = "/l96_time_series_seed_0000_dim_40_diff_0.10_F_08.0_tanl_0.10_nanl_50000_spin_5000_h_0.005.jld"

# load the file name with the path
time_series = path * fname 

########################################################################################################################

########################################################################################################################
# Filters
########################################################################################################################
# filter state estimation, arguments are
# time_series, scheme, seed, obs_un, obs_dim, γ, N_ens, infl = args

function filter_state_exp()
    args = (time_series, "etkf", 0, 1.0, 40, 1.00, 21, 1.03)
    filter_state(args)
end


########################################################################################################################
# filter joint state-parameter estimation, arguments are
# time_series, scheme, seed, obs_un, obs_dim, γ, param_err, param_wlk, N_ens, state_infl, param_infl = args

function filter_param_exp()
    args = (time_series, "etkf", 0, 1.0, 40, 1.0, 0.03, 0.0000, 21, 1.02, 1.0)
    filter_param(args)
end


########################################################################################################################

########################################################################################################################
# Classic smoothers
########################################################################################################################
# classic EnKS style smoothing for state estimation, arguments are
# time_series, method, seed, lag, shift, obs_un, obs_dim, γ, N_ens, infl = args

function classic_smoother_state_exp()
    args = (time_series, "etks", 0, 4, 4, 1.0, 40, 1.0, 21, 1.03)
    classic_state(args)
end


########################################################################################################################
# classic EnKS style smoothing for joint state-parameter estimation, arguments are
# time_series, method, seed, lag, shift, obs_un, obs_dim, γ,
# param_err, param_wlk, N_ens, state_infl, param_infl = args

function classic_smoother_param_exp()
    args = (time_series, "etks", 0, 10, 1, 1.0, 40, 1.0, 0.03, 0.001, 21, 1.03, 1.0)
    classic_param(args)
end


########################################################################################################################

########################################################################################################################
# Single iteration smoothers
########################################################################################################################
# Single iteration ensemble Kalman smoothing for state estimation, arguments are
# time_series, method, seed, lag, shift, adaptive, mda, obs_un, obs_dim, γ, N_ens, infl = args

function single_iteration_smoother_state_exp()
    args = (time_series, "etks", 0, 32, 16, true, 1.0, 40, 1.0, 21, 1.03)
    single_iteration_state(args)
end


########################################################################################################################
# Single iteration ensemble Kalman smoothing for joint state-parameter estimation, arguments are
# time_series, method, seed, lag, shift, mda, obs_un, obs_dim, γ,
# param_err, param_wlk, N_ens, state_infl, param_infl = args

function single_iteration_smoother_param_exp()
    args = (time_series, "etks", 0, 10, 1, false, 1.0, 40, 1.0, 0.03, 0.0010, 21, 1.01, 1.00)
    single_iteration_param(args)
end


########################################################################################################################
# Iterative smoothers
########################################################################################################################
# IEnKS style iterative smoothing for state estmation, arugments are
# time_series, method, seed, lag, shift, adaptive, mda, obs_un, obs_dim, γ, N_ens, infl = args

function iterative_smoother_state_exp()
    args = (time_series, "ienks-transform", 0, 32, 16, true, 1.0, 40, 1.0, 21, 1.03)
    iterative_state(args)
end

########################################################################################################################

end
