########################################################################################################################
module SlurmParallelSubmitIEEE
########################################################################################################################
# imports and exports
using FilterExps, SmootherExps, EnsembleKalmanSchemes, DeSolvers, L96, JLD, Debugger

########################################################################################################################
########################################################################################################################
## Time series data 
########################################################################################################################
# observation time series to load into the experiment as truth twin
# time series are named by the model, seed to initialize, the integration scheme used to produce, number of analyses,
# the spinup length, and the time length between observation points

ts01 = "../data/time_series/IEEE39bus_time_series_seed_0000_diff_0.012_tanl_0.01_nanl_50000_spin_5000_h_0.010.jld"
########################################################################################################################

########################################################################################################################
## Experiment parameter generation 
########################################################################################################################
########################################################################################################################
# Filters
########################################################################################################################
#
#
########################################################################################################################
# Classic smoothers
########################################################################################################################
## arguments are
## time_series, method, seed, lag, shift, obs_un, obs_dim, γ, param_err, param_wlk, N_ens, state_infl, param_infl = args
#
## incomplete will determine if the script checks for exisiting data before submitting experiments
#incomplete = true
#
## note, nanl is hard coded in the experiment, and h is inferred from the time series
## data, this is only used for checking versus existing data with the incomplete parameter as above
#nanl = 25000
#h = 0.01
#tanl = 0.01
#diffusion = 0.012
#
## these values set parameters for the experiment when running from scratch, or can
## be tested versus existing data
#sys_dim = 40
#obs_dim = 20
#methods = ["etks"]
#seed = 0
#
## note MDA is only defined for shifts / lags where the lag is a multiple of shift
## MDA is never defined for the classic smoother, but we will use the same parameter
## discretizations for SDA for reference values
#lags = 1:3:52
#shifts = [1]
#
## observation parameters, gamma controls nonlinearity
##gammas = Array{Float64}(1:11)
#gammas = [1.0]
#obs_un = 0.1
#param_err = 0.03
#param_infl = 1.0
#param_wlk = round.(LinRange(0, 10, 11) * 0.001, digits=3)
#
## if varying nonlinearity in obs, typically take a single ensemble value
##N_ens = [21]
#N_ens = 11:2:41
#
## inflation values, finite size versions should only be 1.0 generally 
##state_infl = [1.0]
#state_infl = LinRange(1.00, 1.10, 11)
#
## set the time series of observations for the truth-twin
#time_series = [ts01]
#
## time_series, method, seed, lag, shift, obs_un, obs_dim, γ, param_err, param_wlk, N_ens, state_infl, param_infl = args
## load the experiments as a tuple
#args = Tuple[]
#for ts in time_series
#    for method in methods
#        for γ in gammas
#            for l in 1:length(lags)
#                lag = lags[l]
#                #shifts = lags[1:l]
#                for shift in shifts
#                    for N in N_ens
#                        for s_infl in state_infl
#                            for wlk in param_wlk
#                                if incomplete == true
#                                    #tanl = parse(Float64,ts[71:74])
#                                    #diffusion = parse(Float64,ts[60:64])
#                                    name = method *
#                                                "-classic_IEEE39bus_param_seed_" * lpad(seed, 4, "0") *
#                                                "_diff_" * rpad(diffusion, 5, "0") *
#                                                "_sysD_" * lpad(sys_dim, 2, "0") *
#                                                "_obsD_" * lpad(obs_dim, 2, "0") *
#                                                "_obsU_" * rpad(obs_un, 4, "0") *
#                                                "_gamma_" * lpad(γ, 5, "0") *
#                                                "_paramE_" * rpad(param_err, 4, "0") *
#                                                "_paramW_" * rpad(wlk, 6, "0") *
#                                                "_nanl_" * lpad(nanl, 5, "0") *
#                                                "_tanl_" * rpad(tanl, 4, "0") *
#                                                "_h_" * rpad(h, 4, "0") *
#                                                "_lag_" * lpad(lag, 3, "0") *
#                                                "_shift_" * lpad(shift, 3, "0") *
#                                                "_mda_false" *
#                                                "_nens_" * lpad(N, 3,"0") *
#                                                "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
#                                                "_paramInfl_" * rpad(round(param_infl, digits=2), 4, "0") *
#                                                ".jld"
#
#                                    fpath = "/x/capa/scratch/cgrudzien/power_grid_data/" * method * "-classic/"
#                                    try
#                                        f = load(fpath*name)
#                                    catch
#                                        tmp = (ts, method, seed, lag, shift, obs_un, obs_dim, γ, param_err, wlk, N, s_infl, param_infl)
#                                        push!(args, tmp)
#                                    end
#                                else
#                                    tmp = (ts, method, seed, lag, shift, obs_un, obs_dim, γ, param_err, wlk, N, s_infl, param_infl)
#                                    push!(args, tmp)
#                                end
#                            end
#                        end
#                    end
#                end
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/DataAssimilationBenchmarks/data/input_data/classic_param_smoother_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    write(f,"#SBATCH --mem-per-cpu=2500M\n")
#    write(f,"#SBATCH --account=cpu-s1-ahn-0\n")
#    write(f,"#SBATCH --partition=cpu-s1-ahn-0\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"classic_smoother_param\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
########################################################################################################################
# Single-iteration smoothers 
########################################################################################################################
## arguments are
## time_series, method, seed, lag, shift, mda, obs_un, obs_dim, γ,
## param_err, param_wlk, N_ens, state_infl, param_infl = args
#
## incomplete will determine if the script checks for exisiting data before submitting experiments
#incomplete = false 
#
## note, nanl is hard coded in the experiment, and h is inferred from the time series
## data, this is only used for checking versus existing data with the incomplete parameter as above
#nanl = 25000
#h = 0.01
#tanl = 0.01
#diffusion = 0.012
#
## these values set parameters for the experiment when running from scratch, or can
## be tested versus existing data
#sys_dim = 40
#obs_dim = 20
#methods = ["etks"]
#seed = 0
#
## note MDA is only defined for shifts / lags where the lag is a multiple of shift
## MDA is never defined for the classic smoother, but we will use the same parameter
## discretizations for SDA for reference values
#lags = 1:3:52
#shifts = [1]
#
## observation parameters, gamma controls nonlinearity
##gammas = Array{Float64}(1:11)
#gammas = [1.0]
#obs_un = 0.1
#param_err = 0.03
#param_infl = 1.0
#param_wlk = round.(LinRange(0, 10, 11) * 0.001, digits=3)
#
## if varying nonlinearity in obs, typically take a single ensemble value
##N_ens = [21]
#N_ens = 11:2:41
#
## inflation values, finite size versions should only be 1.0 generally 
##state_infl = [1.0]
#state_infl = LinRange(1.00, 1.10, 11)
#
## set the time series of observations for the truth-twin
#time_series = [ts01]
#
## set MDA true or false
#mdas = [false, true]
#
## time_series, method, seed, lag, shift, mda, obs_un, obs_dim, γ, param_err, param_wlk, N_ens, state_infl, param_infl = args
## load the experiments as a tuple
#args = Tuple[]
#for ts in time_series
#    for mda in mdas
#        for method in methods
#            for γ in gammas
#                for l in 1:length(lags)
#                    lag = lags[l]
#                    #shifts = lags[1:l]
#                    for shift in shifts
#                        for N in N_ens
#                            for s_infl in state_infl
#                                for wlk in param_wlk
#                                    if incomplete == true
#                                        #tanl = parse(Float64,ts[71:74])
#                                        #diffusion = parse(Float64,ts[60:64])
#                                        name = method *
#                                                    "-single-iteration_IEEE39bus_param_seed_" * lpad(seed, 4, "0") *
#                                                    "_diff_" * rpad(diffusion, 5, "0") *
#                                                    "_sysD_" * lpad(sys_dim, 2, "0") *
#                                                    "_obsD_" * lpad(obs_dim, 2, "0") *
#                                                    "_obsU_" * rpad(obs_un, 4, "0") *
#                                                    "_gamma_" * lpad(γ, 5, "0") *
#                                                    "_paramE_" * rpad(param_err, 4, "0") *
#                                                    "_paramW_" * rpad(wlk, 6, "0") *
#                                                    "_nanl_" * lpad(nanl, 5, "0") *
#                                                    "_tanl_" * rpad(tanl, 4, "0") *
#                                                    "_h_" * rpad(h, 4, "0") *
#                                                    "_lag_" * lpad(lag, 3, "0") *
#                                                    "_shift_" * lpad(shift, 3, "0") *
#                                                    "_mda_" * string(mda) *
#                                                    "_nens_" * lpad(N, 3,"0") *
#                                                    "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
#                                                    "_paramInfl_" * rpad(round(param_infl, digits=2), 4, "0") *
#                                                    ".jld"
#
#                                        fpath = "/x/capa/scratch/cgrudzien/power_grid_data/" * method * "-single-iteration/"
#                                        try
#                                            f = load(fpath*name)
#                                        catch
#                                            tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, param_err, wlk, N, s_infl, param_infl)
#                                            push!(args, tmp)
#                                        end
#                                    else
#                                        tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, param_err, wlk, N, s_infl, param_infl)
#                                        push!(args, tmp)
#                                    end
#                                end
#                            end
#                        end
#                    end
#                end
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/DataAssimilationBenchmarks/data/input_data/single_iteration_param_smoother_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    write(f,"#SBATCH --mem-per-cpu=2500M\n")
#    write(f,"#SBATCH --account=cpu-s1-ahn-0\n")
#    write(f,"#SBATCH --partition=cpu-s1-ahn-0\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"single_iteration_smoother_param\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
########################################################################################################################
# Iterative smoothers
########################################################################################################################
# arguments are
# time_series, method, seed, lag, shift, mda, obs_un, obs_dim, γ,
# param_err, param_wlk, N_ens, state_infl, param_infl = args

# incomplete will determine if the script checks for exisiting data before submitting experiments
incomplete = false 

# note, nanl is hard coded in the experiment, and h is inferred from the time series
# data, this is only used for checking versus existing data with the incomplete parameter as above
nanl = 25000
h = 0.01
tanl = 0.01
diffusion = 0.012

# these values set parameters for the experiment when running from scratch, or can
# be tested versus existing data
sys_dim = 40
obs_dim = 20
methods = ["ienks-transform"]
seed = 0

# note MDA is only defined for shifts / lags where the lag is a multiple of shift
# MDA is never defined for the classic smoother, but we will use the same parameter
# discretizations for SDA for reference values
lags = 1:3:52
shifts = [1]

# observation parameters, gamma controls nonlinearity
#gammas = Array{Float64}(1:11)
gammas = [1.0]
obs_un = 0.1
param_err = 0.03
param_infl = 1.0
#param_wlk = round.(LinRange(0, 10, 11) * 0.001, digits=3)
param_wlk = [0.000]

# if varying nonlinearity in obs, typically take a single ensemble value
#N_ens = [21]
N_ens = 11:2:41

# inflation values, finite size versions should only be 1.0 generally 
#state_infl = [1.0]
state_infl = LinRange(1.00, 1.10, 11)

# set the time series of observations for the truth-twin
time_series = [ts01]

# set MDA true or false
mdas = [false, true]

# time_series, method, seed, lag, shift, mda, obs_un, obs_dim, γ, param_err, param_wlk, N_ens, state_infl, param_infl = args
# load the experiments as a tuple
args = Tuple[]
for ts in time_series
    for mda in mdas
        for method in methods
            for γ in gammas
                for l in 1:length(lags)
                    lag = lags[l]
                    #shifts = lags[1:l]
                    for shift in shifts
                        for N in N_ens
                            for s_infl in state_infl
                                for wlk in param_wlk
                                    if incomplete == true
                                        #tanl = parse(Float64,ts[71:74])
                                        #diffusion = parse(Float64,ts[60:64])
                                        name = method *
                                                    "_IEEE39bus_param_seed_" * lpad(seed, 4, "0") *
                                                    "_diff_" * rpad(diffusion, 5, "0") *
                                                    "_sysD_" * lpad(sys_dim, 2, "0") *
                                                    "_obsD_" * lpad(obs_dim, 2, "0") *
                                                    "_obsU_" * rpad(obs_un, 4, "0") *
                                                    "_gamma_" * lpad(γ, 5, "0") *
                                                    "_paramE_" * rpad(param_err, 4, "0") *
                                                    "_paramW_" * rpad(wlk, 6, "0") *
                                                    "_nanl_" * lpad(nanl, 5, "0") *
                                                    "_tanl_" * rpad(tanl, 4, "0") *
                                                    "_h_" * rpad(h, 4, "0") *
                                                    "_lag_" * lpad(lag, 3, "0") *
                                                    "_shift_" * lpad(shift, 3, "0") *
                                                    "_mda_" * string(mda) *
                                                    "_nens_" * lpad(N, 3,"0") *
                                                    "_stateInfl_" * rpad(round(s_infl, digits=2), 4, "0") *
                                                    "_paramInfl_" * rpad(round(param_infl, digits=2), 4, "0") *
                                                    ".jld"

                                        fpath = "/x/capa/scratch/cgrudzien/power_grid_data/" * method * "/"
                                        try
                                            f = load(fpath*name)
                                        catch
                                            tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, param_err, wlk, N, s_infl, param_infl)
                                            push!(args, tmp)
                                        end
                                    else
                                        tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, param_err, wlk, N, s_infl, param_infl)
                                        push!(args, tmp)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

name = "/home/cgrudzien/DataAssimilationBenchmarks/data/input_data/iterative_param_smoother_input_args.jld"
save(name, "experiments", args)

for j in 1:length(args) 
    f = open("./submit_job.sl", "w")
    write(f,"#!/bin/bash\n")
    write(f,"#SBATCH -n 1\n")
    write(f,"#SBATCH --mem-per-cpu=2500M\n")
    write(f,"#SBATCH --account=cpu-s1-ahn-0\n")
    write(f,"#SBATCH --partition=cpu-s1-ahn-0\n")
    write(f,"#SBATCH -o ensemble_run.out\n")
    write(f,"#SBATCH -e ensemble_run.err\n")
    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"iterative_smoother_param\"")
    close(f)
    my_command = `sbatch  submit_job.sl`
    run(my_command)
end

#########################################################################################################################

end

