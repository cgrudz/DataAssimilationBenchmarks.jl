########################################################################################################################
module SlurmParallelSubmitPronghorn
########################################################################################################################
# imports and exports
push!(LOAD_PATH, "/data/gpfs/home/cgrudzien/da_benchmark/data")
push!(LOAD_PATH, "/data/gpfs/home/cgrudzien/da_benchmark")
push!(LOAD_PATH, "/data/gpfs/home/cgrudzien/da_benchmark/methods")
push!(LOAD_PATH, "/data/gpfs/home/cgrudzien/da_benchmark/models")
push!(LOAD_PATH, "/data/gpfs/home/cgrudzien/da_benchmark/experiments")
using FilterExps, SmootherExps, EnsembleKalmanSchemes, DeSolvers, L96, JLD, Debugger

########################################################################################################################
########################################################################################################################
## Time series data 
########################################################################################################################
# observation time series to load into the experiment as truth twin
# time series are named by the model, seed to initialize, the integration scheme used to produce, number of analyses,
# the spinup length, and the time length between observation points

ts01 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.05_nanl_50000_spin_5000_h_0.010.jld"
ts02 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.10_nanl_50000_spin_5000_h_0.010.jld"
ts03 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.15_nanl_50000_spin_5000_h_0.010.jld"
ts04 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.20_nanl_50000_spin_5000_h_0.010.jld"
ts05 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.25_nanl_50000_spin_5000_h_0.010.jld"
ts06 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.30_nanl_50000_spin_5000_h_0.010.jld"
ts07 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.35_nanl_50000_spin_5000_h_0.010.jld"
ts08 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.40_nanl_50000_spin_5000_h_0.010.jld"
ts09 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.45_nanl_50000_spin_5000_h_0.010.jld"
ts10 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.50_nanl_50000_spin_5000_h_0.010.jld"
ts11 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.55_nanl_50000_spin_5000_h_0.010.jld"
ts12 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_F_08.0_tanl_0.60_nanl_50000_spin_5000_h_0.010.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.10_tanl_0.05_nanl_50000_spin_5000_h_0.005.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.10_tanl_0.10_nanl_50000_spin_5000_h_0.005.jld"
########################################################################################################################

########################################################################################################################
## Experiment parameter generation 
########################################################################################################################
########################################################################################################################
# Filters
########################################################################################################################
# arguments are 
# time_series, method, seed, obs_un, obs_dim, γ, N_ens, infl = args
#
#methods = ["mlef-transform", "mlef-ls-transform"]
#seed = 0
#obs_un = 1.0
#obs_dim = 40
#gammas = Array{Float64}(1:10)
#N_ens = 15:2:43 
#infl = LinRange(1.0, 1.10, 11)
#time_series = [time_series_1]
#
## load the experiments
#args = Tuple[]
#for ts in time_series
#    for method in methods
#        for γ in gammas
#            for N in N_ens
#                for α in infl
#                    tmp = (ts, method, seed, obs_un, obs_dim, γ, N, α)
#                    push!(args, tmp)
#                end
#            end
#        end
#    end
#end
#
#name = "/data/gpfs/home/cgrudzien/da_benchmark/data/input_data/filter_state_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    write(f,"#SBATCH -t 12-00:00:00\n")
#    write(f,"#SBATCH --mem-per-cpu=2500M\n")
#    write(f,"#SBATCH --account=cpu-s1-ahn-0\n")
#    write(f,"#SBATCH --partition=cpu-s1-ahn-0\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriverPronghorn.jl " * "\"" *string(j) * "\"" * " \"filter_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
#
########################################################################################################################
# Classic smoothers
########################################################################################################################
# arguments are
# time_series, method, seed, lag, shift, obs_un, obs_dim, γ, N_ens, state_infl = args

# these values set parameters for the experiment when running from scratch, or can
# be tested versus existing data
sys_dim = 40
obs_dim = 40
methods = ["etks"]
seed = 0

# note MDA is only defined for shifts / lags where the lag is a multiple of shift
# MDA is never defined for the classic smoother, but we will use the same parameter
# discretizations for SDA for reference values
lags = 1:3:52
shifts = [1]
#lags = [1, 2, 4, 8, 16, 32, 64]

# observation parameters, gamma controls nonlinearity
#gammas = Array{Float64}(1:11)
gammas = [1.0]
obs_un = 1.0
obs_dim = 40

# if varying nonlinearity in obs, typically take a single ensemble value
N_ens = [21]
#N_ens = 15:2:43

# inflation values, finite size versions should only be 1.0 generally 
#state_infl = [1.0]
state_infl = LinRange(1.00, 1.10, 11)

# set the time series of observations for the truth-twin
#time_series = [ts01, ts02]
#time_series = [ts01, ts02, ts03, ts04, ts05, ts06, ts07, ts08, ts09, ts10]
time_series = [ts06, ts07, ts08, ts09, ts10]

# load the experiments as a tuple
args = Tuple[]
for ts in time_series
    for method in methods
        for γ in gammas
            for l in 1:length(lags)
                lag = lags[l]
                #shifts = lags[1:l]
                for shift in shifts
                    for N in N_ens
                        for s_infl in state_infl
                            tmp = (ts, method, seed, lag, shift, obs_un, obs_dim, γ, N, s_infl)
                            push!(args, tmp)
                        end
                    end
                end
            end
        end
    end
end

name = "/data/gpfs/home/cgrudzien/da_benchmark/data/input_data/classic_state_smoother_input_args.jld"
save(name, "experiments", args)

for j in 1:length(args) 
    f = open("./submit_job.sl", "w")
    write(f,"#!/bin/bash\n")
    write(f,"#SBATCH -n 1\n")
    write(f,"#SBATCH -t 12-00:00:00\n")
    write(f,"#SBATCH --mem-per-cpu=2500M\n")
    write(f,"#SBATCH --account=cpu-s1-ahn-0\n")
    write(f,"#SBATCH --partition=cpu-s1-ahn-0\n")
    write(f,"#SBATCH -o ensemble_run.out\n")
    write(f,"#SBATCH -e ensemble_run.err\n")
    write(f,"julia SlurmExperimentDriverPronghorn.jl " * "\"" *string(j) * "\"" * " \"classic_smoother_state\"")
    close(f)
    my_command = `sbatch  submit_job.sl`
    run(my_command)
end

########################################################################################################################
# Single-iteration smoothers 
########################################################################################################################
## arguments are
## time_series, method, seed, lag, shift, mda, obs_un, obs_dim, γ, N_ens, state_infl = args
#
## these values set parameters for the experiment when running from scratch, or can
## be tested versus existing data
#sys_dim = 40
#obs_dim = 40
#methods = ["etks"]
#seed = 0
#mdas = [false, true]
#
## note MDA is only defined for shifts / lags where the lag is a multiple of shift
#lags = 1:3:52
#shifts = [1]
##lags = [1, 2, 4, 8, 16, 32, 64]
#
## observation parameters, gamma controls nonlinearity
##gammas = Array{Float64}(1:11)
#gammas = [1.0]
#obs_un = 1.0
#obs_dim = 40
#
## if varying nonlinearity in obs, typically take a single ensemble value
#N_ens = [21]
##N_ens = 15:2:43
#
## inflation values, finite size versions should only be 1.0 generally 
##state_infl = [1.0]
#state_infl = LinRange(1.00, 1.10, 11)
#
## set the time series of observations for the truth-twin
##time_series = [ts01, ts02]
#time_series = [ts01, ts02, ts03, ts04, ts05, ts06, ts07, ts08, ts09, ts10]
#
## load the experiments as a tuple
#args = Tuple[]
#for mda in mdas
#    for ts in time_series
#        for γ in gammas
#            for method in methods
#                for l in 1:length(lags)
#                    lag = lags[l]
#                    #shifts = lags[1:l]
#                    for shift in shifts
#                        for N in N_ens
#                            for s_infl in state_infl
#                                tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, N, s_infl)
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
## save the input data to be looped over in the next stage
#name = "/data/gpfs/home/cgrudzien/da_benchmark/data/input_data/single_iteration_state_smoother_input_args.jld"
#save(name, "experiments", args)
#
## the loop will sequentially write and submit different experiments based on the parameter combinations
## in the input data
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    write(f,"#SBATCH --mem-per-cpu=2500M\n")
#    write(f,"#SBATCH --account=cpu-s1-ahn-0\n")
#    write(f,"#SBATCH --partition=cpu-s1-ahn-0\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriverPronghorn.jl " * "\"" *string(j) * "\"" * " \"single_iteration_smoother_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
#
########################################################################################################################
# Iterative smoothers
########################################################################################################################
## arguments are
## time_series, method, seed, lag, shift, adaptive, mda, obs_un, obs_dim, γ, N_ens, infl = args
##
## these values set parameters for the experiment when running from scratch, or can
## be tested versus existing data
#sys_dim = 40
#obs_dim = 40
#methods = ["lin-ienks-transform", "ienks-transform"]
#seed = 0
#mdas = [false, true]
#
## note MDA is only defined for shifts / lags where the lag is a multiple of shift
## this defines the ranged lag and shift parameters
##lags = [1, 2, 4, 8, 16, 32, 64] 
#
## this defines static, standard lag and shift parameters
#lags = 1:3:52
#shifts = [1]
#
## observation parameters, gamma controls nonlinearity
##gammas = Array{Float64}(1:11)
#gammas = [1.0]
#obs_un = 1.0
#obs_dim = 40
#
## if varying nonlinearity in obs, typically take a single ensemble value
#N_ens = [21]
##N_ens = 15:2:43
#
## inflation values, finite size versions should only be 1.0 generally 
##state_infl = [1.0]
#state_infl = LinRange(1.00, 1.10, 11)
#
## set the time series of observations for the truth-twin
##time_series = [ts01, ts02]
##time_series = [ts01, ts02, ts03, ts04, ts05, ts06, ts07, ts08, ts09, ts10]
#time_series = [ts06, ts07, ts08, ts09, ts10]
#
## load the experiments
#args = Tuple[]
#for mda in mdas
#    for ts in time_series
#        for γ in gammas
#            for method in methods
#                for l in 1:length(lags)
#                    lag = lags[l]
#                    # optional definition of shifts in terms of the current lag parameter for a
#                    # range of shift values
#                    #shifts = lags[1:l]
#                    for shift in shifts
#                        for N in N_ens
#                            for s_infl in state_infl
#                                tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, N, s_infl)
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
#
#name = "/data/gpfs/home/cgrudzien/da_benchmark/data/input_data/iterative_state_smoother_input_args.jld"
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
#    write(f,"julia SlurmExperimentDriverPronghorn.jl " * "\"" *string(j) * "\"" * " \"iterative_smoother_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
#########################################################################################################################

end

