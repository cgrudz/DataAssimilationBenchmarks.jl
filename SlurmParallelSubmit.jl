########################################################################################################################
module SlurmParallelSubmit 
########################################################################################################################
# imports and exports
push!(LOAD_PATH, "/home/cgrudzien/da_benchmark/data")
push!(LOAD_PATH, "/home/cgrudzien/da_benchmark")
push!(LOAD_PATH, "/home/cgrudzien/da_benchmark/methods")
push!(LOAD_PATH, "/home/cgrudzien/da_benchmark/models")
push!(LOAD_PATH, "/home/cgrudzien/da_benchmark/experiments")
using FilterExps, SmootherExps, EnsembleKalmanSchemes, DeSolvers, L96, JLD, Debugger

########################################################################################################################
########################################################################################################################
## Timeseries data 
########################################################################################################################
# observation timeseries to load into the experiment as truth twin
# timeseries are named by the model, seed to initialize, the integration scheme used to produce, number of analyses,
# the spinup length, and the time length between observation points
#
time_series_1 = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.00_tanl_0.05_nanl_50000_spin_5000_h_0.010.jld"
time_series_2 = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.00_tanl_0.10_nanl_50000_spin_5000_h_0.010.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.10_tanl_0.05_nanl_50000_spin_5000_h_0.005.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.10_tanl_0.10_nanl_50000_spin_5000_h_0.005.jld"
########################################################################################################################

########################################################################################################################
## Experiment parameter generation 
########################################################################################################################
########################################################################################################################
# Filters
########################################################################################################################
# submit the experiments given the parameters and write to text files over the initializations
# [time_series, scheme, seed, obs_un, obs_dim, N_ens, infl] = args
#
#schemes = ["etkf", "enkf"]
#seed = 0
#obs_un = 1.0
#obs_dim = 40
#N_ens = 14:41
#infl = LinRange(1.0, 1.20, 21)
#time_series = [time_series_1, time_series_2]
#
## load the experiments
#args = Tuple[]
#for ts in time_series
#    for scheme in schemes
#        for N in N_ens
#            for α in infl
#                tmp = (ts, scheme, seed, obs_un, obs_dim, N, α)
#                push!(args, tmp)
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/da_benchmark/data/input_data/filter_state_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"filter_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
#
########################################################################################################################
# Classic smoothers
########################################################################################################################
# submit the experiments given the parameters and write to text files over the initializations
# time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, state_infl = args
#
schemes = ["enks-n"]
seed = 0
lag = 1:3:52
shift = 1
obs_un = 1.0
obs_dim = 40
N_ens = 15:2:43
state_infl = [1.0]#LinRange(1.0, 1.10, 11)
time_series = [time_series_1, time_series_2]

# load the experiments
args = Tuple[]
for ts in time_series
    for scheme in schemes
        for l in lag
            for N in N_ens
                for s_infl in state_infl
                    tmp = (ts, scheme, seed, l, shift, obs_un, obs_dim, N, s_infl)
                    push!(args, tmp)
                end
            end
        end
    end
end

name = "/home/cgrudzien/da_benchmark/data/input_data/classic_state_smoother_input_args.jld"
save(name, "experiments", args)

for j in 1:length(args) 
    f = open("./submit_job.sl", "w")
    write(f,"#!/bin/bash\n")
    write(f,"#SBATCH -n 1\n")
    write(f,"#SBATCH -o ensemble_run.out\n")
    write(f,"#SBATCH -e ensemble_run.err\n")
    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"classic_smoother_state\"")
    close(f)
    my_command = `sbatch  submit_job.sl`
    run(my_command)
end

########################################################################################################################
# Hybrid smoothers
########################################################################################################################
# submit the experiments given the parameters and write to text files over the initializations
# [time_series, method, seed, lag, shift, mda, obs_un, obs_dim, N_ens, state_infl = args
#
#schemes = ["etks_adaptive"]
#seed = 0
#lag = 1:3:52
#shift = 1
#obs_un = 1.0
#obs_dim = 40
#N_ens = 15:2:43
#state_infl = [1.0] #LinRange(1.00, 1.10, 11)
#time_series = [time_series_1, time_series_2]
#mda = false
#
#
## load the experiments
#args = Tuple[]
#for ts in time_series
#    for scheme in schemes
#        for l in lag
#            for N in N_ens
#                for s_infl in state_infl
#                    tmp = (ts, scheme, seed, l, shift, mda, obs_un, obs_dim, N, s_infl)
#                    push!(args, tmp)
#                end
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/da_benchmark/data/input_data/hybrid_state_smoother_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"hybrid_smoother_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
#
########################################################################################################################
# Iterative smoothers
########################################################################################################################
# [time_series, method, seed, lag, shift, adaptive, mda, obs_un, obs_dim, N_ens, infl] = args
#
#schemes = ["ienks-bundle", "ienks-transform"]
#seed = 0
#lag = 1:3:52
#shift = 1
#obs_un = 1.0
#obs_dim = 40
#N_ens = 15:2:43
#state_infl = [1.0]#LinRange(1.00, 1.10, 11)
#adaptive = true 
#mda = false
#time_series = [time_series_1, time_series_2]
#
## load the experiments
#args = Tuple[]
#for ts in time_series
#    for scheme in schemes
#        for l in lag
#            for N in N_ens
#                for s_infl in state_infl
#                    tmp = (ts, scheme, seed, l, shift, adaptive, mda, obs_un, obs_dim, N, s_infl)
#                    push!(args, tmp)
#                end
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/da_benchmark/data/input_data/iterative_state_smoother_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"iterative_smoother_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
########################################################################################################################

end
