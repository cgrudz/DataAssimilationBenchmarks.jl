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
time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.00_tanl_0.05_nanl_50000_spin_5000_h_0.010.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.00_tanl_0.10_nanl_50000_spin_5000_h_0.010.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.10_tanl_0.05_nanl_50000_spin_5000_h_0.005.jld"
#time_series = "./data/timeseries/l96_timeseries_seed_0000_dim_40_diff_0.10_tanl_0.10_nanl_50000_spin_5000_h_0.005.jld"
########################################################################################################################

########################################################################################################################
## Experiment parameter generation 
########################################################################################################################
########################################################################################################################
# Classic smoothers
########################################################################################################################
# hybrid_state single run for degbugging, arguments are
# time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, state_infl = args
#
#schemes = ["etks"]
#seed = 0
#lag = 1:3:52
#shift = 1
#obs_un = 1.0
#obs_dim = 40
#N_ens = 15:2:43
#state_infl = LinRange(1.0, 1.10, 11)
#
## load the experiments
#args = Tuple[]
#for scheme in schemes
#    for l in lag
#        for N in N_ens
#            for s_infl in state_infl
#                tmp = (time_series, scheme, seed, l, shift, obs_un, obs_dim, N, s_infl)
#                push!(args, tmp)
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/da_benchmark/data/input_data/classic_state_smoother_input_args.jld"
#save(name, "experiments", args)
#
#
########################################################################################################################
# Hybrid smoothers
########################################################################################################################
# hybrid_state single run for degbugging, arguments are
# [time_series, method, seed, lag, shift, mda, obs_un, obs_dim, N_ens, state_infl = args
#
schemes = ["etks"]
seed = 0
lag = 1:3:52
shift = 1
obs_un = 1.0
obs_dim = 40
N_ens = 15:2:43
state_infl = LinRange(1.00, 1.10, 11)
mda = true

# load the experiments
args = Tuple[]
for scheme in schemes
    for l in lag
        for N in N_ens
            for s_infl in state_infl
                tmp = (time_series, scheme, seed, l, shift, mda, obs_un, obs_dim, N, s_infl)
                push!(args, tmp)
            end
        end
    end
end

name = "/home/cgrudzien/da_benchmark/data/input_data/hybrid_state_smoother_input_args.jld"
save(name, "experiments", args)

########################################################################################################################
########################################################################################################################
# Iterative smoothers
########################################################################################################################
# submit the experiments given the parameters and write to text files over the initializations
# [time_series, method, seed, lag, shift, mda, obs_un, obs_dim, N_ens, infl] = args
#
#schemes = ["ienks-bundle"]
#seed = 0
#lag = 1:3:52
#shift = 1
#obs_un = 1.0
#obs_dim = 40
#N_ens = 15:2:43
#state_infl = LinRange(1.00, 1.10, 11)
#mda = false
#
## load the experiments
#args = Tuple[]
#for scheme in schemes
#    for l in lag
#        for N in N_ens
#            for s_infl in state_infl
#                tmp = (time_series, scheme, seed, l, shift, mda, obs_un, obs_dim, N, s_infl)
#                push!(args, tmp)
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/da_benchmark/data/input_data/iterative_state_smoother_input_args.jld"
#save(name, "experiments", args)
#
########################################################################################################################
# Loop submit the experiments in the above to the queue
########################################################################################################################
for j in 1:length(args) 
    f = open("./submit_job.sl", "w")
    write(f,"#!/bin/bash\n")
    write(f,"#SBATCH -n 1\n")
    write(f,"#SBATCH -o experiment.out\n")
    write(f,"#SBATCH -e experiment.err\n")
    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"")
    close(f)
    my_command = `sbatch  submit_job.sl`
    run(my_command)
end

end
