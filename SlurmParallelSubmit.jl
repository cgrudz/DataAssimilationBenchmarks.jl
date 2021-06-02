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
## Time series data 
########################################################################################################################
# observation time series to load into the experiment as truth twin
# time series are named by the model, seed to initialize, the integration scheme used to produce, number of analyses,
# the spinup length, and the time length between observation points
#
time_series_1 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_tanl_0.05_nanl_50000_spin_5000_h_0.010.jld"
time_series_2 = "./data/time_series/l96_time_series_seed_0000_dim_40_diff_0.00_tanl_0.10_nanl_50000_spin_5000_h_0.010.jld"
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
# time_series, method, seed, obs_un, obs_dim, N_ens, infl = args
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
#name = "/home/cgrudzien/da_benchmark/data/input_data/filter_state_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    # slow partition is for Okapi, uncomment when necessary
#    write(f,"#SBATCH -p slow\n")
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
## arguments are
## time_series, method, seed, lag, shift, obs_un, obs_dim, γ, N_ens, state_infl = args
#
## incomplete will determine if the script checks for exisiting data before submitting experiments
#incomplete = true
#
## note, nanl is hard coded in the experiment, and h is inferred from the time series
## data, this is only used for checking versus existing data with the incomplete parameter as above
#nanl = 25000
#h = 0.01
#
## these values set parameters for the experiment when running from scratch, or can
## be tested versus existing data
#sys_dim = 40
#obs_dim = 40
#methods = ["mles-n-transform"]
#seed = 0
#
#shift = 1
## if shift is 2, we must reset the range of lags
#if shift == 1
#    lags = 1:3:52
#elseif shift == 4
#    lags = 4:4:52
#end
#
## observation parameters, gamma controls nonlinearity
#gammas = Array{Float64}(1:11)
##gammas = [1.0]
#obs_un = 1.0
#obs_dim = 40
#
## if varying nonlinearity in obs, typically take a single ensemble value
#N_ens = [21]
##N_ens = 15:2:43
#
## inflation values, finite size versions should only be 1.0 generally 
#state_infl = [1.0]
##state_infl = LinRange(1.00, 1.10, 11)
#
## set the time series of observations for the truth-twin
#time_series = [time_series_1, time_series_2]
#
## load the experiments as a tuple
#args = Tuple[]
#for ts in time_series
#    for method in methods
#        for γ in gammas
#            for l in lags
#                for N in N_ens
#                    for s_infl in state_infl
#                        if incomplete == true
#                            tanl = parse(Float64,ts[68:71])
#                            diffusion = parse(Float64,ts[58:61])
#                            name = method *
#                                        "_classic_l96_state_benchmark_seed_" * lpad(seed, 4, "0") *
#                                        "_diffusion_" * rpad(diffusion, 4, "0") *
#                                        "_sys_dim_" * lpad(sys_dim, 2, "0") *
#                                        "_obs_dim_" * lpad(obs_dim, 2, "0") *
#                                        "_obs_un_" * rpad(obs_un, 4, "0") *
#                                        "_gamma_" * lpad(γ, 5, "0") *
#                                        "_nanl_" * lpad(nanl, 5, "0") *
#                                        "_tanl_" * rpad(tanl, 4, "0") *
#                                        "_h_" * rpad(h, 4, "0") *
#                                        "_lag_" * lpad(l, 3, "0") *
#                                        "_shift_" * lpad(shift, 3, "0") *
#                                        "_mda_false" *
#                                        "_N_ens_" * lpad(N, 3,"0") *
#                                        "_state_inflation_" * rpad(round(s_infl, digits=2), 4, "0") *
#                                        ".jld"
#
#                            fpath = "/x/capa/scratch/cgrudzien/final_experiment_data/versus_operator/" * method * "_classic/"
#                            try
#                                f = load(fpath*name)
#                            catch
#                                tmp = (ts, method, seed, l, shift, obs_un, obs_dim, γ, N, s_infl)
#                                push!(args, tmp)
#                            end
#                        else
#                            tmp = (ts, method, seed, l, shift, obs_un, obs_dim, γ, N, s_infl)
#                            push!(args, tmp)
#                        end
#                    end
#                end
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/da_benchmark/data/input_data/classic_state_smoother_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    # slow partition is for Okapi, uncomment when necessary
#    #write(f,"#SBATCH -p slow\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"classic_smoother_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
########################################################################################################################
# Single-iteration smoothers 
########################################################################################################################
## arguments are
## time_series, method, seed, lag, shift, mda, obs_un, obs_dim, N_ens, state_infl = args
#
## incomplete will determine if the script checks for exisiting data before submitting experiments
#incomplete = true 
#
## note, nanl is hard coded in the experiment, and h is inferred from the time series
## data, this is only used for checking versus existing data with the incomplete parameter as above
#nanl = 25000
#h = 0.01
#
## these values set parameters for the experiment when running from scratch, or can
## be tested versus existing data
#sys_dim = 40
#obs_dim = 40
#methods = ["mles-n-transform"]
#seed = 0
#mdas = [false]
#
## note MDA is only defined for shifts / lags where the lag is a multiple of shift
##lags = [1, 2, 4, 8, 16, 32, 64]
#
## this defines static, standard lag and shift parameters
#lags = 1:3:52
#shifts = [1]
## observation parameters, gamma controls nonlinearity
#gammas = Array{Float64}(1:11)
##gammas = [1.0]
#obs_un = 1.0
#obs_dim = 40
#
## if varying nonlinearity in obs, typically take a single ensemble value
#N_ens = [21]
##N_ens = 15:2:43
#
## inflation values, finite size versions should only be 1.0 generally 
#state_infl = [1.0]
##state_infl = LinRange(1.00, 1.10, 11)
#
## set the time series of observations for the truth-twin
#time_series = [time_series_1, time_series_2]
#
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
#                                if incomplete == true
#                                    tanl = parse(Float64,ts[68:71])
#                                    diffusion = parse(Float64,ts[58:61])
#                                    name = method *
#                                                "_single_iteration_l96_state_benchmark_seed_" * lpad(seed, 4, "0") *
#                                                "_diffusion_" * rpad(diffusion, 4, "0") *
#                                                "_sys_dim_" * lpad(sys_dim, 2, "0") *
#                                                "_obs_dim_" * lpad(obs_dim, 2, "0") *
#                                                "_obs_un_" * rpad(obs_un, 4, "0") *
#                                                "_gamma_" * lpad(γ, 5, "0") *
#                                                "_nanl_" * lpad(nanl, 5, "0") *
#                                                "_tanl_" * rpad(tanl, 4, "0") *
#                                                "_h_" * rpad(h, 4, "0") *
#                                                "_lag_" * lpad(lag, 3, "0") *
#                                                "_shift_" * lpad(shift, 3, "0") *
#                                                "_mda_" * string(mda) *
#                                                "_N_ens_" * lpad(N, 3,"0") *
#                                                "_state_inflation_" * rpad(round(s_infl, digits=2), 4, "0") *
#                                                ".jld"
#
#                                    fpath = "/x/capa/scratch/cgrudzien/final_experiment_data/versus_operator/" * method * "_single_iteration/"
#                                    try
#                                        f = load(fpath*name)
#                                    catch
#                                        tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, N, s_infl)
#                                        push!(args, tmp)
#                                    end
#                                else
#                                    tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, N, s_infl)
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
## save the input data to be looped over in the next stage
#name = "/home/cgrudzien/da_benchmark/data/input_data/single_iteration_state_smoother_input_args.jld"
#save(name, "experiments", args)
#
## the loop will sequentially write and submit different experiments based on the parameter combinations
## in the input data
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    # slow partition is for Okapi, uncomment when necessary
#    #write(f,"#SBATCH -p slow\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"single_iteration_smoother_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
#
########################################################################################################################
# Iterative smoothers
########################################################################################################################
# arguments are
# [time_series, method, seed, lag, shift, adaptive, mda, obs_un, obs_dim, N_ens, infl] = args
#
# incomplete will determine if the script checks for exisiting data before submitting experiments
incomplete = true 

# note, nanl is hard coded in the experiment, and h is inferred from the time series
# data, this is only used for checking versus existing data with the incomplete parameter as above
nanl = 25000
h = 0.01

# these values set parameters for the experiment when running from scratch, or can
# be tested versus existing data
sys_dim = 40
obs_dim = 40
methods = ["ienks-n-transform"]
seed = 0
mdas = [false]

# note MDA is only defined for shifts / lags where the lag is a multiple of shift
# this defines the ranged lag and shift parameters
#lags = [1, 2, 4, 8, 16, 32, 64] 

# this defines static, standard lag and shift parameters
lags = 1:3:52
shifts = [1]

# observation parameters, gamma controls nonlinearity
gammas = Array{Float64}(1:11)
#gammas = [1.0]
obs_un = 1.0
obs_dim = 40

# if varying nonlinearity in obs, typically take a single ensemble value
N_ens = [21]
#N_ens = 15:2:43

# inflation values, finite size versions should only be 1.0 generally 
state_infl = [1.0]
#state_infl = LinRange(1.00, 1.10, 11)

# set the time series of observations for the truth-twin
time_series = [time_series_1]

# load the experiments
args = Tuple[]
for mda in mdas
    for ts in time_series
        for γ in gammas
            for method in methods
                for l in 1:length(lags)
                    # optional definition of shifts in terms of the current lag parameter for a
                    # range of shift values
                    lag = lags[l]
                    #shifts = lags[1:l]
                    for shift in shifts
                        for N in N_ens
                            for s_infl in state_infl
                                if incomplete == true
                                    tanl = parse(Float64,ts[68:71])
                                    diffusion = parse(Float64,ts[58:61])
                                    name = method *
                                                "_l96_state_benchmark_seed_" * lpad(seed, 4, "0") * 
                                                "_diffusion_" * rpad(diffusion, 4, "0") *
                                                "_sys_dim_" * lpad(sys_dim, 2, "0") *
                                                "_obs_dim_" * lpad(obs_dim, 2, "0") *
                                                "_obs_un_" * rpad(obs_un, 4, "0") *
                                                "_gamma_" * lpad(γ, 5, "0") *
                                                "_nanl_" * lpad(nanl, 5, "0") *
                                                "_tanl_" * rpad(tanl, 4, "0") *
                                                "_h_" * rpad(h, 4, "0") *
                                                "_lag_" * lpad(lag, 3, "0") *
                                                "_shift_" * lpad(shift, 3, "0") *
                                                "_mda_" * string(mda) *
                                                "_N_ens_" * lpad(N, 3,"0") *
                                                "_state_inflation_" * rpad(round(s_infl, digits=2), 4, "0") *
                                                ".jld"

                                    fpath = "/x/capa/scratch/cgrudzien/final_experiment_data/versus_operator/" * method * "/"
                                    try
                                        f = load(fpath*name)
                                        
                                    catch
                                        tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, N, s_infl)
                                        push!(args, tmp)
                                    end
                                else
                                    tmp = (ts, method, seed, lag, shift, mda, obs_un, obs_dim, γ, N, s_infl)
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


name = "/home/cgrudzien/da_benchmark/data/input_data/iterative_state_smoother_input_args.jld"
save(name, "experiments", args)

#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    # slow partition is for Okapi, uncomment when necessary
#    #write(f,"#SBATCH -p slow\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"iterative_smoother_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
#########################################################################################################################

end

