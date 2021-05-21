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
#schemes = ["mlef-transform", "mlef-ls-transform"]
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
#    for scheme in schemes
#        for γ in gammas
#            for N in N_ens
#                for α in infl
#                    tmp = (ts, scheme, seed, obs_un, obs_dim, γ, N, α)
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
# time_series, method, seed, lag, shift, obs_un, obs_dim, γ, N_ens, state_infl = args
#
#schemes = ["mles-transform"]
#seed = 0
#lag = 1:3:52
#shift = 1
#obs_un = 1.0
#obs_dim = 40
#gammas = Array{Float64}(9:11)
##N_ens = 15:2:43
#N_ens = [21]
##state_infl = [1.0]
#state_infl = LinRange(1.0, 1.10, 11)
#time_series = [time_series_1]
#
## load the experiments
#args = Tuple[]
#for ts in time_series
#    for γ in gammas
#        for scheme in schemes
#            for l in lag
#                for N in N_ens
#                    for s_infl in state_infl
#                        tmp = (ts, scheme, seed, l, shift, obs_un, obs_dim, γ, N, s_infl)
#                        push!(args, tmp)
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
#    write(f,"#SBATCH -p slow\n")
#    write(f,"#SBATCH -o ensemble_run.out\n")
#    write(f,"#SBATCH -e ensemble_run.err\n")
#    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"classic_smoother_state\"")
#    close(f)
#    my_command = `sbatch  submit_job.sl`
#    run(my_command)
#end
#
########################################################################################################################
# Hybrid smoothers
########################################################################################################################
# submit the experiments given the parameters and write to text files over the initializations
# [time_series, method, seed, lag, shift, mda, obs_un, obs_dim, N_ens, state_infl = args
#
#sys_dim = 40
#obs_dim = 40
#nanl = 25000
#h = 0.01
#schemes = ["mles-transform"]
#seed = 0
#lag = 1:3:52
##lag = [2;lag]
#gammas = Array{Float64}(1:11)
#shift = 1
#obs_un = 1.0
#obs_dim = 40
#N_ens = [21]
##N_ens = 15:2:43
##state_infl = [1.0]
#state_infl = LinRange(1.00, 1.10, 11)
#time_series = [time_series_1]
#mda = true 
#
#
## load the experiments
#args = Tuple[]
#for ts in time_series
#    for scheme in schemes
#        for γ in gammas
#            for l in lag
#                for N in N_ens
#                    for s_infl in state_infl
#                        #tanl = parse(Float64,ts[66:69])
#                        #name = scheme *
#                        #            "_single_iteration_l96_state_benchmark_seed_0000" *
#                        #            "_sys_dim_" * lpad(sys_dim, 2, "0") *
#                        #            "_obs_dim_" * lpad(obs_dim, 2, "0") *
#                        #            "_obs_un_" * rpad(obs_un, 4, "0") *
#                        #            "_gamma_" * rpad(γ, 4, "0") *
#                        #            "_nanl_" * lpad(nanl, 5, "0") *
#                        #            "_tanl_" * rpad(tanl, 4, "0") *
#                        #            "_h_" * rpad(h, 4, "0") *
#                        #            "_lag_" * lpad(l, 3, "0") *
#                        #            "_shift_" * lpad(shift, 3, "0") *
#                        #            "_mda_" * string(mda) *
#                        #            "_N_ens_" * lpad(N, 3,"0") *
#                        #            "_state_inflation_" * rpad(round(s_infl, digits=2), 4, "0") *
#                        #            ".jld"
#
#                        #fpath = "/x/capa/scratch/cgrudzien/final_experiment_data/all_ens/" * scheme * "_single_iteration/"
#                        #try
#                        #    f = load(fpath*name)
#                        #catch
#                        #    tmp = (ts, scheme, seed, l, shift, mda, obs_un, obs_dim, γ, N, s_infl)
#                        #    push!(args, tmp)
#                        #end
#                        tmp = (ts, scheme, seed, l, shift, mda, obs_un, obs_dim, γ, N, s_infl)
#                        push!(args, tmp)
#                    end
#                end
#            end
#        end
#    end
#end
#
#name = "/home/cgrudzien/da_benchmark/data/input_data/single_iteration_state_smoother_input_args.jld"
#save(name, "experiments", args)
#
#for j in 1:length(args) 
#    f = open("./submit_job.sl", "w")
#    write(f,"#!/bin/bash\n")
#    write(f,"#SBATCH -n 1\n")
#    write(f,"#SBATCH -p slow\n")
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
# [time_series, method, seed, lag, shift, adaptive, mda, obs_un, obs_dim, N_ens, infl] = args
# load all standard parameter configurations
sys_dim = 40
obs_dim = 40
nanl = 25000
h = 0.01
schemes = ["lin-ienks-transform"]
seed = 0
lag = 1:3:52
gammas = Array{Float64}(1:11)
shift = 1
obs_un = 1.0
obs_dim = 40
#N_ens = 15:2:43
N_ens = [21]
state_infl = LinRange(1.00, 1.10, 11)
#state_infl = [1.0]
mdas = [true]
time_series = [time_series_1]

# load the experiments
args = Tuple[]
for mda in mdas
    for ts in time_series
        for γ in gammas
            for scheme in schemes
                for l in lag
                    for N in N_ens
                        for s_infl in state_infl
                            #tanl = parse(Float64,ts[66:69])
                            #name = scheme *
                            #            "_l96_state_benchmark_seed_0000" *
                            #            "_sys_dim_" * lpad(sys_dim, 2, "0") *
                            #            "_obs_dim_" * lpad(obs_dim, 2, "0") *
                            #            "_obs_un_" * rpad(obs_un, 4, "0") *
                            #            "_gamma_" * rpad(γ, 4, "0") *
                            #            "_nanl_" * lpad(nanl, 5, "0") *
                            #            "_tanl_" * rpad(tanl, 4, "0") *
                            #            "_h_" * rpad(h, 4, "0") *
                            #            "_lag_" * lpad(l, 3, "0") *
                            #            "_shift_" * lpad(shift, 3, "0") *
                            #            "_mda_" * string(mda) *
                            #            "_N_ens_" * lpad(N, 3,"0") *
                            #            "_state_inflation_" * rpad(round(s_infl, digits=2), 4, "0") *
                            #            ".jld"

                            #fpath = "/x/capa/scratch/cgrudzien/final_experiment_data/all_ens/" * scheme * "/"
                            #try
                            #    f = load(fpath*name)
                            #    
                            #catch
                            #    tmp = (ts, scheme, seed, l, shift, mda, obs_un, obs_dim, γ, N, s_infl)
                            #    push!(args, tmp)
                            #end
                            tmp = (ts, scheme, seed, l, shift, mda, obs_un, obs_dim, γ, N, s_infl)
                            push!(args, tmp)
                        end
                    end
                end
            end
        end
    end
end


name = "/home/cgrudzien/da_benchmark/data/input_data/iterative_state_smoother_input_args.jld"
save(name, "experiments", args)

for j in 1:length(args) 
    f = open("./submit_job.sl", "w")
    write(f,"#!/bin/bash\n")
    write(f,"#SBATCH -n 1\n")
    #write(f,"#SBATCH -p slow\n")
    write(f,"#SBATCH -o ensemble_run.out\n")
    write(f,"#SBATCH -e ensemble_run.err\n")
    write(f,"julia SlurmExperimentDriver.jl " * "\"" *string(j) * "\"" * " \"iterative_smoother_state\"")
    close(f)
    my_command = `sbatch  submit_job.sl`
    run(my_command)
end

########################################################################################################################

end

