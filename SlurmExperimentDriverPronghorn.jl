########################################################################################################################
module SlurmExperimentDriver 
########################################################################################################################
# imports and exports
using Random, Distributions
using Debugger
using Distributed
using LinearAlgebra
using JLD
using DeSolvers
using L96 
using EnsembleKalmanSchemes
using FilterExps
using SmootherExps

########################################################################################################################
########################################################################################################################
## Experiments to run as a single function call
########################################################################################################################
# Filters 
########################################################################################################################
function filter_state_exp(j)
    f = load("/data/gpfs/home/cgrudzien/DataAssimilationBenchmarks/data/input_data/filter_state_input_args.jld")
    args = f["experiments"][j]
    filter_state(args)
end


########################################################################################################################
# Classic smoothers
########################################################################################################################
function classic_smoother_state_exp(j)
    f = load("/data/gpfs/home/cgrudzien/DataAssimilationBenchmarks/data/input_data/classic_state_smoother_input_args.jld")
    args = f["experiments"][j]
    classic_state(args)
end


########################################################################################################################
# Single-iteration smoothers
########################################################################################################################
function single_iteration_smoother_state_exp(j)
    f = load("/data/gpfs/home/cgrudzien/DataAssimilationBenchmarks/data/input_data/single_iteration_state_smoother_input_args.jld")
    args = f["experiments"][j]
    single_iteration_state(args)
end


########################################################################################################################
########################################################################################################################
# Iterative smoothers
########################################################################################################################
function iterative_smoother_state_exp(j)
    f = load("/data/gpfs/home/cgrudzien/DataAssimilationBenchmarks/data/input_data/iterative_state_smoother_input_args.jld")
    args = f["experiments"][j]
    iterative_state(args)
end


########################################################################################################################
########################################################################################################################
# Run experiments
########################################################################################################################
t = parse(Int64, ARGS[1])
s = ARGS[2]

if s == "filter_state"
    filter_state_exp(t)
elseif s == "classic_smoother_state"
    classic_smoother_state_exp(t)
elseif s == "single_iteration_smoother_state"
    single_iteration_smoother_state_exp(t)
elseif s == "iterative_smoother_state"
    iterative_smoother_state_exp(t)
end

########################################################################################################################

end
