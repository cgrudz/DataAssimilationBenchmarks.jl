##############################################################################################
module SlurmExperimentDriver 
##############################################################################################
##############################################################################################
# imports and exports
using Random, Distributions
using Debugger
using Distributed
using LinearAlgebra
using JLD2
using DeSolvers
using L96 
using EnsembleKalmanSchemes
using FilterExps
using SmootherExps

##############################################################################################
# Definition of globals

path = joinpath(@__DIR__, "../data/") 


##############################################################################################
## Experiments to run as a single function call
##############################################################################################
# Filters 
##############################################################################################

function filter_state_exp(j)
    f = load(path * "input_data/filter_state_input_args.jld2")
    args = f["experiments"][j]
    filter_state(args)
end


##############################################################################################
# Classic smoothers
##############################################################################################

function classic_smoother_state_exp(j)
    f = load(path * "input_data/classic_state_smoother_input_args.jld2")
    args = f["experiments"][j]
    classic_state(args)
end

function classic_smoother_param_exp(j)
    f = load(path * "input_data/classic_param_smoother_input_args.jld2")
    args = f["experiments"][j]
    classic_param(args)
end


##############################################################################################
# Single-iteration smoothers
##############################################################################################

function single_iteration_smoother_state_exp(j)
    f = load(path * "input_data/single_iteration_state_smoother_input_args.jld2")
    args = f["experiments"][j]
    single_iteration_state(args)
end

function single_iteration_smoother_param_exp(j)
    f = load(path * "input_data/single_iteration_param_smoother_input_args.jld2")
    args = f["experiments"][j]
    single_iteration_param(args)
end


##############################################################################################
# Iterative smoothers
##############################################################################################

function iterative_smoother_state_exp(j)
    f = load("../data/input_data/iterative_state_smoother_input_args.jld2")
    args = f["experiments"][j]
    iterative_state(args)
end

function iterative_smoother_param_exp(j)
    f = load("../data/input_data/iterative_param_smoother_input_args.jld2")
    args = f["experiments"][j]
    iterative_param(args)
end


##############################################################################################
# Run experiments
##############################################################################################

t = parse(Int64, ARGS[1])
s = ARGS[2]

if s == "filter_state"
    filter_state_exp(t)
elseif s == "classic_smoother_state"
    classic_smoother_state_exp(t)
elseif s == "classic_smoother_param"
    classic_smoother_param_exp(t)
elseif s == "single_iteration_smoother_state"
    single_iteration_smoother_state_exp(t)
elseif s == "single_iteration_smoother_param"
    single_iteration_smoother_param_exp(t)
elseif s == "iterative_smoother_state"
    iterative_smoother_state_exp(t)
elseif s == "iterative_smoother_param"
    iterative_smoother_param_exp(t)
end

##############################################################################################
# end module

end
