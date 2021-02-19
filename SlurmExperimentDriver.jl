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
export hybrid_smoother_state_exp, iterative_smoother_state_exp

########################################################################################################################
########################################################################################################################
## Experiments to run as a single function call
########################################################################################################################
# Classic smoothers
########################################################################################################################
function classic_smoother_state_exp(j)
    f = load("/home/cgrudzien/da_benchmark/data/input_data/classic_state_smoother_input_args.jld")
    args = f["experiments"][j]
    classic_state(args)
end


########################################################################################################################
# Hybrid smoothers
########################################################################################################################
function hybrid_smoother_state_exp(j)
    f = load("/home/cgrudzien/da_benchmark/data/input_data/hybrid_state_smoother_input_args.jld")
    args = f["experiments"][j]
    hybrid_state(args)
end


########################################################################################################################
########################################################################################################################
# Iterative smoothers
########################################################################################################################
function iterative_smoother_state_exp(j)
    f = load("/home/cgrudzien/da_benchmark/data/input_data/iterative_state_smoother_input_args.jld")
    args = f["experiments"][j]
    iterative_state(args)
end


########################################################################################################################
########################################################################################################################
# Run experiments
########################################################################################################################
# comment or uncomment to run with slurm
t = parse(Int64, ARGS[1])

#classic_smoother_state_exp(t)
hybrid_smoother_state_exp(t)
#iterative_smoother_state_exp(t)

########################################################################################################################

end
