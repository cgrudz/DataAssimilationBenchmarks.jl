##############################################################################################
module DataAssimilationBenchmarks
##############################################################################################
# imports and exports
using LinearAlgebra
export VecA, ArView, ParamDict, ParamSample, CovM, ConM, TransM, StepKwargs

##############################################################################################
# Global type union declarations for multiple dispatch and type aliases

"""
    function VecA(type)
        Union{Vector{T}, SubArray{T, 1}} where T <: type
    end

Type constructor for union of Vectors and 1-D SubArrays.  This is utilzed  in order to pass
columns of an ensemble maxtrix into integration schemes and related array operations.
"""
function VecA(type)
    Union{Vector{T}, SubArray{T, 1}} where T <: type
end

"""
    function ArView(type)
        Union{Array{T, 2}, SubArray{T, 2}} where T <: type
    end

Type constructor for union of Arrays and SubArrays for use within ensemble conditioning
operations, integration schemes and other array operations.
"""
function ArView(type)
    Union{Array{T, 2}, SubArray{T, 2}} where T <: type
end

"""
    function ParamDict(type)
        Union{Dict{String, Array{T}}, Dict{String, Vector{T}}} where T <: type
    end

Type constructor for Dictionary of model parameters to be passed to derivative functions
by name.  This allows one to pass both vector parameters (and scalars written as
vectors), as well as matrix valued parameters such as diffusion arrays.
"""
function ParamDict(type)
    Union{Dict{String, Array{T}}, Dict{String, Vector{T}}} where T <: type
end

"""
    ParamSample = Dict{String, Vector{UnitRange{Int64}}}

Dictionary containing key and index pairs to subset the state vector and
then merge a statistical sample of parameters that govern the equations of motion with
the ParamDict `dx_params` in parameter estimation problems.
"""
ParamSample = Dict{String, Vector{UnitRange{Int64}}}


"""
    function CovM(type)
        Union{UniformScaling{T}, Diagonal{T, Vector{T}},
              Symmetric{T, Matrix{T}}} where T <: type
    end

Type constructor for union of covariance matrix types, for multiple dispatch
based on their special characteristics as symmetric, positive definite operators.
"""
function CovM(type)
    Union{UniformScaling{T}, Diagonal{T, Vector{T}},
          Symmetric{T, Matrix{T}}} where T <: type
end

"""
    function ConM(type)
        Union{UniformScaling{T}, Symmetric{T}} where T <: type
    end

Type union of conditioning matrix types, which are used for optimization routines in the
transform method.
"""
function ConM(type)
    Union{UniformScaling{T}, Symmetric{T}} where T <: type
end

"""
    function TransM(type)
        Union{Tuple{Symmetric{T,Array{T,2}},Array{T,1},Array{T,2}},
              Tuple{Symmetric{T,Array{T,2}},Array{T,2},Array{T,2}}} where T <: type
    end

Type union constructor for tuples representing the ensemble update step with a right
ensemble anomaly transformation, mean update weights and mean-preserving orthogonal
transformation.
"""
function TransM(type)
    Union{Tuple{Symmetric{T,Array{T,2}},Array{T,1},Array{T,2}},
          Tuple{Symmetric{T,Array{T,2}},Array{T,2},Array{T,2}}} where T <: type
end

"""
    StepKwargs = Dict{String,Any}

Key word arguments for twin experiment time stepping. Arguments are given as:

REQUIRED:
  * `dx_dt`     - time derivative function with arguments x and dx_params
  * `dx_params` - parameters necessary to resolve dx_dt, not including parameters to be estimated in the extended state vector;
  * `h` - numerical time discretization step size

OPTIONAL:
  * `diffusion` - tunes the standard deviation of the Wiener process, equal to `sqrt(h) * diffusion`;
  * `diff_mat` - structure matrix for the diffusion coefficients, replaces the default uniform scaling;
  * `s_infl` - ensemble anomalies of state components are scaled by this parameter for calculation of emperical covariance;
  * `p_infl` - ensemble anomalies of extended-state components for parameter sample replicates are scaled by this parameter for calculation of emperical covariance, `state_dim` must be defined below;
  * `state_dim` - keyword for parameter estimation, specifying the dimension of the dynamic state, less than the dimension of full extended state;
  * `param_sample` - `ParamSample` dictionary for merging extended state with `dx_params`;
  * `ξ` - random array size `state_dim`, can be defined in `kwargs` to provide a particular realization for method validation;
  * `γ` - controls nonlinearity of the alternating_obs_operatori.

See [`DataAssimilationBenchmarks.ObsOperators.alternating_obs_operator`](@ref) for
a discusssion of the `γ` parameter.
"""
StepKwargs = Union{Dict{String,Any}}


##############################################################################################
# imports and exports of sub-modules
include("methods/DeSolvers.jl")
include("methods/EnsembleKalmanSchemes.jl")
include("methods/XdVAR.jl")
include("models/L96.jl")
include("models/IEEE39bus.jl")
include("models/ObsOperators.jl")
include("experiments/GenerateTimeSeries.jl")
include("experiments/FilterExps.jl")
include("experiments/SmootherExps.jl")
include("experiments/SingleExperimentDriver.jl")
include("experiments/ParallelExperimentDriver.jl")
include("experiments/D3VARExps.jl")
include("experiments/VarAnalysisExperimentDriver.jl")
using .DeSolvers
using .EnsembleKalmanSchemes
using .L96
using .IEEE39bus
using .ObsOperators
using .GenerateTimeSeries
using .FilterExps
using .SmootherExps
using .SingleExperimentDriver
using .ParallelExperimentDriver
using .D3VARExps
export DeSolvers, EnsembleKalmanSchemes, XdVAR, L96, IEEE39bus, ObsOperators,
       GenerateTimeSeries, FilterExps, SingleExperimentDriver, ParallelExperimentDriver,
       D3VARExps
##############################################################################################
# info

function Info()
    print("  _____        _                         ")
    printstyled("_",color=9)
    print("           ")
    printstyled("_",color=2)
    print(" _       _   ")
    printstyled("_              \n",color=13)
    print(" |  __ \\      | |          /\\           ")
    printstyled("(_)",color=9)
    printstyled("         (_)",color=2)
    print(" |     | | ")
    printstyled("(_)             \n",color=13)
    print(" | |  | | __ _| |_ __ _   /  \\   ___ ___ _ _ __ ___  _| | __ _| |_ _  ___  _ __   \n")
    print(" | |  | |/ _` | __/ _` | / /\\ \\ / __/ __| | '_ ` _ \\| | |/ _` | __| |/ _ \\| '_ \\  \n")
    print(" | |__| | (_| | || (_| |/ ____ \\\\__ \\__ \\ | | | | | | | | (_| | |_| | (_) | | | | \n")
    print(" |_____/ \\__,_|\\__\\__,_/_/    \\_\\___/___/_|_| |_| |_|_|_|\\__,_|\\__|_|\\___/|_| |_| \n")
    print("\n")
    print("  ____                  _                          _         ")
    printstyled(" _ ", color=12)
    print("_\n")
    print(" |  _ \\                | |                        | |        ")
    printstyled("(_)",color=12)
    print(" |                \n")
    print(" | |_) | ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____   _| |                \n")
    print(" |  _ < / _ \\ '_ \\ / __| '_ \\| '_ ` _ \\ / _` | '__| |/ / __| | | |                \n")
    print(" | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\\__ \\_| | |                \n")
    print(" |____/ \\___|_| |_|\\___|_| |_|_| |_| |_|\\__,_|_|  |_|\\_\\___(_) |_|                \n")
    print("                                                            _/ |                  \n")
    print("                                                           |__/                   \n")

    print("\n")
    printstyled(" Welcome to DataAssimilationBenchmarks!\n", bold=true)
    print(" Version 0.20, Copyright 2022 Colin Grudzien (cgrudzien@ucsd.edu) et al.\n")
    print(" Licensed under the Apache License, Version 2.0 \n")
    print(" https://github.com/cgrudz/DataAssimilationBenchmarks/blob/master/LICENSE.md\n")
    print("\n")
    nothing

end

##############################################################################################

end
