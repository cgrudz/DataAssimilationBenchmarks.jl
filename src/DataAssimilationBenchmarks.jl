##############################################################################################
module DataAssimilationBenchmarks
##############################################################################################
# imports and exports
using LinearAlgebra
export VecA, ArView, ParamDict, ParamSample, CovM, ConM, TransM, StepKwargs

##############################################################################################
# Global type union declarations for multiple dispatch and type aliases

"""
    VecA = Union{Vector{Float64}, SubArray{Float64, 1}}

Type union of Vectors and SubArrays in order to pass columns of an ensemble maxtrix into
integration schemes and related array operations.
"""
VecA = Union{Vector{Float64}, SubArray{Float64, 1}}

"""
    ArView = Union{Array{Float64, 2}, SubArray{Float64, 2}}

Type union of arrays and views of arrays for use within ensemble conditioning operations,
integration schemes and other array operations.
"""
ArView = Union{Array{Float64, 2}, SubArray{Float64, 2}}

"""
    ParamDict = Union{Dict{String, Array{Float64}}, Dict{String, Vector{Float64}}}

Dictionary for model parameters to be passed to derivative functions by name.  This allows
one to pass both vector parameters (and scalars written as vectors), as well as matrix
valued parameters such as diffusion arrays.
"""
ParamDict = Union{Dict{String, Array{Float64}}, Dict{String, Vector{Float64}}}

"""
    ParamSample = Dict{String, Vector{UnitRange{Int64}}}

Dictionary containing key and index pairs to subset the state vector and
then merge a statistical sample of parameters that govern the equations of motion with
the ParamDict `dx_params` in parameter estimation problems.
"""
ParamSample = Dict{String, Vector{UnitRange{Int64}}}


"""
    CovM = Union{UniformScaling{Float64}, Diagonal{Float64}, Symmetric{Float64}}

Type union of covariance matrix types, for optimized computation based on their
special characteristics as symmetric, positive definite operators.
"""
CovM = Union{UniformScaling{Float64}, Diagonal{Float64}, Symmetric{Float64}}

"""
    ConM = Union{UniformScaling{Float64}, Symmetric{Float64}}

Type union of conditioning matrix types, which are used for optimization routines in the
transform method.
"""
ConM = Union{UniformScaling{Float64}, Symmetric{Float64}}

"""
    TransM = Union{Tuple{Symmetric{Float64, Array{Float64,2}}, Array{Float64,2},
                         Array{Float64,2}},
                   Tuple{Symmetric{Float64, Array{Float64,2}},
                         Array{Float64,1}, Array{Float64,2}},
                   Array{Float64,2}}

Type union of right ensemble transform types, including soley a transform,
or a transform, weights and rotation package.
"""
TransM = Union{Tuple{Symmetric{Float64,Array{Float64,2}},Array{Float64,2},Array{Float64,2}},
               Tuple{Symmetric{Float64,Array{Float64,2}},Array{Float64,1},Array{Float64,2}},
               Array{Float64,2}}

"""
    StepKwargs = Dict{String,Any}

Key word arguments for twin experiment time stepping. Arguments are given as:

```
    REQUIRED:
    dx_dt        -- time derivative function with arguments x and dx_params
    dx_params    -- parameters necessary to resolve dx_dt, not including
                    parameters to be estimated in the extended state vector 
    h            -- numerical time discretization step size
    γ            -- controls nonlinearity of the alternating_obs_operator

    OPTIONAL:
    diffusion    -- tunes the standard deviation of the Wiener process, 
                    equal to sqrt(h) * diffusion
    diff_mat     -- structure matrix for the diffusion coefficients,
                    replaces the default uniform scaling 
    state_dim    -- keyword for parameter estimation, dimension of the
                    dynamic state < dimension of full extended state
    param_sample -- ParamSample dictionary for merging extended state with dx_params
    ξ            -- random array size state_dim, can be defined in kwargs
                    to provide a particular realization for method validation
```
See [`DataAssimilationBenchmarks.EnsembleKalmanSchemes.alternating_obs_operator`](@ref) for
a discusssion of the `γ` parameter.
"""
StepKwargs = Union{Dict{String,Any}}


##############################################################################################
# imports and exports of sub-modules
include("methods/DeSolvers.jl")
include("methods/EnsembleKalmanSchemes.jl")
include("models/L96.jl")
include("models/IEEE39bus.jl")
include("experiments/GenerateTimeSeries.jl")
include("experiments/FilterExps.jl")
include("experiments/SmootherExps.jl")
include("experiments/SingleExperimentDriver.jl")
using .DeSolvers
using .EnsembleKalmanSchemes
using .L96
using .IEEE39bus
using .GenerateTimeSeries
using .FilterExps
using .SmootherExps
using .SingleExperimentDriver
export DeSolvers, EnsembleKalmanSchemes, L96, IEEE39bus, GenerateTimeSeries, FilterExps,
       SingleExperimentDriver

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
    print(" Version 0.20, Copyright 2021 Colin James Grudzien (cgrudz@mailbox.org)\n")
    print(" Licensed under the Apache License, Version 2.0 \n")
    print(" https://github.com/cgrudz/DataAssimilationBenchmarks/blob/master/LICENSE.md\n")
    print("\n")
    nothing

end

##############################################################################################

end
