##############################################################################################
module DataAssimilationBenchmarks
##############################################################################################
# imports and exports
using LinearAlgebra
export VecA, ArView, ParamDict, ParamSample, CovM, ConM, TransM

##############################################################################################
# Global type union declarations for multiple dispatch and type aliases

"""
    VecA = Union{Vector{Float64}, SubArray{Float64, 1}}

Type union of vectors and ensemble members of sample for using in integration schemes and
related array operations.
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

Dictionary for model parameters to be passed to derivative functions by name.
"""
ParamDict = Union{Dict{String, Array{Float64}}, Dict{String, Vector{Float64}}}

"""
    ParamSample = Dict{String, Vector{UnitRange{Int64}}}

Dictionary containing key and index pairs to subset the state vector and
then merge with dx_params in parameter estimation problems.
"""
ParamSample = Dict{String, Vector{UnitRange{Int64}}}


"""
    CovM = Union{UniformScaling{Float64}, Diagonal{Float64}, Symmetric{Float64}}

Type union of covariance matrix types, for optimized computation based on characteristics.
"""
CovM = Union{UniformScaling{Float64}, Diagonal{Float64}, Symmetric{Float64}}

"""
    ConM = Union{UniformScaling{Float64}, Symmetric{Float64}}

Type union of conditioning matrix types, which will be used for optimization routines in the
transform method.
"""
ConM = Union{UniformScaling{Float64}, Symmetric{Float64}}

"""
    TransM = Union{Tuple{Symmetric{Float64,Array{Float64,2}},Array{Float64,2},Array{Float64,2}},
                   Tuple{Symmetric{Float64,Array{Float64,2}},Array{Float64,1},Array{Float64,2}},
                   Array{Float64,2}}

Type union of right transform types, including soley a transform, or a transform, weights
and rotation package.
"""
TransM = Union{Tuple{Symmetric{Float64,Array{Float64,2}},Array{Float64,2},Array{Float64,2}},
               Tuple{Symmetric{Float64,Array{Float64,2}},Array{Float64,1},Array{Float64,2}},
               Array{Float64,2}}



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
    printstyled(" Description\n", bold=true)
    print(" This is my personal data assimilation benchmark research code with an emphasis on testing and validation\n")
    print(" of ensemble-based filters and smoothers in chaotic toy models. DataAssimilationBenchmarks is a wrapper\n")
    print(" library including the core numerical solvers for ordinary and stochastic differential  equations,\n")
    print(" solvers for data assimilation routines and the core process model code for running twin experiments\n")
    print(" with benchmark models. These methods can be run stand-alone in other programs by calling these\n")
    print(" functions from the DeSolvers, EnsembleKalmanSchemes and L96 sub-modules from this library.\n")
    print(" Future solvers and models will be added as sub-modules in the methods and models directories respectively.\n")
    print("\n")
    print(" In order to get the full functionality of this package you you will need to install the dev version.\n")
    print(" This provides the access to edit all of the outer-loop routines for setting up twin experiments. \n")
    print(" These routines are defined in the modules in the \"experiments\" directory.  The \"slurm_submit_scripts\"\n")
    print(" directory includes routines for parallel submission of experiments in Slurm.  Data processing scripts\n")
    print(" and visualization scripts (written in Python with Matplotlib and Seaborn) are included in the \"analysis\"\n")
    print(" directory. \n")
    print(" \n")
    print(" Instructions on how to install the dev version of this package are included in the README.md\n")

    nothing

end

##############################################################################################

end
