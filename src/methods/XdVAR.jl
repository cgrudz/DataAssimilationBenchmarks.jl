##############################################################################################
module XdVAR
##############################################################################################
# imports and exports
using Random, Distributions, Statistics
using LinearAlgebra, SparseArrays
using ..DataAssimilationBenchmarks
##############################################################################################
# Main methods
##############################################################################################
"""
    D3_var_cost(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
    obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
"""

function D3_var_cost(x::VecA(T), obs::VecA(T), x_background::VecA(T), state_cov::CovM(T), 
    H_obs::Function, obs_cov::CovM(T), kwargs::StepKwargs) where T <: Real
    # initializations
    obs_dim = length(obs)
    # obs operator
    H = H_obs(x,obs_dim,kwargs)

    back_component = transpose((x - x_background))*(state_cov^(-1))*(x - x_background)
    obs_component = transpose((obs - H))*(obs_cov^(-1))*(obs - H)
    J = back_component + obs_component
    return J
end


##############################################################################################
# end module

end