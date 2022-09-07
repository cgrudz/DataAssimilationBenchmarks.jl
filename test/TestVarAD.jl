##############################################################################################
module TestVarAD
##############################################################################################
# imports and exports
using DataAssimilationBenchmarks.XdVAR, DataAssimilationBenchmarks.ObsOperators
using ForwardDiff, LinearAlgebra, Random, Distributions
##############################################################################################
"""
    testCost() 

    Tests the 3dVAR cost function for known behavior.
"""
function testCost()
    # initialization
    x = ones(40) * 0.5
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    params = Dict{String, Any}()
    H_obs = alternating_obs_operator

    cost = D3_var_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, params)

    if abs(cost - 10) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    testGrad() 

    Tests the gradient 3dVAR cost function for known behavior using ForwardDiff.
"""
function testGrad()
    # initialization
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    params = Dict{String, Any}()
    H_obs = alternating_obs_operator

    # wrapper function
    function wrap_cost(x)
        D3_var_cost(x, obs, x_bkg, state_cov, H_obs, obs_cov, params)
    end
    # input
    x = ones(40) * 0.5

    grad = ForwardDiff.gradient(wrap_cost, x)
    
    if norm(grad) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    testNewton() 

    Tests the Newton optimization of the 3dVAR cost function.
"""
function testNewton()
    # initialization
    obs = zeros(40)
    x_bkg = ones(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    params = Dict{String, Any}()
    H_obs = alternating_obs_operator

    # perform Simple Newton optimization
    op = D3_var_NewtonOp(x_bkg, obs, state_cov, H_obs, obs_cov, params)
  
    if abs(sum(op - ones(40) * 0.5)) < 0.001
        true
    else
        false
    end
end


##############################################################################################
"""
    testNewtonNoise() 

    Tests the Newton optimization of the 3dVAR cost function with noise.
"""
function testNewtonNoise()
    # initialization
    Random.seed!(123)
    obs = rand(Normal(0, 1), 40)
    x_bkg = zeros(40)
    state_cov = 1.0I
    obs_cov = 1.0I
    params = Dict{String, Any}()
    H_obs = alternating_obs_operator

    # perform Simple Newton optimization
    op = D3_var_NewtonOp(x_bkg, obs, state_cov, H_obs, obs_cov, params)
    
    if abs(sum(op - obs * 0.5)) < 0.001
        true
    else
        false
    end
end


##############################################################################################
# end module

end
